from torch.utils.data import DataLoader
import torch
import math
from copy import deepcopy
from tqdm import tqdm
import torch.distributed as dist
import torch.optim as optim
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Curriculum: ramp p_push from 0 → target over warmup_pf_epochs
# then hold.  This prevents the model diverging before it has a decent
# single-step mapping.
# ─────────────────────────────────────────────────────────────────────────────


def _p_push_schedule(
    epoch: int,
    target: float,
    warmup_epochs: int,
    current_radius: int = None,
    Nr: int = None,
    steepness: float = 8.0,
) -> float:
    # Epoch-level ramp: 0 → 1 over warmup_epochs
    if warmup_epochs <= 0:
        p_epoch = 1.0
    else:
        p_epoch = min(1.0, epoch / warmup_epochs)

    # No radial info: old behaviour preserved exactly
    if current_radius is None or Nr is None:
        return target * p_epoch

    # Sigmoid anchored so that at the last block p_radial = target exactly
    def sig(x):
        return 1.0 / (1.0 + math.exp(-steepness * (x - 0.5)))

    radial_fraction = current_radius / (Nr - 1)  # 0.0 → 1.0
    p_radial = target * sig(radial_fraction) / sig(1.0)

    return p_epoch * p_radial


def _build_grid_embeddings(
    current_radius, batch_size, radial_embeddings, angular_embeddings, device
):
    return torch.cat(
        [
            radial_embeddings[current_radius]
            .unsqueeze(0)
            .expand(batch_size, -1, -1)
            .unsqueeze(1),
            angular_embeddings[0].unsqueeze(0).expand(batch_size, -1, -1).unsqueeze(1),
            angular_embeddings[1].unsqueeze(0).expand(batch_size, -1, -1).unsqueeze(1),
            angular_embeddings[2].unsqueeze(0).expand(batch_size, -1, -1).unsqueeze(1),
        ],
        dim=1,
    )


def train(
    model,
    train_dataset,
    val_dataset,
    test_dataset,
    train_sampler,
    val_sampler,
    test_sampler,
    global_rank,
    n_epochs,
    batch_size,
    loss_fn,
    device,
    save_path=None,
    prediction_horizon: int = 5,
    p_push_forward: float = 0.7,
    warmup_pf_epochs: int = None,
    lr: float = 8e-4,
    weight_decay: float = 1e-4,
    grad_clip: float = 1.0,
    lr_patience: int = 10,
    lr_factor: float = 0.5,
    lr_min: float = 1e-6,
    patience: int = 30,
    verbose: bool = True,
    feed_grid_embeddings: bool = True,
    radial_pushforward: bool = True,
):
    fields = ["vr"]

    if warmup_pf_epochs is None:
        warmup_pf_epochs = max(1, int(0.20 * n_epochs))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        pin_memory=True,
        num_workers=0,
        # persistent_workers=True, # Avoids worker respawn overhead
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        pin_memory=True,
        num_workers=0,
        # persistent_workers=True, # Avoids worker respawn overhead
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        sampler=test_sampler,
        pin_memory=True,
        num_workers=0,
        # persistent_workers=True, # Avoids worker respawn overhead
    )

    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.95),
    )

    warmup_epochs = max(1, int(0.1 * n_epochs))

    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_epochs,
    )
    plateau_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=lr_patience,
        min_lr=lr_min,
    )

    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    best_epoch = -1
    best_state_dict = None
    no_improve = 0

    ang_emb = [t.to(device) for t in train_dataset.angular_embeddings]
    rad_emb = train_dataset.radial_embeddings.to(device)

    is_distributed = dist.is_available() and dist.is_initialized()

    for epoch in range(n_epochs):
        p_push_current = _p_push_schedule(epoch, p_push_forward, warmup_pf_epochs)
        num_train_samples = 0

        # ══════════════════════════════════════════════════════════════════════
        # TRAIN
        # ══════════════════════════════════════════════════════════════════════
        model.train()
        running_train_loss = 0.0

        if hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)  # For shuffling in distributed training

        for batch in tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{n_epochs} [Train | p_pf={p_push_current:.2f}]",
            leave=False,
        ):
            optimizer.zero_grad(set_to_none=True)

            x = (
                torch.stack([batch[f"{f}_0"] for f in fields], dim=1)
                .unsqueeze(2)
                .to(device)
            )
            y = torch.stack([batch[f"{f}_1_139"] for f in fields], dim=1).to(device)

            B = x.shape[0]
            num_train_samples += B
            Nr = train_dataset.Nr

            pushed_state = x
            current_radius = 0
            n_blocks = 0
            accumulated_loss = 0.0

            while current_radius < Nr - 1:
                if radial_pushforward:
                    p_push_current = _p_push_schedule(
                        epoch, p_push_forward, warmup_pf_epochs, current_radius, Nr
                    )
                if current_radius == 0:
                    model_input = x
                elif torch.rand(1).item() < p_push_current:
                    model_input = pushed_state  # detached — no grad through input
                else:
                    model_input = y[:, :, current_radius - 1 : current_radius]

                if feed_grid_embeddings:
                    grid_embeddings = _build_grid_embeddings(
                        current_radius, B, rad_emb, ang_emb, device
                    )
                else:
                    grid_embeddings = None
                pred = model(model_input, grid_embeddings=grid_embeddings)

                remaining = Nr - 1 - current_radius
                keep = min(prediction_horizon, remaining)
                target_block = y[:, :, current_radius : current_radius + keep]
                pred_block = pred[:, :, :keep]

                block_loss = loss_fn(pred_block, target_block)

                # Single block loss backward pass
                block_loss.backward()
                accumulated_loss += block_loss.item() * keep

                # Multi-block loss backward pass
                # accumulated_loss += block_loss * keep

                n_blocks += 1
                pushed_state = pred[:, :, keep - 1 : keep].detach()
                current_radius += keep

            # Multi-block loss backward pass
            mean_block_loss = accumulated_loss / n_blocks
            # mean_block_loss.backward()
            # mean_block_loss = mean_block_loss.item()  # Convert to scalar for logging

            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            running_train_loss += mean_block_loss * B

        train_loss_tensor = torch.tensor(running_train_loss, device=device)

        num_train_tensor = torch.tensor(num_train_samples, device=device)

        if is_distributed:
            dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_train_tensor, op=dist.ReduceOp.SUM)

        epoch_train_loss = train_loss_tensor.item() / num_train_tensor.item()
        train_losses.append(epoch_train_loss)

        # ══════════════════════════════════════════════════════════════════════
        # VALIDATION
        # ══════════════════════════════════════════════════════════════════════
        model.eval()
        running_val_loss = 0.0
        num_val_samples = 0

        if hasattr(val_sampler, "set_epoch"):
            val_sampler.set_epoch(epoch)

        with torch.no_grad():
            for batch in tqdm(
                val_loader, desc=f"Epoch {epoch+1}/{n_epochs} [Val]", leave=False
            ):
                state = (
                    torch.stack([batch[f"{f}_0"] for f in fields], dim=1)
                    .unsqueeze(2)
                    .to(device)
                )
                targets = torch.stack([batch[f"{f}_1_139"] for f in fields], dim=1).to(
                    device
                )

                B = state.shape[0]
                num_val_samples += B
                Nr = val_dataset.Nr

                current_state = state
                current_radius = 0
                n_blocks = 0
                accumulated_loss = 0.0

                while current_radius < Nr - 1:

                    if feed_grid_embeddings:
                        grid_embeddings = _build_grid_embeddings(
                            current_radius, B, rad_emb, ang_emb, device
                        )
                    else:
                        grid_embeddings = None
                    pred = model(current_state, grid_embeddings=grid_embeddings)
                    remaining = Nr - 1 - current_radius
                    keep = min(prediction_horizon, remaining)

                    prediction = pred[:, :, :keep]
                    target = targets[:, :, current_radius : current_radius + keep]
                    accumulated_loss += loss_fn(prediction, target) * keep
                    n_blocks += 1

                    current_state = pred[:, :, keep - 1 : keep]
                    current_radius += keep

                # Single block loss
                mean_block_loss = accumulated_loss / n_blocks
                running_val_loss += mean_block_loss.item() * B

        val_loss_tensor = torch.tensor(running_val_loss, device=device)

        num_val_tensor = torch.tensor(num_val_samples, device=device)

        if is_distributed:

            dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_val_tensor, op=dist.ReduceOp.SUM)

        epoch_val_loss = val_loss_tensor.item() / num_val_tensor.item()
        val_losses.append(epoch_val_loss)

        if epoch < warmup_epochs:
            warmup_scheduler.step()
        else:
            plateau_scheduler.step(epoch_val_loss)

        # scheduler.step(epoch_val_loss) no need to step scheduler here since we are using OneCycleLR

        if save_path is not None and global_rank == 0:
            model.eval()
            test_predictions = []
            with torch.no_grad():
                for batch in tqdm(
                    test_loader, desc=f"Epoch {epoch+1}/{n_epochs} [Test]", leave=False
                ):
                    state = (
                        torch.stack([batch[f"{f}_0"] for f in fields], dim=1)
                        .unsqueeze(2)
                        .to(device)
                    )
                    B = state.shape[0]
                    Nr = test_dataset.Nr

                    current_state = state
                    current_radius = 0
                    predictions = []

                    while current_radius < Nr - 1:
                        if feed_grid_embeddings:
                            grid_embeddings = _build_grid_embeddings(
                                current_radius, B, rad_emb, ang_emb, device
                            )
                        else:
                            grid_embeddings = None
                        pred = model(current_state, grid_embeddings=grid_embeddings)
                        remaining = Nr - 1 - current_radius
                        keep = min(prediction_horizon, remaining)

                        predictions.append(pred[:, :, :keep].cpu())
                        current_state = pred[:, :, keep - 1 : keep]
                        current_radius += keep

                    test_predictions.append(torch.cat(predictions, dim=2))

            test_predictions = torch.cat(test_predictions, dim=0)
            print(test_predictions.shape)
            np.savez_compressed(
                f"{save_path}/epoch_{epoch+1}.npz",
                predictions=test_predictions.numpy(),
            )

        if verbose and global_rank == 0:
            print(
                f"Epoch {epoch+1:>4}/{n_epochs} | "
                f"p_pf={p_push_current:.2f} | "
                f"Train={epoch_train_loss:.4e} | "
                f"Val={epoch_val_loss:.4e} | "
                f"LR={optimizer.param_groups[0]['lr']:.2e}"
            )

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_epoch = epoch
            no_improve = 0
            if global_rank == 0:
                state = (
                    model.module.state_dict()
                    if hasattr(model, "module")
                    else model.state_dict()
                )
                best_state_dict = deepcopy(state)
            else:
                best_state_dict = None
            if verbose and global_rank == 0:
                print(f"  ✓ New best val loss: {best_val_loss:.4e}")
        else:
            no_improve += 1

        stop_flag = torch.tensor(int(no_improve >= patience), device=device)
        if is_distributed:
            dist.broadcast(stop_flag, src=0)

        if stop_flag.item():
            if verbose and global_rank == 0:
                print(
                    f"\nEarly stopping at epoch {epoch+1} (no improvement for {patience} epochs)."
                )
            break

    if best_state_dict is not None:
        target = model.module if hasattr(model, "module") else model
        target.load_state_dict(best_state_dict)
        print(
            f"\nRestored best weights from epoch {best_epoch+1} (val loss={best_val_loss:.4e})."
        )
    if is_distributed:
        dist.barrier()

    return train_losses, val_losses, best_epoch, best_state_dict
