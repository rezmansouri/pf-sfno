import os
import sys
import numpy as np
import torch
import json
import pandas as pd
from training_utils import train
import torch.distributed as dist
from neuralop.losses import H1Loss
from model import MultiModalSFNO
from utils import (
    SphericalNOTestDataset,
    SphericalNOTrainDataset,
    get_cr_dirs,
    collect_sim_paths,
    setup_ddp,
    seed_everything,
)
from torch.utils.data import RandomSampler, SequentialSampler
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler


def main():
    (
        data_path,
        split_csv_path,
        training_dataset_stats_path,
        batch_size,
        n_epochs,
        lift_project_ratio,
        hidden_channels,
        n_layers,
        prediction_horizon,
        warmup_pf_epochs,
        p_push_forward,
        save_test_preds,
        feed_grid_embeddings,
        radial_pushforward,
    ) = sys.argv[1:15]

    local_rank, global_rank = setup_ddp()

    print(
        f"[setup] local_rank={local_rank} global_rank={global_rank} "
        f"world_size={dist.get_world_size() if dist.is_initialized() else 1}",
        flush=True,
    )

    seed_everything(42 + global_rank)

    is_distributed = dist.is_available() and dist.is_initialized()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    print("using device:", device)

    (
        batch_size,
        lift_project_ratio,
        n_epochs,
        hidden_channels,
        n_layers,
        prediction_horizon,
        warmup_pf_epochs,
        p_push_forward,
        save_test_preds,
        feed_grid_embeddings,
        radial_pushforward,
    ) = (
        int(batch_size),
        int(lift_project_ratio),
        int(n_epochs),
        int(hidden_channels),
        int(n_layers),
        int(prediction_horizon),
        int(warmup_pf_epochs),
        float(p_push_forward),
        save_test_preds.lower() == "true",
        feed_grid_embeddings.lower() == "true",
        radial_pushforward.lower() == "true",
    )

    split_df = pd.read_csv(split_csv_path)

    print(
        f"[setup] local_rank={local_rank} global_rank={global_rank} "
        f"world_size={dist.get_world_size() if dist.is_initialized() else 1}",
        flush=True,
    )

    cr_train = split_df.loc[split_df["Split"] == "train", "FolderName"].tolist()

    cr_val = split_df.loc[split_df["Split"] == "val", "FolderName"].tolist()

    cr_test = split_df.loc[split_df["Split"] == "test", "FolderName"].tolist()

    # Keep only folders that actually exist
    available_crs = set(get_cr_dirs(data_path))

    cr_train = sorted(
        [cr for cr in cr_train if cr in available_crs], key=lambda x: int(x[2:])
    )
    cr_val = sorted(
        [cr for cr in cr_val if cr in available_crs], key=lambda x: int(x[2:])
    )
    cr_test = sorted(
        [cr for cr in cr_test if cr in ["cr1626", "cr1815"]], key=lambda x: int(x[2:])
    )

    print(f"Train CRs: {len(cr_train)}")
    print(f"Val CRs:   {len(cr_val)}")
    print(f"Test CRs:  {len(cr_test)}")

    train_paths = collect_sim_paths(data_path, cr_train)
    val_paths = collect_sim_paths(data_path, cr_val)
    test_paths = collect_sim_paths(data_path, cr_test)

    print(f"Train simulations: {len(train_paths)}")
    print(f"Val simulations:   {len(val_paths)}")
    print(f"Test simulations:  {len(test_paths)}")

    if training_dataset_stats_path.lower() == "none":
        train_stats = None
    else:
        with open(training_dataset_stats_path, "r", encoding="utf-8") as f:
            train_stats = json.load(f)
            for field in train_stats:
                train_stats[field]["mean"] = np.array(train_stats[field]["mean"])
                train_stats[field]["std"] = np.array(train_stats[field]["std"])

    train_dataset = SphericalNOTrainDataset(train_paths, stats=train_stats)
    train_stats = train_dataset.get_stats()

    val_dataset = SphericalNOTestDataset(val_paths, stats=train_stats)

    test_dataset = SphericalNOTestDataset(test_paths, stats=train_stats)

    if is_distributed:
        train_sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=True)
        val_sampler = DistributedSampler(val_dataset, shuffle=False, drop_last=True)
    else:
        train_sampler = RandomSampler(train_dataset)
        val_sampler = SequentialSampler(val_dataset)

    test_sampler = SequentialSampler(test_dataset)

    if prediction_horizon == 1:
        loss_fn = H1Loss(
            d=2,
            periodic_in_x=False,  # theta: non-periodic, one-sided FD at poles
            periodic_in_y=True,  # phi:   periodic, torch.roll wrapping
        )
    else:
        loss_fn = H1Loss(
            d=3,
            periodic_in_x=False,  # r:   non-periodic, one-sided FD
            periodic_in_y=False,  # theta: non-periodic, one-sided FD at poles
            periodic_in_z=True,  # phi:   periodic, torch.roll wrapping
        )

    stats = {}
    for field, field_stats in train_stats.items():

        stats[field] = {}

        for key, value in field_stats.items():

            if isinstance(value, np.ndarray):
                stats[field][key] = value.tolist()

            elif isinstance(value, np.generic):
                stats[field][key] = value.item()

            else:
                stats[field][key] = value

    cfg = {
        "num_epochs": n_epochs,
        "batch_size": batch_size,
        "learning_rate": 8e-4,
        "train_files": cr_train,
        "val_files": cr_val,
        "hidden_channels": hidden_channels,
        "n_layers": n_layers,
        "lift_project_ratio": lift_project_ratio,
        "prediction_horizon": prediction_horizon,
        "warmup_pf_epochs": warmup_pf_epochs,
        "p_push_forward": p_push_forward,
        "feed_grid_embeddings": feed_grid_embeddings,
        "radial_pushforward": radial_pushforward,
    }

    out_path = f"n_layers-{n_layers}_hidden_channels-{hidden_channels}-lift_project_ratio-{lift_project_ratio}_prediction_horizon-{prediction_horizon}_warmup_pf_epochs-{warmup_pf_epochs}_p_push_forward-{p_push_forward}_feed_grid_embeddings-{feed_grid_embeddings}_radial_pushforward-{radial_pushforward}"

    if global_rank == 0:
        os.makedirs(out_path, exist_ok=True)
        if save_test_preds:
            os.makedirs(os.path.join(out_path, "test_preds"), exist_ok=True)
        with open(os.path.join(out_path, "cfg.json"), "w", encoding="utf-8") as f:
            json.dump(cfg, f)

    if is_distributed:
        dist.barrier()

    model = MultiModalSFNO(
        n_modalities=1,
        in_channels=1,
        out_channels=prediction_horizon,
        hidden_channels=hidden_channels,
        n_modes=(110, 128),
        n_layers=n_layers,
        lifting_channel_ratio=lift_project_ratio,
        projection_channel_ratio=lift_project_ratio,
        grid_embedding_dim=4 if feed_grid_embeddings else None,
    ).to(device)

    model = (
        DistributedDataParallel(model, device_ids=[local_rank])
        if is_distributed
        else model
    )

    (
        train_losses,
        val_losses,
        best_epoch,
        best_state_dict,
    ) = train(
        model,
        train_dataset,
        val_dataset,
        test_dataset,
        train_sampler,
        val_sampler,
        test_sampler,
        global_rank=global_rank,
        n_epochs=n_epochs,
        batch_size=batch_size,
        loss_fn=loss_fn,
        device=device,
        lr=8e-4,
        weight_decay=0.0,
        prediction_horizon=prediction_horizon,
        warmup_pf_epochs=warmup_pf_epochs,
        p_push_forward=p_push_forward,
        save_path=f"{out_path}/test_preds" if save_test_preds else None,
        feed_grid_embeddings=feed_grid_embeddings,
        radial_pushforward=radial_pushforward,
    )

    if global_rank == 0:
        torch.save(best_state_dict, os.path.join(out_path, "model.pt"))
        with open(
            os.path.join(out_path, f"best_epoch-{best_epoch}.txt"),
            "w",
            encoding="utf-8",
        ) as f:
            f.write(f"best_epoch: {best_epoch}")
        np.save(os.path.join(out_path, "train_losses.npy"), train_losses)
        np.save(os.path.join(out_path, "val_losses.npy"), val_losses)

    print("Training completed.")

    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
