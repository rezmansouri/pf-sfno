import os
import sys
import json
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import SequentialSampler, DataLoader

from model import MultiModalSFNO
from utils import (
    SphericalNOTestDataset,
    seed_everything,
    inverse_transform_model_batch,
)
from training_utils import _build_grid_embeddings

fields = ["vr"]


def main():

    simulation_path, results_path, training_dataset_stats_path = sys.argv[1:4]

    seed_everything(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using device:", device)

    # ------------------------------------------------------------
    # Load configuration
    # ------------------------------------------------------------

    cfg_path = os.path.join(results_path, "cfg.json")

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    if "feed_grid_embeddings" not in cfg:
        cfg["feed_grid_embeddings"] = True

    # print("cfg:")
    # print(json.dumps(cfg, indent=4))

    # ------------------------------------------------------------
    # Load training statistics
    # ------------------------------------------------------------

    with open(training_dataset_stats_path, "r", encoding="utf-8") as f:
        train_stats = json.load(f)

    for field in train_stats:
        train_stats[field]["mean"] = np.array(train_stats[field]["mean"])
        train_stats[field]["std"] = np.array(train_stats[field]["std"])

    # ------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------

    test_dataset = SphericalNOTestDataset(
        [simulation_path], stats=train_stats, system="cgs"
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        sampler=SequentialSampler(test_dataset),
        pin_memory=True,
        num_workers=0,
    )

    print("Number of simulations:", len(test_dataset))
    print("Nr:", test_dataset.Nr)

    # ------------------------------------------------------------
    # Grid embeddings
    # ------------------------------------------------------------

    ang_emb = [t.to(device) for t in test_dataset.angular_embeddings]

    rad_emb = test_dataset.radial_embeddings.to(device)

    theta = test_dataset.theta.to(device)
    phi = test_dataset.phi.to(device)
    r = test_dataset.r.to(device)

    # ------------------------------------------------------------
    # Model
    # ------------------------------------------------------------

    model = MultiModalSFNO(
        n_modalities=1,
        in_channels=1,
        out_channels=cfg["prediction_horizon"],
        hidden_channels=cfg["hidden_channels"],
        n_modes=(110, 128),
        n_layers=cfg["n_layers"],
        lifting_channel_ratio=cfg["lift_project_ratio"],
        projection_channel_ratio=cfg["lift_project_ratio"],
        grid_embedding_dim=(4 if cfg["feed_grid_embeddings"] else None),
    ).to(device)

    checkpoint_path = os.path.join(
        results_path,
        "model.pt",
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint,
        strict=False,
    )

    model.eval()

    print("Loaded model:", checkpoint_path)

    # ------------------------------------------------------------
    # Autoregressive prediction
    # ------------------------------------------------------------

    predictions_list = []
    targets_list = []

    with torch.no_grad():

        for batch in tqdm(
            test_loader,
            desc="Predicting",
            leave=True,
        ):

            state = (
                torch.stack(
                    [batch[f"{f}_0"] for f in fields],
                    dim=1,
                )
                .unsqueeze(2)
                .to(device)
            )

            targets = torch.stack(
                [batch[f"{f}_1_139"] for f in fields],
                dim=1,
            ).to(device)

            B = state.shape[0]
            Nr = test_dataset.Nr

            current_state = state
            current_radius = 0

            blocks = []

            while current_radius < Nr - 1:

                if cfg["feed_grid_embeddings"]:
                    grid_embeddings = _build_grid_embeddings(
                        current_radius,
                        B,
                        rad_emb,
                        ang_emb,
                        device,
                    )
                else:
                    grid_embeddings = None

                pred = model(
                    current_state,
                    grid_embeddings=grid_embeddings,
                )

                remaining = Nr - 1 - current_radius

                keep = min(
                    cfg["prediction_horizon"],
                    remaining,
                )

                prediction = pred[:, :, :keep]

                blocks.append(prediction)

                # Last predicted radius becomes the input
                # for the next autoregressive block.
                current_state = pred[:, :, keep - 1 : keep]

                current_radius += keep

            predictions = torch.cat(
                blocks,
                dim=2,
            )

            # Convert both prediction and target back to original
            # physical units.
            predictions_cgs = inverse_transform_model_batch(
                predictions,
                train_stats,
                drop_first_radius=True,
            )

            targets_cgs = inverse_transform_model_batch(
                targets,
                train_stats,
                drop_first_radius=True,
            )

            predictions_list.append(predictions_cgs.cpu().numpy())

            targets_list.append(targets_cgs.cpu().numpy())

    # ------------------------------------------------------------
    # Combine predictions and targets
    # ------------------------------------------------------------

    predictions = np.concatenate(
        predictions_list,
        axis=0,
    )

    target = np.concatenate(
        targets_list,
        axis=0,
    )

    print("Prediction shape:", predictions.shape)
    print("Target shape:", target.shape)

    # ------------------------------------------------------------
    # Save
    # ------------------------------------------------------------

    output_path = os.path.join(
        results_path,
        "predictions",
    )

    os.makedirs(output_path, exist_ok=True)

    output_path = os.path.join(
        results_path,
        "predictions",
        "_".join(simulation_path.split("/")[-2:]) + ".npz",
    )

    np.savez_compressed(
        output_path,
        predictions=predictions,
        target=target,
    )

    print("Saved prediction and target to:")
    print(output_path)


if __name__ == "__main__":
    main()
