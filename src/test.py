import os
import sys
import numpy as np
import torch
import json
import pandas as pd
from tqdm import tqdm
from model import MultiModalSFNO
from utils import (
    SphericalNOTestDataset,
    get_cr_dirs,
    collect_sim_paths,
    seed_everything,
    inverse_transform_model_batch,
)
from torch.utils.data import SequentialSampler, DataLoader
from training_utils import _build_grid_embeddings
from metrics import *

fields = ["vr"]


def main():
    (
        data_path,
        split_csv_path,
        training_dataset_stats_path,
        climatologies_path,
        results_path,
        batch_size,
        single_instrument,
        multiply_kmps,
    ) = sys.argv[1:9]

    multiply_kmps = multiply_kmps.lower() == "true"

    seed_everything(42)

    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

    print("using device:", device)

    batch_size = int(batch_size)

    split_df = pd.read_csv(split_csv_path)

    single_instrument = single_instrument.lower() == "true"

    cfg_path = os.path.join(results_path, "cfg.json")

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    if "feed_grid_embeddings" not in cfg:
        cfg["feed_grid_embeddings"] = True

    cr_test = split_df.loc[split_df["Split"] == "test", "FolderName"].tolist()

    # Keep only folders that actually exist
    available_crs = set(get_cr_dirs(data_path))

    cr_test = sorted(
        [cr for cr in cr_test if cr in available_crs], key=lambda x: int(x[2:])
    )

    print(f"Test CRs:  {len(cr_test)}")

    test_paths = collect_sim_paths(
        data_path, cr_test, single_instrument=single_instrument
    )

    print(f"Test simulations:  {len(test_paths)}")

    with open(training_dataset_stats_path, "r", encoding="utf-8") as f:
        train_stats = json.load(f)
        for field in train_stats:
            train_stats[field]["mean"] = np.array(train_stats[field]["mean"])
            train_stats[field]["std"] = np.array(train_stats[field]["std"])

    test_dataset = SphericalNOTestDataset(test_paths, stats=train_stats)

    test_sampler = SequentialSampler(test_dataset)

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        sampler=test_sampler,
        pin_memory=True,
        num_workers=0,
        # persistent_workers=True, # Avoids worker respawn overhead
    )

    climatologies = []

    for field in fields:
        climatology = np.load(
            os.path.join(
                climatologies_path,
                f"{field}_climatology.npy",
            )
        )[1:]

        if multiply_kmps:
            climatology *= 1e-5

        climatologies.append(
            torch.as_tensor(
                climatology,
                dtype=torch.float32,
                device=device,
            )
        )

    if len(climatologies) == 1:
        climatologies = climatologies[0].unsqueeze(0)
    else:
        climatologies = torch.stack(climatologies, dim=0)

    print("climatologies shape:", climatologies.shape)
    print("climatologies device:", climatologies.device)

    model = MultiModalSFNO(
        n_modalities=1,
        in_channels=1,
        out_channels=cfg["prediction_horizon"],
        hidden_channels=cfg["hidden_channels"],
        n_modes=(110, 128),
        n_layers=cfg["n_layers"],
        lifting_channel_ratio=cfg["lift_project_ratio"],
        projection_channel_ratio=cfg["lift_project_ratio"],
        grid_embedding_dim=4 if cfg["feed_grid_embeddings"] else None,
    ).to(device)

    checkpoint = torch.load(
        os.path.join(results_path, "model.pt"), map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint, strict=False)

    ang_emb = [t.to(device) for t in test_dataset.angular_embeddings]
    rad_emb = test_dataset.radial_embeddings.to(device)

    theta = test_dataset.theta.to(device)
    phi = test_dataset.phi.to(device)
    r = test_dataset.r.to(device)

    r_pred = r[1:]

    if multiply_kmps:
        r_pred *= 1e-5

    model.eval()

    rmse_list = []
    area_weighted_rmse_list = []
    gradient_weighted_rmse_list = []

    rmgse_list = []
    normalized_rmgse_list = []

    acc_list = []
    area_weighted_acc_list = []
    gradient_weighted_acc_list = []

    with torch.no_grad():
        for batch in tqdm(test_loader, leave=False):
            state = (
                torch.stack([batch[f"{f}_0"] for f in fields], dim=1)
                .unsqueeze(2)
                .to(device)
            )
            targets = torch.stack([batch[f"{f}_1_139"] for f in fields], dim=1).to(
                device
            )

            B = state.shape[0]
            Nr = test_dataset.Nr

            current_state = state
            current_radius = 0
            n_blocks = 0

            blocks = []

            while current_radius < Nr - 1:

                if cfg["feed_grid_embeddings"]:
                    grid_embeddings = _build_grid_embeddings(
                        current_radius, B, rad_emb, ang_emb, device
                    )
                else:
                    grid_embeddings = None
                pred = model(current_state, grid_embeddings=grid_embeddings)
                remaining = Nr - 1 - current_radius
                keep = min(cfg["prediction_horizon"], remaining)

                prediction = pred[:, :, :keep]
                n_blocks += 1

                current_state = pred[:, :, keep - 1 : keep]
                current_radius += keep
                blocks.append(prediction)
            predictions = torch.concat(blocks, dim=2)

            predictions_cgs = inverse_transform_model_batch(
                predictions, train_stats, drop_first_radius=True
            )
            targets_cgs = inverse_transform_model_batch(
                targets, train_stats, drop_first_radius=True
            )

            if multiply_kmps:
                predictions_cgs *= 1e-5
                targets_cgs *= 1e-5

            rmse_list.append(rmse(targets_cgs, predictions_cgs).detach().cpu().numpy())
            area_weighted_rmse_list.append(
                area_weighted_rmse(targets_cgs, predictions_cgs, theta)
                .detach()
                .cpu()
                .numpy()
            )
            gradient_weighted_rmse_list.append(
                gradient_weighted_rmse(
                    targets_cgs,
                    predictions_cgs,
                    r_pred,
                    theta,
                    phi,
                )
                .detach()
                .cpu()
                .numpy()
            )

            rmgse_list.append(
                rmgse(
                    targets_cgs,
                    predictions_cgs,
                    r_pred,
                    theta,
                    phi,
                )
                .detach()
                .cpu()
                .numpy()
            )
            normalized_rmgse_list.append(
                normalized_rmgse(
                    targets_cgs,
                    predictions_cgs,
                    r_pred,
                    theta,
                    phi,
                )
                .detach()
                .cpu()
                .numpy()
            )

            acc_list.append(
                acc(targets_cgs, predictions_cgs, climatologies).detach().cpu().numpy()
            )
            area_weighted_acc_list.append(
                area_weighted_acc(targets_cgs, predictions_cgs, climatologies, theta)
                .detach()
                .cpu()
                .numpy()
            )
            gradient_weighted_acc_list.append(
                gradient_weighted_acc(
                    targets_cgs,
                    predictions_cgs,
                    climatologies,
                    r_pred,
                    theta,
                    phi,
                )
                .detach()
                .cpu()
                .numpy()
            )
    rmse_list = np.concatenate(rmse_list, axis=0)
    area_weighted_rmse_list = np.concatenate(area_weighted_rmse_list, axis=0)
    gradient_weighted_rmse_list = np.concatenate(gradient_weighted_rmse_list, axis=0)

    rmgse_list = np.concatenate(rmgse_list, axis=0)
    normalized_rmgse_list = np.concatenate(normalized_rmgse_list, axis=0)

    acc_list = np.concatenate(acc_list, axis=0)
    area_weighted_acc_list = np.concatenate(area_weighted_acc_list, axis=0)
    gradient_weighted_acc_list = np.concatenate(gradient_weighted_acc_list, axis=0)

    output_name = (
        ("kmps_" if multiply_kmps else "cgs_")
        + "test_metrics_"
        + ("single_instrument" if single_instrument else "multi_instrument")
        + ".npz"
    )
    np.savez_compressed(
        os.path.join(results_path, output_name),
        rmse=rmse_list,
        area_weighted_rmse=area_weighted_rmse_list,
        gradient_weighted_rmse=gradient_weighted_rmse_list,
        rmgse=rmgse_list,
        normalized_rmgse=normalized_rmgse_list,
        acc=acc_list,
        area_weighted_acc=area_weighted_acc_list,
        gradient_weighted_acc=gradient_weighted_acc_list,
    )


if __name__ == "__main__":
    main()
