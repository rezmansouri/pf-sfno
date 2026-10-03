import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from tqdm import trange
from metrics import *
from hux_code.hux_propagation import apply_hux_f_model
from utils import get_cr_dirs, collect_sim_paths, HUXDataset

fields = ["vr"]


def get_hux_pred(f, r, p, t):
    r_plot = (695700) * r
    dr_vec = r_plot[1:] - r_plot[:-1]
    dp_vec = p[1:] - p[:-1]

    dr_vec = np.array(dr_vec, dtype=np.float32)
    dp_vec = np.array(dp_vec, dtype=np.float32)

    hux_f_res = np.ones((np.shape(f)[0], np.shape(f)[1], np.shape(f)[2]))
    for ii in range(len(t)):
        hux_f_res[:, ii, :] = apply_hux_f_model(f[:, ii, 0], dr_vec, dp_vec).T
    result = np.transpose(hux_f_res, [2, 1, 0])
    return result[1:, :, :]


def main():

    (
        data_path,
        split_csv_path,
        climatologies_path,
        single_instrument,
    ) = sys.argv[1:5]

    split_df = pd.read_csv(split_csv_path)

    single_instrument = single_instrument.lower() == "true"

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

    test_dataset = HUXDataset(test_paths)

    climatologies = []

    for field in fields:
        climatology = np.load(
            os.path.join(
                climatologies_path,
                f"{field}_climatology.npy",
            )
        )[1:] * 1e-5

        climatologies.append(torch.as_tensor(climatology, dtype=torch.float32))

    if len(climatologies) == 1:
        climatologies = climatologies[0].unsqueeze(0)
    else:
        climatologies = torch.stack(climatologies, dim=0)

    print("climatologies shape:", climatologies.shape)
    print("climatologies device:", climatologies.device)

    rmse_list = []
    area_weighted_rmse_list = []
    gradient_weighted_rmse_list = []

    rmgse_list = []
    normalized_rmgse_list = []

    acc_list = []
    area_weighted_acc_list = []
    gradient_weighted_acc_list = []

    for i in trange(len(test_dataset)):
        instance = test_dataset[i]
        v, r, phi, theta = (
            instance["x"]["v"],
            instance["x"]["r"],
            instance["x"]["p"],
            instance["x"]["t"],
        )

        r_pred = 695700 * r[1:]
        
        torch_r = torch.tensor(r_pred, dtype=torch.float64)
        torch_phi = torch.tensor(phi, dtype=torch.float64)
        torch_theta = torch.tensor(theta, dtype=torch.float64)

        targets_cgs = (
            torch.tensor(instance["y"].copy(), dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        pred = get_hux_pred(v, r, phi, theta)
        predictions_cgs = (
            torch.tensor(pred.copy(), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        )

        rmse_list.append(rmse(targets_cgs, predictions_cgs).detach().cpu().numpy())
        area_weighted_rmse_list.append(
            area_weighted_rmse(
                targets_cgs, predictions_cgs, torch_theta
            )
            .detach()
            .cpu()
            .numpy()
        )
        gradient_weighted_rmse_list.append(
            gradient_weighted_rmse(
                targets_cgs,
                predictions_cgs,
                torch_r,
                torch_theta,
                torch_phi,
            )
            .detach()
            .cpu()
            .numpy()
        )

        rmgse_list.append(
            rmgse(
                targets_cgs,
                predictions_cgs,
                torch_r,
                torch_theta,
                torch_phi,
            )
            .detach()
            .cpu()
            .numpy()
        )
        normalized_rmgse_list.append(
            normalized_rmgse(
                targets_cgs,
                predictions_cgs,
                torch_r,
                torch_theta,
                torch_phi,
            )
            .detach()
            .cpu()
            .numpy()
        )

        acc_list.append(
            acc(targets_cgs, predictions_cgs, climatologies).detach().cpu().numpy()
        )
        area_weighted_acc_list.append(
            area_weighted_acc(targets_cgs, predictions_cgs, climatologies, torch_theta)
            .detach()
            .cpu()
            .numpy()
        )
        gradient_weighted_acc_list.append(
            gradient_weighted_acc(
                targets_cgs,
                predictions_cgs,
                climatologies,
                torch_r,
                torch_theta,
                torch_phi,
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
        "hux_kmps_test_metrics_"
        + ("single_instrument" if single_instrument else "multi_instrument")
        + ".npz"
    )
    np.savez_compressed(
        output_name,
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
