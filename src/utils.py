import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import os
import random
from pyhdf.SD import SD, SDC
import torch.distributed as dist

from common_grid_utils import common_grid


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # PyTorch 1.8+
    torch.use_deterministic_algorithms(True)

    # Required for some CUDA operations
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def setup_ddp():
    is_distributed = "LOCAL_RANK" in os.environ

    if not is_distributed:
        # Plain python run — no torchrun, no GPU required
        return 0, 0

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device(f"cuda:{local_rank}"))
    global_rank = dist.get_rank()
    return local_rank, global_rank


# ============================================================
# CONFIG
# ============================================================

FILE_NAMES = [
    "vr002.hdf",
]

dataset_names = ["Data-Set-2", "fakeDim0", "fakeDim1", "fakeDim2"]


# ============================================================
# FIELD GROUPS
# ============================================================

LOG_FIELDS = {"vr"}
ARCSINH_FIELDS = set()
ALL_FIELDS = LOG_FIELDS | ARCSINH_FIELDS

MODEL_FIELDS = ["vr"]

_LOG_EPS = np.finfo(np.float64).tiny


# ============================================================
# IO
# ============================================================


def read_hdf(hdf_path, dataset_names):
    f = SD(hdf_path, SDC.READ)
    return [f.select(name).get() for name in dataset_names]


def get_cr_dirs(data_path):
    """Return list of CR directories (crXXXX) inside data_path."""
    cr_dirs = sorted(
        [
            d
            for d in os.listdir(data_path)
            if d.startswith("cr") and os.path.isdir(os.path.join(data_path, d))
        ]
    )
    return cr_dirs


def list_dir(path):
    """
    returns only directories within a given path, ignoring hidden files and directories (those starting with a dot)
    """
    return sorted(
        [
            f
            for f in os.listdir(path)
            if os.path.isdir(os.path.join(path, f)) and not f.startswith(".")
        ]
    )


def collect_sim_paths(data_path, cr_list, single_instrument=False):
    if single_instrument:
        print("Warning: Collecting only one instrument per CR")
    sim_paths = []
    for cr in cr_list:
        cr_path = os.path.join(data_path, cr)
        for instrument in list_dir(cr_path):
            instrument_path = os.path.join(cr_path, instrument)
            if os.path.exists(instrument_path):
                sim_paths.append(instrument_path)
            if single_instrument:
                break
    return sim_paths


# ============================================================
# PHYSICS PIPELINE
# ============================================================


def cgs(data_dict):
    r = data_dict["r"] * 6.96e10
    vr = data_dict["vr"] * 481.3711 * 1e5

    return {
        "vr": vr,
        "r": r,
        "theta": data_dict["theta"],
        "phi": data_dict["phi"],
    }


def mks(data_dict):
    r = data_dict["r"] * 6.96e8
    vr = data_dict["vr"] * 481.3711 * 1e3

    return {
        "vr": vr,
        "r": r,
        "theta": data_dict["theta"],
        "phi": data_dict["phi"],
    }


def remove_radial_profile(data_dict):
    r = data_dict["r"]
    theta = data_dict["theta"]
    phi = data_dict["phi"]

    return {"r": r, "theta": theta, "phi": phi, "vr": data_dict["vr"]}


# ============================================================
# SIM PIPELINE (UNCHANGED)
# ============================================================


def read_sim(sim_path):

    vr, vr_phi, _, vr_r = read_hdf(f"{sim_path}/vr002.hdf", dataset_names)
    _, _, br_theta, _ = read_hdf(f"{sim_path}/br002.hdf", dataset_names)

    return (
        {"vr": vr},
        vr_r,
        br_theta,
        vr_phi,
    )


def get_sim(sim_path):

    out, r, theta, phi = read_sim(sim_path)

    data_dict = common_grid(out, r, theta, phi)
    data_dict = remove_radial_profile(data_dict)
    data_dict = cgs(data_dict)

    return data_dict


# ============================================================
# DATASET
# ============================================================


class SphericalNOTrainDataset(Dataset):
    """
    Final cached dataset. Pure in-memory, single-copy.

    Memory profile:
      - self.data[f]: ONE array of shape (N, Nr, Ntheta, Nphi) per field — the only
        thing that lives for the lifetime of the dataset.
      - During construction: at most ONE sim's raw arrays in RAM at a time.
      - Sims are read from disk twice (once for stats, once for final write) instead
        of being kept in a Python list — this trades I/O for RAM, which is the whole
        point of "no multiple copies in RAM".
    """

    def __init__(self, sim_paths, stats=None):
        self.sim_paths = sim_paths
        self.stats = stats

        self.r = None
        self.theta = None
        self.phi = None
        self.grid_embeddings = {}
        self.climatology = {}

        self.N = len(sim_paths)
        self._discover_shape()  # sets Nr, Ntheta, Nphi, grid tensors

        if self.stats is None:
            self._compute_stats()  # PASS 1: streaming, O(Nr) extra memory

        self._fill_data()  # PASS 2: writes directly into final arrays

    # =========================================================
    # SHAPE + GRID (load ONE sim, discard it)
    # =========================================================
    def _discover_shape(self):
        d = get_sim(self.sim_paths[0])
        self.Nr, self.Ntheta, self.Nphi = d["vr"].shape

        self.r = torch.from_numpy(d["r"]).float()
        self.theta = torch.from_numpy(d["theta"]).float()
        self.phi = torch.from_numpy(d["phi"]).float()

        r_norm = (d["r"] - d["r"].min()) / (d["r"].max() - d["r"].min())
        shape = d["vr"].shape
        rrr_norm = np.broadcast_to(r_norm[:, None, None], shape)
        sin_theta = np.broadcast_to(np.sin(d["theta"])[:, None], shape[1:])
        sin_phi = np.broadcast_to(np.sin(d["phi"])[None, :], shape[1:])
        cos_phi = np.broadcast_to(np.cos(d["phi"])[None, :], shape[1:])

        # .copy() because broadcast_to returns a read-only view into d["theta"]/d["phi"];
        # d is deleted below, so torch.tensor(...) must own its memory, not view it.
        self.radial_embeddings = torch.tensor(rrr_norm.copy(), dtype=torch.float32)
        self.angular_embeddings = [
            torch.tensor(sin_theta.copy(), dtype=torch.float32),
            torch.tensor(sin_phi.copy(), dtype=torch.float32),
            torch.tensor(cos_phi.copy(), dtype=torch.float32),
        ]
        del d

    # =========================================================
    # PASS 1: STATS — running accumulators only, never store sims
    # =========================================================
    def _compute_stats(self):
        self.stats = {}

        # =========================================================
        # a_scale (arcsinh fields): running sum of |x|
        # =========================================================
        for f in ARCSINH_FIELDS:
            total, count = 0.0, 0

            for sim_path in tqdm(self.sim_paths, desc=f"a_scale [{f}]"):
                x = get_sim(sim_path)[f]

                total += float(np.sum(np.abs(x)))
                count += x.size

                del x

            a_scale = (total / count) if count else 1.0

            self.stats[f] = {
                "transform": "arcsinh",
                "a_scale": a_scale or 1.0,
            }

        # =========================================================
        # Per-radius mean/std + climatology
        #
        # This is the SAME simulation-reading pass.
        # =========================================================
        for f in ALL_FIELDS:

            transform = "log" if f in LOG_FIELDS else "arcsinh"

            # Running statistics in transformed space
            s = np.zeros(self.Nr, dtype=np.float64)
            ss = np.zeros(self.Nr, dtype=np.float64)

            count_per_r = 0

            for sim_path in tqdm(self.sim_paths, desc=f"Stats [{f}]"):
                x = get_sim(sim_path)[f]

                # -------------------------------------------------
                # Existing normalization statistics
                # -------------------------------------------------
                x = x.astype(np.float64)

                if transform == "log":
                    x = np.log(np.maximum(x, _LOG_EPS))
                else:
                    x = np.arcsinh(x / self.stats[f]["a_scale"])

                s += x.sum(axis=(1, 2))
                ss += (x**2).sum(axis=(1, 2))

                count_per_r += x.shape[1] * x.shape[2]

                del x

            # -----------------------------------------------------
            # Existing mean/std
            # -----------------------------------------------------

            mean = s / count_per_r

            var = np.maximum(
                ss / count_per_r - mean**2,
                0.0,
            )

            std = np.sqrt(var)
            std = np.where(std < 1e-30, 1.0, std)

            self.stats.setdefault(f, {})

            self.stats[f]["transform"] = transform
            self.stats[f]["mean"] = mean.astype(np.float32)
            self.stats[f]["std"] = std.astype(np.float32)

    # =========================================================
    # PASS 2: fill preallocated final arrays, one sim at a time
    # =========================================================
    def _fill_data(self):
        # Preallocate the ONLY long-lived copy of the dataset up front.
        self.data = {
            f: np.empty((self.N, self.Nr, self.Ntheta, self.Nphi), dtype=np.float32)
            for f in ALL_FIELDS
        }

        for i, sim_path in enumerate(tqdm(self.sim_paths, desc="Filling data")):
            d = get_sim(sim_path)  # one sim in RAM
            for f in ALL_FIELDS:
                x = d[f]
                if self.stats[f]["transform"] == "log":
                    x = np.log(np.maximum(x, _LOG_EPS))
                else:
                    x = np.arcsinh(x / self.stats[f]["a_scale"])
                mean = self.stats[f]["mean"][:, None, None]
                std = self.stats[f]["std"][:, None, None]
                x = (x - mean) / std
                self.data[f][i] = x.astype(
                    np.float32
                )  # write into final slot, no copy kept
            del d  # sim's raw arrays freed here

    # =========================================================
    # DATASET API
    # =========================================================
    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        sample = {}
        for f in ALL_FIELDS:
            field = self.data[f][idx]
            sample[f"{f}_0"] = torch.from_numpy(field[0]).float()
            sample[f"{f}_1_139"] = torch.from_numpy(field[1:]).float()
        return sample

    def get_stats(self):
        return self.stats

    def get_climatologies(self):
        """
        Returns the full 3D physical climatology for all fields.

        The climatology is returned in the original CGS physical space,
        including restoration of the radial profiles removed by
        remove_radial_profile().

        Returns
        -------
        dict
            {field: np.ndarray of shape (Nr, Ntheta, Nphi), dtype float32}

            Values are in the original CGS physical units.
        """

        climatologies = {}

        # ---------------------------------------------------------
        # Dimensionless radial coordinate used by
        # remove_radial_profile()
        #
        # self.r is stored after cgs(), so it is in cm.
        # ---------------------------------------------------------
        r = self.r.numpy() / 6.96e10

        rr = r[:, None, None]

        for f in ALL_FIELDS:

            # -----------------------------------------------------
            # 1. Undo normalization
            # -----------------------------------------------------

            mean = self.stats[f]["mean"][:, None, None]
            std = self.stats[f]["std"][:, None, None]

            x = self.data[f].astype(np.float64)
            x = x * std + mean

            # -----------------------------------------------------
            # 2. Undo nonlinear transform
            # -----------------------------------------------------

            if self.stats[f]["transform"] == "log":

                x = np.exp(x)

            elif self.stats[f]["transform"] == "arcsinh":

                a = self.stats[f]["a_scale"]
                x = np.sinh(x) * a

            else:
                raise ValueError(
                    f"Unknown transform '{self.stats[f]['transform']}' "
                    f"for field '{f}'"
                )

            # -----------------------------------------------------
            # 3. Restore the radial profile removed by
            #    remove_radial_profile()
            # -----------------------------------------------------

            if f in {"br", "bt", "bp", "rho"}:
                x = x / rr**2

            elif f == "t":
                x = x / rr

            # vr, vt, vp have no radial profile removed

            # -----------------------------------------------------
            # 4. Average over simulations in physical CGS space
            # -----------------------------------------------------

            climatologies[f] = np.mean(
                x,
                axis=0,
                dtype=np.float64,
            ).astype(np.float32)

            del x

        return climatologies


class SphericalNOTestDataset(SphericalNOTrainDataset):

    def __init__(self, data, stats):
        super().__init__(data, stats)

    def inverse_transform(self, x, field_name, is_pred=False):
        """
        Inverts:
            1. normalization
            2. log / arcsinh transform

        Output stays in CGS (since CGS is already what was transformed).
        """

        stats = self.stats[field_name]

        # -----------------------------------------------------
        # 1. denormalize
        # -----------------------------------------------------

        if is_pred:
            mean = stats["mean"][1:][None, :, None, None]
            std = stats["std"][1:][None, :, None, None]

        else:
            mean = stats["mean"][None, :, None, None]
            std = stats["std"][None, :, None, None]

        x = x * std + mean

        # -----------------------------------------------------
        # 2. inverse transform
        # -----------------------------------------------------

        if stats["transform"] == "log":

            x = np.exp(x)
            x = np.maximum(x, _LOG_EPS)

        elif stats["transform"] == "arcsinh":

            a = stats["a_scale"]
            x = np.sinh(x) * a

        else:
            raise ValueError(f"Unknown transform: {stats['transform']}")

        return x

    def get_physical_sample(self, idx):
        """
        Returns full CGS physical state (no normalization, no transforms).
        """

        sample = {}

        for f in ALL_FIELDS:

            x_norm = self.data[f][idx]  # (Nr, theta, phi)

            x_phys = self.inverse_transform(x_norm, f)

            sample[f] = x_phys

        return sample


def inverse_transform_model_batch(
    batch,
    stats,
    drop_first_radius=False,
):
    """
    Convert a model batch from normalized/transformed space back to
    physical CGS values.

    Parameters
    ----------
    batch : torch.Tensor
        Shape:
            (B, M, R, Theta, Phi)

        Modalities must follow MODEL_FIELDS.

    stats : dict
        Dataset statistics.

    drop_first_radius : bool
        If True, use statistics for r[1:] rather than r[:].

        This should be True for model predictions and targets with
        shape (B, M, Nr-1, Theta, Phi).

    Returns
    -------
    torch.Tensor
        Physical CGS tensor on the SAME device as `batch`.
    """

    if not torch.is_tensor(batch):
        raise TypeError(f"`batch` must be a torch.Tensor, got {type(batch)}")

    if batch.ndim != 5:
        raise ValueError(
            f"Expected batch shape (B, M, R, Theta, Phi), " f"got {tuple(batch.shape)}"
        )

    B, M, R, Ntheta, Nphi = batch.shape

    if M != len(MODEL_FIELDS):
        raise ValueError(
            f"Expected {len(MODEL_FIELDS)} modalities, got {M}. "
            f"Expected order: {MODEL_FIELDS}"
        )

    physical = torch.empty_like(batch)

    for modality_idx, field_name in enumerate(MODEL_FIELDS):

        field_stats = stats[field_name]

        mean = torch.as_tensor(
            field_stats["mean"],
            dtype=batch.dtype,
            device=batch.device,
        )

        std = torch.as_tensor(
            field_stats["std"],
            dtype=batch.dtype,
            device=batch.device,
        )

        if drop_first_radius:
            mean = mean[1:]
            std = std[1:]

        if mean.shape[0] != R:
            raise ValueError(
                f"Radial mismatch for {field_name}: "
                f"batch has R={R}, stats provide R={mean.shape[0]}"
            )

        mean = mean[None, :, None, None]
        std = std[None, :, None, None]

        x = batch[:, modality_idx]

        # Undo normalization
        x = x * std + mean

        # Undo nonlinear transform
        if field_stats["transform"] == "log":
            x = torch.exp(x)

        elif field_stats["transform"] == "arcsinh":
            x = torch.sinh(x) * field_stats["a_scale"]

        else:
            raise ValueError(
                f"Unknown transform '{field_stats['transform']}' "
                f"for field '{field_name}'"
            )

        physical[:, modality_idx] = x

    return physical
