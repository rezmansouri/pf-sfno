import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from psipy.model import MASOutput
import os

FILE_NAMES = ["vr002.hdf"]


def average_adjacent(arr, axis):
    """Non-periodic half -> main: shrinks that axis by 1."""
    lo = [slice(None)] * arr.ndim
    hi = [slice(None)] * arr.ndim
    lo[axis] = slice(None, -1)
    hi[axis] = slice(1, None)
    return 0.5 * (arr[tuple(lo)] + arr[tuple(hi)])


def remesh_to_main(arr, code):
    """
    code: 3-tuple/list of bool, (r_half, theta_half, phi_half)
    Moves every half-mesh axis onto the main mesh.
    r, theta -> non-periodic averaging (shrinks by 1)
    phi      -> periodic averaging (no shrink)
    """
    r_half, t_half, p_half = code
    out = arr
    if r_half:
        out = average_adjacent(out, axis=0)  # r: N -> N-1
    if t_half:
        out = average_adjacent(out, axis=1)  # theta: N -> N-1
    if p_half:
        pass
        # out = average_adjacent_periodic(out, axis=2)  # phi: N -> N (circular)
    return out


def get_sim(sim_path):
    model = MASOutput(sim_path)

    vr_model = model["vr"]

    p = vr_model.phi_coords
    t = vr_model.theta_coords
    r = vr_model.r_coords
    v = vr_model.data.values.squeeze()

    return v, r, p, t


def get_sims(sim_paths):
    vs = []
    ys = []
    
    _, r, p, t = get_sim(sim_paths[0])
    
    
    t = t[1:] + np.diff(t) / 2  # move to main mesh
    
    
    for sim_path in tqdm(sim_paths, desc="Loading simulations"):

        v, _, _, _ = get_sim(sim_path)

        v = np.transpose(v, (2, 1, 0))  # Transpose to (140, 111, 128)
        y = remesh_to_main(v, code=(0, 1, 1))  # (140,111,128) -> (140,110,128)
        v = np.transpose(y, (2, 1, 0))  # Transpose back to (128, 110, 140)
        y = y[1:, :, :]  # Remove the first radial layer (r=0.1 Rsun) to match the model output shape
        
        ys.append(y)
        vs.append(v)

    vs = np.stack(vs, axis=0)
    ys = np.stack(ys, axis=0)
    return vs, ys, r, p, t


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


class HUXDataset(Dataset):
    def __init__(
        self,
        sim_paths,
    ):
        super().__init__()
        vs, ys, r, p, t = get_sims(sim_paths)
        self.vs = vs
        self.ys = ys
        self.r = r
        self.p = p
        self.t = t

    def __getitem__(self, index):
        v = self.vs[index]
        y = self.ys[index]
        return {
            "x": {
                "v": v,
                "p": self.p,
                "t": self.t,
                "r": self.r,
            },
            "y": y,
        }

    def __len__(self):
        return len(self.vs)
