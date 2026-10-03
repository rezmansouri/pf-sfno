import numpy as np


def average_adjacent(arr, axis):
    """Non-periodic half -> main: shrinks that axis by 1."""
    lo = [slice(None)] * arr.ndim
    hi = [slice(None)] * arr.ndim
    lo[axis] = slice(None, -1)
    hi[axis] = slice(1, None)
    return 0.5 * (arr[tuple(lo)] + arr[tuple(hi)])


def average_adjacent_periodic(arr, axis):
    """Periodic half -> main: same length, wraps last->first."""
    return 0.5 * (arr + np.roll(arr, -1, axis=axis))


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


def _transpose(field):
    """
    Transpose field from (phi, theta, r) to (r, theta, phi).
    """
    return np.transpose(field, (2, 1, 0))


def common_grid(data_dict, r_140, theta_110, phi):
    
    vr = _transpose(data_dict["vr"])  # (140,111,128)

    # --- Velocity ---
    vr = remesh_to_main(vr, code=(0, 1, 1))  # (140,111,128) -> (140,110,128)

    return {
        "vr": vr,
        "r": r_140,
        "theta": theta_110,
        "phi": phi,
    }
