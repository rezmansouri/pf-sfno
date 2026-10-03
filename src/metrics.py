import torch
import numpy as np

## UTILS


def ensure_float64(*tensors):
    return tuple(
        x if x.dtype == torch.float64 else x.to(torch.float64) for x in tensors
    )


def spherical_gradient(
    field: torch.Tensor,
    r: torch.Tensor,
    theta: torch.Tensor,
    phi: torch.Tensor,
    theta_unsafe=1.0,  # degrees
):
    """
    Physical gradient magnitude components for a scalar spherical field.

    field: (B, M, R, Theta, Phi)

    theta and phi must be in radians.
    phi must be a uniform periodic grid without a duplicated endpoint.
    """

    assert field.ndim == 5
    # field, r, theta, phi = ensure_float64(field, r, theta, phi)

    # ------------------------------------------------------------
    # Radial
    # ------------------------------------------------------------

    d_dr = torch.gradient(
        field,
        spacing=(r,),
        dim=2,
        edge_order=1,
    )[0]

    # ------------------------------------------------------------
    # Theta
    # ------------------------------------------------------------

    d_dtheta = torch.gradient(
        field,
        spacing=(theta,),
        dim=3,
        edge_order=1,
    )[0]

    # ------------------------------------------------------------
    # Phi: periodic central difference
    # ------------------------------------------------------------

    dphi = phi[1] - phi[0]

    d_dphi = (torch.roll(field, -1, dims=4) - torch.roll(field, 1, dims=4)) / (
        2.0 * dphi
    )

    # ------------------------------------------------------------
    # Physical spherical factors
    # ------------------------------------------------------------

    r_view = r.view(1, 1, -1, 1, 1)

    sin_theta = torch.sin(theta)

    sin_theta_view = sin_theta.view(1, 1, 1, -1, 1)

    grad_r = d_dr

    grad_theta = d_dtheta / r_view

    grad_phi = d_dphi / (r_view * sin_theta_view)

    # Do not trust phi derivative at the poles
    theta_min = torch.deg2rad(
        torch.tensor(theta_unsafe, device=theta.device, dtype=theta.dtype)
    )
    theta_max = torch.pi - theta_min

    mask = (theta >= theta_min) & (theta <= theta_max)

    grad_phi = torch.where(
        mask.view(1, 1, 1, -1, 1),
        grad_phi,
        torch.zeros_like(grad_phi),
    )

    return grad_r, grad_theta, grad_phi


def spherical_area_weights(
    theta: torch.Tensor,
    B: int,
    M: int,
    R: int,
    P: int,
) -> torch.Tensor:
    """
    Compute normalized spherical surface-area weights.

    Args:
        theta: (Theta,) in radians, ranging from 0 to pi.
        B: batch size.
        M: number of modalities.
        R: number of radial slices.
        P: number of phi points.

    Returns:
        weights: (B, M, R, Theta, Phi)

        For every (b, m, r):
            weights[b, m, r].sum() == 1
    """
    # (theta,) = ensure_float64(theta)

    # Area weighting for each latitude/colatitude row.
    weights = torch.sin(theta).clamp_min(0.0)

    # Expand across phi.
    weights = weights[:, None].expand(-1, P)

    # Normalize over the entire spherical surface.
    weights = weights / weights.sum()

    # Shape: (1, 1, 1, Theta, Phi)
    weights = weights.view(1, 1, 1, -1, P)

    # Expand to (B, M, R, Theta, Phi)
    weights = weights.expand(B, M, R, -1, -1)

    return weights


def calculate_grad_magnitude_radially_L1_scaled(
    grad_r,
    grad_theta,
    grad_phi,
    radially_scaled=True,
):
    """
    Calculate gradient magnitude.

    Args:
        grad_r:     (B, M, R, Theta, Phi)
        grad_theta: (B, M, R, Theta, Phi)
        grad_phi:   (B, M, R, Theta, Phi)

        radially_scaled:
            If True, normalize each radial shell independently so that
            the sum over (Theta, Phi) equals 1.

    Returns:
        magnitude: (B, M, R, Theta, Phi)
    """
    # grad_r, grad_theta, grad_phi = ensure_float64(grad_r, grad_theta, grad_phi)

    magnitude = torch.sqrt(grad_r**2 + grad_theta**2 + grad_phi**2)

    if radially_scaled:
        # Sum independently for every (B, M, R)
        # over the angular dimensions.
        radial_sum = magnitude.sum(
            dim=(-2, -1),
            keepdim=True,
        )

        # Prevent division by zero for a completely zero shell.
        radial_sum = radial_sum.clamp_min(1e-12)

        magnitude = magnitude / radial_sum

    return magnitude


## METRICS


def rmse(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """
    Area-weighted RMSE on spherical shells.

    Args:
        y_true: (B, M, R, Theta, Phi), float64
        y_pred: (B, M, R, Theta, Phi), float64
        theta:  (Theta,), colatitude in radians

    Returns:
        rmse: (B, M, R)
    """
    assert y_true.shape == y_pred.shape
    assert y_true.ndim == 5

    y_true, y_pred = ensure_float64(y_true, y_pred)

    squared_error = (y_true - y_pred) ** 2

    mse = (squared_error).mean(dim=(-2, -1))

    return torch.sqrt(mse)


def area_weighted_rmse(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    theta: torch.Tensor,
) -> torch.Tensor:
    """
    Area-weighted RMSE on spherical shells.

    Returns:
        (B, M, R)
    """
    assert y_true.shape == y_pred.shape
    assert y_true.ndim == 5
    assert theta.ndim == 1
    assert y_true.shape[-2] == theta.numel()

    y_true, y_pred, theta = ensure_float64(y_true, y_pred, theta)

    B, M, R, T, P = y_true.shape

    weights = spherical_area_weights(
        theta,
        B,
        M,
        R,
        P,
    ).to(
        device=y_true.device,
        dtype=y_true.dtype,
    )

    squared_error = (y_true - y_pred) ** 2

    mse = (squared_error * weights).sum(dim=(-2, -1))

    return torch.sqrt(mse)


def gradient_weighted_rmse(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    r: torch.Tensor,
    theta: torch.Tensor,
    phi: torch.Tensor,
) -> torch.Tensor:
    """
    Gradient-weighted RMSE.

    The error is weighted according to the ground-truth gradient
    magnitude. Gradient weights are normalized independently for
    every (B, M, R) radial shell so that:

        sum(theta, phi) weights = 1

    Returns:
        (B, M, R)
    """
    assert y_true.shape == y_pred.shape
    assert y_true.ndim == 5

    y_true, y_pred, r, theta, phi = ensure_float64(y_true, y_pred, r, theta, phi)

    # Ground-truth gradient.
    grad_r, grad_theta, grad_phi = spherical_gradient(
        y_true,
        r,
        theta,
        phi,
    )

    # Normalize gradient magnitude independently per radius.
    grad_weights = calculate_grad_magnitude_radially_L1_scaled(
        grad_r,
        grad_theta,
        grad_phi,
        radially_scaled=True,
    )

    squared_error = (y_true - y_pred) ** 2

    weighted_mse = (squared_error * grad_weights).sum(dim=(-2, -1))

    return torch.sqrt(weighted_mse)


def rmgse(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    r: torch.Tensor,
    theta: torch.Tensor,
    phi: torch.Tensor,
) -> torch.Tensor:
    """
    Root Mean Squared Gradient Error (RMSGE).

    Computes the physical spherical gradient of the true and
    predicted fields and measures their component-wise difference.

    Returns:
        (B, M, R)
    """

    assert y_true.shape == y_pred.shape
    assert y_true.ndim == 5

    y_true, y_pred, r, theta, phi = ensure_float64(y_true, y_pred, r, theta, phi)

    true_grad_r, true_grad_theta, true_grad_phi = spherical_gradient(
        y_true,
        r,
        theta,
        phi,
    )

    pred_grad_r, pred_grad_theta, pred_grad_phi = spherical_gradient(
        y_pred,
        r,
        theta,
        phi,
    )

    gradient_squared_error = (
        (true_grad_r - pred_grad_r) ** 2
        + (true_grad_theta - pred_grad_theta) ** 2
        + (true_grad_phi - pred_grad_phi) ** 2
    )

    mse_gradient = gradient_squared_error.mean(dim=(-2, -1))

    return torch.sqrt(mse_gradient)


def normalized_rmgse(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    r: torch.Tensor,
    theta: torch.Tensor,
    phi: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Area-weighted normalized RMS Gradient Error.

    Measures gradient error relative to the magnitude of the
    true gradient on each spherical shell.

    Returns:
        (B, M, R)
    """

    assert y_true.shape == y_pred.shape
    assert y_true.ndim == 5

    y_true, y_pred, r, theta, phi = ensure_float64(y_true, y_pred, r, theta, phi)

    B, M, R, T, P = y_true.shape

    true_grad_r, true_grad_theta, true_grad_phi = spherical_gradient(
        y_true, r, theta, phi
    )

    pred_grad_r, pred_grad_theta, pred_grad_phi = spherical_gradient(
        y_pred, r, theta, phi
    )

    gradient_squared_error = (
        (true_grad_r - pred_grad_r) ** 2
        + (true_grad_theta - pred_grad_theta) ** 2
        + (true_grad_phi - pred_grad_phi) ** 2
    )

    true_gradient_squared = true_grad_r**2 + true_grad_theta**2 + true_grad_phi**2

    error_energy = (gradient_squared_error).mean(dim=(-2, -1))

    true_energy = (true_gradient_squared).mean(dim=(-2, -1))

    return torch.sqrt(error_energy / true_energy.clamp_min(eps))


def acc(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    climatology: torch.Tensor,
) -> torch.Tensor:
    """
    Climatology-based Anomaly Correlation Coefficient.

    Args:
        y_true:
            (B, M, R, Theta, Phi)

        y_pred:
            (B, M, R, Theta, Phi)

        climatology:
            (M, R, Theta, Phi)

    Returns:
        acc: (B, M, R)
    """

    assert y_true.shape == y_pred.shape
    assert y_true.ndim == 5
    assert climatology.shape == y_true.shape[1:]

    y_true, y_pred, climatology = ensure_float64(y_true, y_pred, climatology)

    clim = climatology.unsqueeze(0)

    true_anomaly = y_true - clim
    pred_anomaly = y_pred - clim

    numerator = (true_anomaly * pred_anomaly).sum(dim=(-2, -1))

    true_norm = torch.sqrt((true_anomaly**2).sum(dim=(-2, -1)))

    pred_norm = torch.sqrt((pred_anomaly**2).sum(dim=(-2, -1)))

    denominator = true_norm * pred_norm

    return numerator / denominator.clamp_min(1e-12)


def area_weighted_acc(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    climatology: torch.Tensor,
    theta: torch.Tensor,
) -> torch.Tensor:
    """
    Area-weighted climatology-based ACC.

    Returns:
        (B, M, R)
    """

    assert y_true.shape == y_pred.shape
    assert y_true.ndim == 5
    assert climatology.shape == y_true.shape[1:]
    assert theta.ndim == 1
    assert y_true.shape[-2] == theta.numel()

    y_true, y_pred, climatology, theta = ensure_float64(
        y_true, y_pred, climatology, theta
    )

    B, M, R, T, P = y_true.shape

    weights = spherical_area_weights(
        theta,
        B,
        M,
        R,
        P,
    ).to(
        device=y_true.device,
        dtype=y_true.dtype,
    )

    clim = climatology.unsqueeze(0)

    true_anomaly = y_true - clim
    pred_anomaly = y_pred - clim

    numerator = (weights * true_anomaly * pred_anomaly).sum(dim=(-2, -1))

    true_norm = torch.sqrt((weights * true_anomaly**2).sum(dim=(-2, -1)))

    pred_norm = torch.sqrt((weights * pred_anomaly**2).sum(dim=(-2, -1)))

    denominator = true_norm * pred_norm

    return numerator / denominator.clamp_min(1e-12)


def gradient_weighted_acc(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    climatology: torch.Tensor,
    r: torch.Tensor,
    theta: torch.Tensor,
    phi: torch.Tensor,
) -> torch.Tensor:
    """
    Gradient-weighted climatology-based ACC.

    The spatial contribution to ACC is weighted according to the
    ground-truth gradient magnitude. The gradient weights are
    normalized independently for each radial shell.

    Returns:
        (B, M, R)
    """

    assert y_true.shape == y_pred.shape
    assert y_true.ndim == 5
    assert climatology.shape == y_true.shape[1:]

    y_true, y_pred, climatology, r, theta, phi = ensure_float64(
        y_true, y_pred, climatology, r, theta, phi
    )

    # ---------------------------------------------------------
    # Ground-truth gradient magnitude
    # ---------------------------------------------------------

    grad_r, grad_theta, grad_phi = spherical_gradient(
        y_true,
        r,
        theta,
        phi,
    )

    grad_weights = calculate_grad_magnitude_radially_L1_scaled(
        grad_r,
        grad_theta,
        grad_phi,
        radially_scaled=True,
    )

    # ---------------------------------------------------------
    # Anomalies relative to climatology
    # ---------------------------------------------------------

    clim = climatology.unsqueeze(0)

    true_anomaly = y_true - clim
    pred_anomaly = y_pred - clim

    # ---------------------------------------------------------
    # Weighted ACC
    # ---------------------------------------------------------

    numerator = (grad_weights * true_anomaly * pred_anomaly).sum(dim=(-2, -1))

    true_norm = torch.sqrt((grad_weights * true_anomaly**2).sum(dim=(-2, -1)))

    pred_norm = torch.sqrt((grad_weights * pred_anomaly**2).sum(dim=(-2, -1)))

    denominator = true_norm * pred_norm

    return numerator / denominator.clamp_min(1e-12)
