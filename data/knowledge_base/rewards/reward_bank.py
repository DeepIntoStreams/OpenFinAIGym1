import itertools
import logging
import math
import typing
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import numpy as np
import torch
from scipy import linalg
from sklearn.metrics.pairwise import polynomial_kernel
from torch import nn

try:
    import signatory
except ImportError:
    signatory = None

try:
    import ksig
except ImportError:
    ksig = None

logger = logging.getLogger(__name__)


class Loss(nn.Module):
    """Marker base for evaluation rewards.

    Exists so the AST-based reward-class selector in the assembler can
    distinguish rewards from helper classes. Subclasses implement their own
    ``forward`` returning a scalar tensor — there is no inherited reduction
    or normalisation. The ``**_unused_legacy`` catch-all absorbs any stray
    ``reg=``/``transform=``/``threshold=``/``backward=``/``norm_foo=``
    kwargs from legacy LLM-generated rewards so they don't crash; those
    values are intentionally not stored.
    """

    def __init__(self, name: str = "", **_unused_legacy: Any) -> None:
        super().__init__()
        self.name = name


class BaseAugmentation:
    pass

    def apply(self, *args: List[torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError("Needs to be implemented by child.")


@dataclass
class Scale(BaseAugmentation):
    scale: float = 1
    dim: int = None

    def apply(self, x: torch.Tensor):
        if self.dim is None:
            return self.scale * x
        else:
            x[..., self.dim] = self.scale * x[..., self.dim]
            return x


def apply_augmentations(x: torch.Tensor, augmentations: Tuple) -> torch.Tensor:
    y = x.clone()
    for augmentation in augmentations:
        y = augmentation.apply(y)
    return y


def to_numpy(x):
    """
    Casts torch.Tensor to a numpy ndarray.

    The function detaches the tensor from its gradients, then puts it onto the
    cpu and at last casts it to numpy.
    """
    return x.detach().cpu().numpy()


def cov_torch(x, rowvar=False, bias=True, ddof=None, aweights=None):
    device = x.device
    x = to_numpy(x)
    _, L, C = x.shape
    x = x.reshape(-1, L * C)
    return torch.from_numpy(np.cov(x, rowvar=False)).to(device).float()


def q_var_torch(x: torch.Tensor):
    """Compute the quadratic variation of ``x``.

    Args:
        x: torch.Tensor of shape [B, S, D].

    Returns:
        Quadratic variation of ``x``, shape [B, D].
    """
    return torch.sum(torch.pow(x[:, 1:] - x[:, :-1], 2), 1)


def acf_torch(x: torch.Tensor, max_lag: int, dim: Tuple[int] = (0, 1)) -> torch.Tensor:
    """Compute the autocorrelation function (ACF) of ``x``.

    Args:
        x: torch.Tensor of shape [B, S, D].
        max_lag: Number of lags to compute the ACF for.

    Returns:
        ACF of ``x``, shape [max_lag, D].
    """
    acf_list = list()
    x = x - x.mean((0, 1))
    std = torch.var(x, unbiased=False, dim=(0, 1))
    for i in range(max_lag):
        y = x[:, i:] * x[:, :-i] if i > 0 else torch.pow(x, 2)
        acf_i = torch.mean(y, dim) / std
        acf_list.append(acf_i)
    if dim == (0, 1):
        return torch.stack(acf_list)
    else:
        return torch.cat(acf_list, 1)


def non_stationary_acf_torch(X, symmetric=False):
    """Compute the correlation matrix between any two time points of the time series.

    Args:
        X: torch.Tensor of shape [B, T, D].
        symmetric: When True, return the upper-triangular matrix instead of
            the full matrix.

    Returns:
        Correlation matrix of shape [T, T, D] where each entry
        ``(t_i, t_j, d_i)`` is the correlation between the d_i-th coordinate
        of ``X_{t_i}`` and ``X_{t_j}``.
    """
    # Get the batch size, sequence length, and input dimension from the input tensor
    B, T, D = X.shape

    # Create a tensor to hold the correlations
    correlations = torch.zeros(T, T, D)

    # Loop through each time step from lag to T-1
    for t in range(T):
        # Loop through each lag from 1 to lag
        for tau in range(t, T):
            # Compute the correlation between X_{t, d} and X_{t-tau, d}
            correlation = torch.sum(X[:, t, :] * X[:, tau, :], dim=0) / (
                torch.norm(X[:, t, :], dim=0) * torch.norm(X[:, tau, :], dim=0)
            )
            # print(correlation)
            # Store the correlation in the output tensor
            correlations[t, tau, :] = correlation
            if symmetric:
                correlations[tau, t, :] = correlation

    return correlations


def cacf_torch(x, lags: list, dim=(0, 1)):
    """Compute the cross-correlation between feature dimension and time dimension.

    Args:
        x: Input tensor; standardised along ``dim`` before correlation.
        lags: Number of lags to compute.
        dim: Dimensions to standardise over before correlation.

    Returns:
        Cross-correlation tensor reshaped to
        ``(batch, lags, len(lower_triangular_indices))``.
    """

    # Define a helper function to get the lower triangular indices for a given dimension
    def get_lower_triangular_indices(n):
        return [list(x) for x in torch.tril_indices(n, n)]

    # Get the lower triangular indices for the input tensor x
    ind = get_lower_triangular_indices(x.shape[2])

    # Standardize the input tensor x along the given dimensions
    x = (x - x.mean(dim, keepdims=True)) / x.std(dim, keepdims=True)

    # Split the input tensor into left and right parts based on the lower
    # triangular indices
    x_l = x[..., ind[0]]
    x_r = x[..., ind[1]]

    # Compute the cross-correlation at each lag and store in a list
    cacf_list = list()
    for i in range(lags):
        # Compute the element-wise product of the left and right parts, shifted by the lag if i > 0
        y = x_l[:, i:] * x_r[:, :-i] if i > 0 else x_l * x_r

        # Compute the mean of the product along the time dimension
        cacf_i = torch.mean(y, (1))

        # Append the result to the list of cross-correlations
        cacf_list.append(cacf_i)

    # Concatenate the cross-correlations across lags and reshape to the desired output shape
    cacf = torch.cat(cacf_list, 1)
    return cacf.reshape(cacf.shape[0], -1, len(ind[0]))


def skew_torch(x, dim=(0, 1), dropdims=True):
    x = x - x.mean(dim, keepdims=True)
    x_3 = torch.pow(x, 3).mean(dim, keepdims=True)
    x_std_3 = torch.pow(x.std(dim, unbiased=True, keepdims=True), 3)
    skew = x_3 / x_std_3
    if dropdims:
        skew = skew[0, 0]
    return skew


def kurtosis_torch(x, dim=(0, 1), excess=True, dropdims=True):
    x = x - x.mean(dim, keepdims=True)
    x_4 = torch.pow(x, 4).mean(dim, keepdims=True)
    x_var2 = torch.pow(torch.var(x, dim=dim, unbiased=False, keepdims=True), 2)
    kurtosis = x_4 / x_var2
    if excess:
        kurtosis = kurtosis - 3
    if dropdims:
        kurtosis = kurtosis[0, 0]
    return kurtosis


def acf_diff(x):
    return torch.sqrt(torch.pow(x, 2).sum(0))


def cc_diff(x):
    return torch.abs(x).sum(0)


def cov_diff(x):
    return torch.abs(x).mean()


class ACFLoss(Loss):
    def __init__(self, gt, max_lag=64, stationary=True, **kwargs):
        super().__init__(**kwargs)
        self.max_lag = min(max_lag, gt.shape[1])
        self.stationary = stationary
        if stationary:
            self.acf_real = acf_torch(gt, self.max_lag, dim=(0, 1))
        else:
            self.acf_real = non_stationary_acf_torch(gt, symmetric=False)

    def forward(self, pred):
        if self.stationary:
            acf_fake = acf_torch(pred, self.max_lag)
        else:
            acf_fake = non_stationary_acf_torch(pred, symmetric=False).to(
                pred.device
            )
        return acf_diff(acf_fake - self.acf_real.to(pred.device)).mean()


class MeanLoss(Loss):
    def __init__(self, gt, **kwargs):
        super().__init__(**kwargs)
        self.mean = gt.mean((0, 1))

    def forward(self, pred):
        return torch.abs(pred.mean((0, 1)) - self.mean).mean()


class StdLoss(Loss):
    def __init__(self, gt, **kwargs):
        super().__init__(**kwargs)
        self.std_real = gt.std((0, 1))

    def forward(self, pred):
        return torch.abs(pred.std((0, 1)) - self.std_real).mean()


class SkewnessLoss(Loss):
    def __init__(self, gt, **kwargs):
        super().__init__(**kwargs)
        self.skew_real = skew_torch(gt)

    def forward(self, pred):
        return torch.abs(skew_torch(pred) - self.skew_real).mean()


class KurtosisLoss(Loss):
    def __init__(self, gt, **kwargs):
        super().__init__(**kwargs)
        self.kurtosis_real = kurtosis_torch(gt)

    def forward(self, pred):
        return torch.abs(kurtosis_torch(pred) - self.kurtosis_real).mean()


class CrossCorrelLoss(Loss):
    def __init__(self, gt, max_lag=64, **kwargs):
        super().__init__(**kwargs)
        self.cross_correl_real = cacf_torch(gt, max_lag).mean(0)[0]
        self.max_lag = max_lag

    def forward(self, pred):
        cross_correl_fake = cacf_torch(pred, self.max_lag).mean(0)[0]
        # ``.unsqueeze(0)`` preserves the rank-1 shape that callers (and the
        # prior ``Loss.forward → .mean()`` reduction) previously saw.
        return (
            cc_diff(cross_correl_fake - self.cross_correl_real.to(pred.device))
            .unsqueeze(0)
            .mean()
        )


class CovLoss(Loss):
    def __init__(self, gt, **kwargs):
        super().__init__(**kwargs)
        self.covariance_real = cov_torch(gt)

    def forward(self, pred):
        covariance_fake = cov_torch(pred)
        return cov_diff(
            covariance_fake - self.covariance_real.to(pred.device)
        ).mean()


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Compute the Frechet Distance between two multivariate Gaussians.

    The Frechet distance between ``X_1 ~ N(mu_1, C_1)`` and
    ``X_2 ~ N(mu_2, C_2)`` is
    ``d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2))``.
    Stable version by Dougal J. Sutherland.

    Args:
        mu1: Activations of the pool_3 layer for generated samples (as
            returned by ``get_predictions``).
        sigma1: Covariance matrix over those activations for generated
            samples.
        mu2: Sample mean over the pool_3 activations, precalculated on a
            reference data set.
        sigma2: Covariance matrix over the pool_3 activations,
            precalculated on a reference data set.
        eps: Stability epsilon added to the diagonal before sqrt.

    Returns:
        The Frechet Distance (a non-negative scalar).
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert (
        mu1.shape == mu2.shape
    ), "Training and test mean vectors have different lengths"
    assert (
        sigma1.shape == sigma2.shape
    ), "Training and test covariances have different dimensions"

    diff = mu1 - mu2
    # product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = (
            "fid calculation produces singular product; adding %s to diagonal of cov estimates"
            % eps
        )
        warnings.warn(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return torch.tensor(
        diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
    )


def polynomial_mmd_averages(
    codes_g, codes_r, n_subsets=50, subset_size=1000, ret_var=False, **kernel_args
):
    m = min(codes_g.shape[0], codes_r.shape[0])
    mmds = np.zeros(n_subsets)
    if ret_var:
        vars = np.zeros(n_subsets)
    choice = np.random.choice

    for i in range(n_subsets):
        g = codes_g[choice(len(codes_g), subset_size, replace=False)]
        r = codes_r[choice(len(codes_r), subset_size, replace=False)]
        o = polynomial_mmd(g, r, **kernel_args, var_at_m=m, ret_var=ret_var)
        if ret_var:
            mmds[i], vars[i] = o
        else:
            mmds[i] = o
    return (mmds, vars) if ret_var else torch.tensor(mmds.mean() * 1e3)


def polynomial_mmd(
    codes_g, codes_r, degree=3, gamma=None, coef0=1, var_at_m=None, ret_var=True
):
    # use  k(x, y) = (gamma <x, y> + coef0)^degree
    # default gamma is 1 / dim
    X = codes_g
    Y = codes_r

    K_XX = polynomial_kernel(X, degree=degree, gamma=gamma, coef0=coef0)
    K_YY = polynomial_kernel(Y, degree=degree, gamma=gamma, coef0=coef0)
    K_XY = polynomial_kernel(X, Y, degree=degree, gamma=gamma, coef0=coef0)

    return _mmd2_and_variance(K_XX, K_XY, K_YY, var_at_m=var_at_m, ret_var=ret_var)


def _sqn(arr):
    flat = np.ravel(arr)
    return flat.dot(flat)


def _mmd2_and_variance(
    K_XX,
    K_XY,
    K_YY,
    unit_diagonal=False,
    mmd_est="unbiased",
    block_size=1024,
    var_at_m=None,
    ret_var=True,
):
    # based on
    # https://github.com/dougalsutherland/opt-mmd/blob/master/two_sample/mmd.py
    # but changed to not compute the full kernel matrix at once
    m = K_XX.shape[0]
    assert K_XX.shape == (m, m)
    assert K_XY.shape == (m, m)
    assert K_YY.shape == (m, m)
    if var_at_m is None:
        var_at_m = m

    # Get the various sums of kernels that we'll use
    # Kts drop the diagonal, but we don't need to compute them explicitly
    if unit_diagonal:
        diag_X = diag_Y = 1
        sum_diag_X = sum_diag_Y = m
        sum_diag2_X = sum_diag2_Y = m
    else:
        diag_X = np.diagonal(K_XX)
        diag_Y = np.diagonal(K_YY)

        sum_diag_X = diag_X.sum()
        sum_diag_Y = diag_Y.sum()

        sum_diag2_X = _sqn(diag_X)
        sum_diag2_Y = _sqn(diag_Y)

    Kt_XX_sums = K_XX.sum(axis=1) - diag_X
    Kt_YY_sums = K_YY.sum(axis=1) - diag_Y
    K_XY_sums_0 = K_XY.sum(axis=0)
    K_XY_sums_1 = K_XY.sum(axis=1)

    Kt_XX_sum = Kt_XX_sums.sum()
    Kt_YY_sum = Kt_YY_sums.sum()
    K_XY_sum = K_XY_sums_0.sum()

    if mmd_est == "biased":
        mmd2 = (
            (Kt_XX_sum + sum_diag_X) / (m * m)
            + (Kt_YY_sum + sum_diag_Y) / (m * m)
            - 2 * K_XY_sum / (m * m)
        )
    else:
        assert mmd_est in {"unbiased", "u-statistic"}
        mmd2 = (Kt_XX_sum + Kt_YY_sum) / (m * (m - 1))
        if mmd_est == "unbiased":
            mmd2 -= 2 * K_XY_sum / (m * m)
        else:
            mmd2 -= 2 * (K_XY_sum - np.trace(K_XY)) / (m * (m - 1))

    if not ret_var:
        return mmd2

    Kt_XX_2_sum = _sqn(K_XX) - sum_diag2_X
    Kt_YY_2_sum = _sqn(K_YY) - sum_diag2_Y
    K_XY_2_sum = _sqn(K_XY)

    dot_XX_XY = Kt_XX_sums.dot(K_XY_sums_1)
    dot_YY_YX = Kt_YY_sums.dot(K_XY_sums_0)

    m1 = m - 1
    m2 = m - 2
    zeta1_est = (
        1
        / (m * m1 * m2)
        * (_sqn(Kt_XX_sums) - Kt_XX_2_sum + _sqn(Kt_YY_sums) - Kt_YY_2_sum)
        - 1 / (m * m1) ** 2 * (Kt_XX_sum**2 + Kt_YY_sum**2)
        + 1 / (m * m * m1) * (_sqn(K_XY_sums_1) + _sqn(K_XY_sums_0) - 2 * K_XY_2_sum)
        - 2 / m**4 * K_XY_sum**2
        - 2 / (m * m * m1) * (dot_XX_XY + dot_YY_YX)
        + 2 / (m**3 * m1) * (Kt_XX_sum + Kt_YY_sum) * K_XY_sum
    )
    zeta2_est = (
        1 / (m * m1) * (Kt_XX_2_sum + Kt_YY_2_sum)
        - 1 / (m * m1) ** 2 * (Kt_XX_sum**2 + Kt_YY_sum**2)
        + 2 / (m * m) * K_XY_2_sum
        - 2 / m**4 * K_XY_sum**2
        - 4 / (m * m * m1) * (dot_XX_XY + dot_YY_YX)
        + 4 / (m**3 * m1) * (Kt_XX_sum + Kt_YY_sum) * K_XY_sum
    )
    var_est = (
        4 * (var_at_m - 2) / (var_at_m * (var_at_m - 1)) * zeta1_est
        + 2 / (var_at_m * (var_at_m - 1)) * zeta2_est
    )

    return mmd2, var_est


def AddTime(x):
    def get_time_vector(size: int, length: int) -> torch.Tensor:
        return (
            torch.linspace(1 / length, 1, length).reshape(1, -1, 1).repeat(size, 1, 1)
        )

    t = get_time_vector(x.shape[0], x.shape[1]).to(x.device)
    return torch.cat([t, x], dim=-1)


def Sig_mmd(X, Y, depth):
    # convert torch tensor to numpy
    N, L, C = X.shape
    N1, _, C1 = Y.shape
    X = torch.cat([torch.zeros((N, 1, C)).to(X.device), X], dim=1)
    Y = torch.cat([torch.zeros((N1, 1, C1)).to(X.device), Y], dim=1)
    X = to_numpy(AddTime(X))
    Y = to_numpy(AddTime(Y))
    n_components = 20
    static_kernel = ksig.static.kernels.RBFKernel()
    # an RBF base kernel for vector-valued data which is lifted to a kernel for sequences
    static_feat = ksig.static.features.NystroemFeatures(
        static_kernel, n_components=n_components
    )
    # Nystroem features with an RBF base kernel

    proj = ksig.projections.CountSketchRandomProjection(n_components=n_components)
    # a CountSketch random projection

    lr_sig_kernel = ksig.kernels.LowRankSignatureKernel(
        n_levels=depth, static_features=static_feat, projection=proj
    )
    # sig_kernel = ksig.kernels.SignatureKernel(
    #   n_levels=depth, static_kernel=static_kernel)
    # a SignatureKernel object, which works as a callable for computing the signature kernel matrix
    lr_sig_kernel.fit(X)
    K_XX = lr_sig_kernel(X)  # K_XX has shape (10, 10)
    K_XY = lr_sig_kernel(X, Y)
    K_YY = lr_sig_kernel(Y)
    m = K_XX.shape[0]
    diag_X = np.diagonal(K_XX)
    diag_Y = np.diagonal(K_YY)

    Kt_XX_sums = K_XX.sum(axis=1) - diag_X
    Kt_YY_sums = K_YY.sum(axis=1) - diag_Y
    K_XY_sums_0 = K_XY.sum(axis=0)

    Kt_XX_sum = Kt_XX_sums.sum()
    Kt_YY_sum = Kt_YY_sums.sum()
    K_XY_sum = K_XY_sums_0.sum()
    mmd2 = (Kt_XX_sum + Kt_YY_sum) / (m * (m - 1))
    mmd2 -= 2 * K_XY_sum / (m * m)

    return torch.tensor(mmd2)


class Sig_MMD_loss(Loss):
    def __init__(self, gt, depth=5, **kwargs):
        super().__init__(**kwargs)
        self.gt = gt
        self.depth = depth

    def forward(self, pred):
        return Sig_mmd(self.gt, pred, self.depth).mean()


class cross_correlation(Loss):
    def __init__(self, gt, **kwargs):
        super().__init__(**kwargs)
        self.gt = gt

    def forward(self, pred):
        fake_corre = torch.from_numpy(
            np.corrcoef(pred.mean(1).permute(1, 0))
        ).float()
        real_corre = torch.from_numpy(
            np.corrcoef(self.gt.mean(1).permute(1, 0))
        ).float()
        return torch.abs(fake_corre - real_corre).mean()


def compute_expected_signature(
    x_path, depth: int, augmentations: Tuple, normalise: bool = True
):
    x_path_augmented = apply_augmentations(x_path, augmentations)
    expected_signature = signatory.signature(x_path_augmented, depth=depth).mean(0)
    dim = x_path_augmented.shape[2]
    count = 0
    if normalise:
        for i in range(depth):
            expected_signature[count : count + dim ** (i + 1)] = expected_signature[
                count : count + dim ** (i + 1)
            ] * math.factorial(i + 1)
            count = count + dim ** (i + 1)
    return expected_signature


def rmse(x, y):
    return (x - y).pow(2).sum().sqrt()


class SigW1Metric:
    def __init__(
        self,
        depth: int,
        gt: torch.Tensor,
        augmentations: Optional[Tuple] = (Scale(),),
        normalise: bool = True,
    ):
        assert (
            len(gt.shape) == 3
        ), "Path needs to be 3-dimensional. Received %s dimension(s)." % (
            len(gt.shape),
        )

        self.augmentations = augmentations
        self.depth = depth
        self.n_lags = gt.shape[1]

        self.normalise = normalise
        self.expected_signature_mu = compute_expected_signature(
            gt, depth, augmentations, normalise
        )

    def __call__(self, x_path_nu: torch.Tensor):
        """Computes the SigW1 metric."""
        device = x_path_nu.device
        expected_signature_nu = compute_expected_signature(
            x_path_nu, self.depth, self.augmentations, self.normalise
        )
        loss = rmse(self.expected_signature_mu.to(device), expected_signature_nu)
        return loss


class SigW1Loss(Loss):
    def __init__(self, gt, depth=4, **kwargs):
        # Extract SigW1Metric-specific kwargs; remainder flows to Loss base.
        metric_kwargs = {}
        for key in ("augmentations", "normalise"):
            if key in kwargs:
                metric_kwargs[key] = kwargs.pop(key)
        super().__init__(**kwargs)
        self.sig_w1_metric = SigW1Metric(gt=gt, depth=depth, **metric_kwargs)

    def forward(self, pred):
        return self.sig_w1_metric(pred).mean()


def diff(x):
    return x[:, 1:] - x[:, :-1]


def is_multivariate(x: torch.Tensor):
    """Check if the path / tensor is multivariate."""
    return True if x.shape[-1] > 1 else False


def sig_mmd_permutation_test(X, Y, num_permutation) -> float:
    """Run a two-sample permutation test for signature-MMD equality.

    Args:
        X: Batch of samples of shape ``(N, C)`` or ``(N, T, C)``.
        Y: Batch of samples of shape ``(N, C)`` or ``(N, T, C)``.
        num_permutation: Number of permutations to draw for the null
            distribution.

    Returns:
        ``(power, type1_error)`` — test power and empirical Type I error.
    """
    # compute H1 statistics
    # test_func.eval()

    # We first split the data X into two subsets
    idx = torch.randint(X.shape[0], (X.shape[0],))

    X1 = X[idx[-int(X.shape[0] // 2) :]]
    X = X[idx[: -int(X.shape[0] // 2)]]

    with torch.no_grad():
        t0 = Sig_mmd(X, X1, depth=5).cpu().detach().numpy()
        t1 = Sig_mmd(X, Y, depth=5).cpu().detach().numpy()
        print(t1)
        n, m = X.shape[0], Y.shape[0]
        combined = torch.cat([X, Y])

        statistics = []

        for i in range(num_permutation):
            idx1 = torch.randperm(n + m)

            statistics.append(Sig_mmd(combined[idx1[:n]], combined[idx1[n:]], depth=5))
            # print(statistics)
        # print(np.array(statistics))
    power = (
        t1 > torch.tensor(statistics).cpu().detach().numpy()
    ).sum() / num_permutation
    type1_error = (
        1
        - (t0 > torch.tensor(statistics).cpu().detach().numpy()).sum() / num_permutation
    )
    return power, type1_error


def histogram_torch_update(x, bins, density=True):
    """Compute the histogram of a tensor using provided bins, ignoring NaNs.

    Args:
        x: Input tensor.
        bins: Precomputed bin edges.
        density: When True, normalise the histogram.

    Returns:
        ``(count, bins)`` — bin counts and bin edges, both on the device
        of ``x``.
    """
    # Remove NaNs from the input tensor
    x = x[~torch.isnan(x)]
    # Return zero count if no valid entries remain
    if x.numel() == 0:
        logger.warning(
            "There are only NaNs in the input tensor for histogram computation."
        )
        return torch.zeros(len(bins) - 1, device=x.device), bins

    n_bins = len(bins) - 1

    # torch.histogram does not exist before 1.10 pytorch.
    count = torch.histc(x, bins=n_bins, min=bins[0].item(), max=bins[-1].item())
    if density:
        # Divide by the total count but multiple by n_bins so the average bin height is 1.
        count = count / x.numel() * n_bins
    return count, bins


class HistogramLoss(Loss):
    @staticmethod
    def precompute_histograms(x: torch.Tensor, n_bins: int):
        densities: typing.List = []
        center_bin_locs: typing.List = []
        bin_widths: typing.List = []
        bin_edges: typing.List = []
        sample_sizes: typing.List = []

        for time_step in range(x.shape[1]):
            per_time_densities = []
            per_time_center_bin_locs = []
            per_time_bin_widths = []
            feature_bins = []
            sample_size = []
            for feature_idx in range(x.shape[2]):
                x_ti = x[:, time_step, feature_idx].reshape(-1)
                x_ti = x_ti[~torch.isnan(x_ti)]  # Remove NaNs

                if x_ti.numel() == 0:
                    # Handle the case where all values are NaNs by appending a tensor of zeros
                    per_time_densities.append(torch.zeros(n_bins, device=x.device))
                    per_time_center_bin_locs.append(
                        torch.zeros(n_bins, device=x.device)
                    )
                    per_time_bin_widths.append(
                        torch.tensor(1.0, device=x.device)
                    )  # Default bin width
                    feature_bins.append(torch.zeros(n_bins + 1, device=x.device))
                    sample_size.append(torch.tensor(0, device=x.device))
                    continue

                min_val, max_val = x_ti.min().item(), x_ti.max().item()
                # We catch here the case when the values are all the same for a time and feature.
                if len(x_ti) > 1 and abs(max_val - min_val) < 1e-10:
                    max_val = max_val + 1e-5
                    min_val = min_val - 1e-5
                    logger.warning(
                        "All values are the same for a time and feature. "
                        "Adding a small perturbation to the range. "
                        "The loss might not be as representative as desired."
                    )
                elif len(x_ti) == 1:
                    max_val = max_val + 1e-5
                    min_val = min_val - 1e-5

                bins = torch.linspace(min_val, max_val, n_bins + 1, device=x.device)
                density, bins = histogram_torch_update(x_ti, bins, density=True)
                per_time_densities.append(density)
                bin_width = bins[1] - bins[0]
                center_bin_loc = 0.5 * (bins[1:] + bins[:-1])
                per_time_center_bin_locs.append(center_bin_loc)
                per_time_bin_widths.append(bin_width)
                feature_bins.append(bins)
                sample_size.append(torch.tensor(x_ti.numel(), device=x.device))

            densities.append(per_time_densities)
            center_bin_locs.append(per_time_center_bin_locs)
            bin_widths.append(per_time_bin_widths)
            bin_edges.append(feature_bins)
            sample_sizes.append(sample_size)

        # For all time stamps, they should be the same dimensions, hence stackable.
        # Can't do ParamList of ParamList. First nest per feature, second per time and inside per bin.
        densities: typing.List = [torch.stack(d) for d in densities]
        center_bin_locs: typing.List = [torch.stack(loc) for loc in center_bin_locs]
        bin_widths: typing.List = [torch.stack(d) for d in bin_widths]
        bin_edges: typing.List = [torch.stack(b) for b in bin_edges]
        sample_sizes: typing.List = [torch.stack(s) for s in sample_sizes]

        return densities, center_bin_locs, bin_widths, bin_edges, sample_sizes

    def __init__(self, gt: torch.Tensor, n_bins: int = 50, **kwargs):
        """Initialise the HistogramLoss with the real data distribution.

        Args:
            gt: Real data tensor of shape ``(N, L, D)``.
            n_bins: Number of bins for the histograms.
            **kwargs: Forwarded to :class:`Loss` (``name``, ``reg``, …).
        """
        super().__init__(**kwargs)
        self.n_bins = n_bins
        self.num_samples, self.num_time_steps, self.num_features = gt.shape

        logger.debug(
            (
                f"Initializing HistogramLoss with {self.num_samples} samples, "
                f"{self.num_time_steps} time steps, and {self.num_features} features."
            )
        )

        (
            self.densities,
            self.center_bin_locs,
            self.bin_widths,
            self.bin_edges,
            self.sample_sizes,
        ) = self.precompute_histograms(gt, n_bins)

        self.densities = nn.ParameterList(
            [nn.Parameter(density, requires_grad=False) for density in self.densities]
        )
        self.center_bin_locs = nn.ParameterList(
            [nn.Parameter(loc, requires_grad=False) for loc in self.center_bin_locs]
        )
        self.bin_widths = nn.ParameterList(
            [
                nn.Parameter(bin_width, requires_grad=False)
                for bin_width in self.bin_widths
            ]
        )
        # Bin edges are retained for plotting, not loss computation.
        self.bin_edges = nn.ParameterList(
            [nn.Parameter(bin, requires_grad=False) for bin in self.bin_edges]
        )

        self.sample_sizes = nn.Parameter(
            torch.stack(self.sample_sizes, dim=0), requires_grad=False
        )
        return

    def _compute(self, pred):
        """Compute the histogram loss between real and fake distributions.

        Internal helper called from ``forward``. Renamed from ``compute`` to
        comply with R6 (the ``Loss`` base has no ``forward → compute``
        wrapper, so a public ``compute`` method is misleading). The
        comparison is dodgy in the Dirac-measure regime — use with caution.

        Args:
            pred: Fake data tensor of shape ``(N, L, D)``.

        Returns:
            Component-wise loss of shape ``(L, D)`` — loss per time per
            feature.
        """
        # check if pred in the same device as the gt
        if pred.device != self.densities[0].device:
            logger.warning(
                f"pred is on device {pred.device}, but densities are on {self.densities[0].device}. Moving pred to the correct device."
            )
            pred = pred.to(self.densities[0].device)
        assert (
            pred.shape[2] == self.num_features
        ), f"Expected {self.num_features} features in pred, but got {pred.shape[2]}."
        assert (
            pred.shape[1] == self.num_time_steps
        ), f"Expected {self.num_time_steps} time steps in pred, but got {pred.shape[1]}."

        all_losses: typing.List = []
        # To store time steps with NaNs
        nan_features_per_time_step: typing.List[
            typing.Tuple[int, typing.List[int]]
        ] = []

        for time_step in range(self.num_time_steps):
            per_time_losses = []
            nan_features: typing.List[
                int
            ] = []  # Collect indices with NaNs for this time step
            for feature_idx in range(self.num_features):
                # Localisation of the bins
                loc: torch.Tensor = self.center_bin_locs[time_step][feature_idx]
                if (loc < 1e-16).all():
                    # Means it is a case when the distribution computed had no values
                    # for that feature so the error has to be maximal.
                    per_time_losses.append(torch.tensor(2.0, device=pred.device))
                    continue

                # Fake samples at time step t for feature i
                x_ti = pred[:, time_step, feature_idx].reshape(-1)
                x_ti = x_ti[~torch.isnan(x_ti)].reshape(-1)  # Remove NaNs
                if x_ti.numel() == 0:
                    # Record feature index with all NaNs
                    nan_features.append(feature_idx)
                    # Maximal loss for this time-feature because the discrepancy is total.
                    per_time_losses.append(torch.tensor(2.0, device=pred.device))
                    nan_features_per_time_step.append(
                        (time_step, nan_features)
                    )  # Store if all features are NaNs for this step
                    continue  # Skip if no valid entries after removing NaNs

                # Compute histogram using torch.histc
                # Shape (num_bins)
                density_fake = torch.histc(
                    x_ti,
                    bins=self.n_bins,
                    min=self.bin_edges[time_step][feature_idx][0].item(),
                    max=self.bin_edges[time_step][feature_idx][-1].item(),
                )

                # Normalize fake densities. Dvidie by total number of elements, not just the ones in the range.
                density_fake = density_fake / x_ti.numel() * self.n_bins

                # Compute absolute difference between real and fake densities
                abs_metric = torch.abs(
                    density_fake - self.densities[time_step][feature_idx]
                ).mean()

                # Very fast. Is ok.
                num_samples_oob = torch.sum(
                    (x_ti < self.bin_edges[time_step][feature_idx][0])
                    | (x_ti > self.bin_edges[time_step][feature_idx][-1])
                )

                out_of_bound_error = num_samples_oob / x_ti.numel()

                # Compute final loss by averaging across bins
                per_time_losses.append(abs_metric + out_of_bound_error)
            all_losses.append(torch.stack(per_time_losses))

        # Raise error if no valid data was found for any time step and feature
        if not all_losses:
            logger.error(
                "All time steps and features contain NaNs or empty data, yielding no valid losses."
            )

        all_losses: torch.Tensor = torch.stack(all_losses)
        return (
            all_losses / 2.0
        )  # dividing by 2 to represent the difference between both histograms.

    def forward(self, pred, ignore_features: list = None):
        # Use this
        try:
            if ignore_features is None or (
                hasattr(ignore_features, "__len__") and len(ignore_features) == 0
            ):
                return self._weigh_by_sample_size(self._compute(pred))

            ignore_indices = torch.tensor(ignore_features, dtype=torch.long)
            mask = torch.ones(pred.shape[2], dtype=torch.bool)
            mask[ignore_indices] = False
            return self._weigh_by_sample_size(self._compute(pred))[:, mask].mean()
        except Exception as e:
            logger.error(
                f"Error in the computation of the HistogramLoss. Return maximal error. Details: {e}"
            )
            return torch.tensor(1.0, device=pred.device)

    def _weigh_by_sample_size(self, loss):
        assert loss.shape == self.sample_sizes.shape, (
            "Loss and sample sizes should have the same shape, "
            f"but got {loss.shape} and {self.sample_sizes.shape}."
        )
        return (loss * self.sample_sizes / self.sample_sizes.sum()).sum()


def tail_metric(x, alpha, statistic):
    # DO NOT MODIFY!
    res = list()
    for i in range(x.shape[2]):
        tmp_res = list()
        # Exclude the initial point
        for t in range(x.shape[1]):
            x_ti = x[:, t, i]
            sorted_arr, _ = torch.sort(x_ti)
            var_alpha_index = int(alpha * len(sorted_arr))
            var_alpha = sorted_arr[var_alpha_index]
            if statistic == "es":
                es_values = sorted_arr[: var_alpha_index + 1]
                es_alpha = es_values.mean()
                tmp_res.append(es_alpha)
            else:
                tmp_res.append(var_alpha)
        res.append(tmp_res)
    return res


class VARMetricLoss(Loss):
    """
    Tail distribution discrepancy between real and fake via VaR or ES.

    Uses tail_metric(x, alpha, statistic) on the captured real data and the
    fake samples independently, and returns the mean absolute difference
    over all features and time steps.
    """

    def __init__(
        self,
        gt: torch.Tensor,
        alpha: float = 0.05,
        statistic: str = "var",
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert statistic in {"var", "es"}
        self.gt = gt
        self.alpha = alpha
        self.statistic = statistic
        # Precompute the real-side tail statistic once.
        self._real_tail = tail_metric(gt, self.alpha, self.statistic)

    def forward(self, pred: torch.Tensor):
        # pred: [N, T, D]
        fake_tail = tail_metric(pred, self.alpha, self.statistic)

        real_tail_t = torch.tensor(
            self._real_tail, dtype=pred.dtype, device=pred.device
        )
        fake_tail_t = torch.tensor(
            fake_tail, dtype=pred.dtype, device=pred.device
        )

        return (real_tail_t - fake_tail_t).abs().mean()


class ONNDMetricLoss(Loss):
    """
    Outgoing Nearest-Neighbour Distance:
    For each fake sample, compute the L2 distance to its nearest real sample
    (real samples captured in ``__init__``), and average over all fake samples.
    """

    def __init__(self, gt: torch.Tensor, **kwargs):
        super().__init__(**kwargs)
        self.gt = gt
        # Precompute the flattened real samples once.
        self._real_flat = gt.reshape(gt.size(0), -1)

    def forward(self, pred: torch.Tensor):
        # pred: [N_fake, T, D]
        fake_flat = pred.reshape(pred.size(0), -1)
        real_flat = self._real_flat.to(device=fake_flat.device, dtype=fake_flat.dtype)

        dists = torch.cdist(fake_flat, real_flat, p=2)  # [N_fake, N_real]
        nearest_dist, _ = torch.min(dists, dim=1)
        return nearest_dist.mean()


class ICDMetricLoss(Loss):
    """
    Intra-Class Distance for generated samples.
    Average pairwise L2 distance among fake samples. Small ICD suggests
    potential mode collapse.
    """

    def forward(self, pred: torch.Tensor):
        # pred: [N_fake, T, D]
        fake_flat = pred.reshape(pred.size(0), -1)
        dists = torch.cdist(fake_flat, fake_flat, p=2)  # [N_fake, N_fake]
        return dists.mean()


# Forecasting Metrics
# Reference (gt) is captured in __init__ per RULE A — the agent never has
# direct access to ground truth, so forward() takes only the prediction.
class MSELoss(Loss):
    """Mean squared error between predicted and target."""

    def __init__(self, gt: torch.Tensor, **kwargs):
        super().__init__(**kwargs)
        self.gt = gt.float()

    def forward(self, pred: torch.Tensor):
        return ((pred.float() - self.gt) ** 2).mean()


class RMSELoss(Loss):
    """Root mean squared error."""

    def __init__(self, gt: torch.Tensor, **kwargs):
        super().__init__(**kwargs)
        self.gt = gt.float()

    def forward(self, pred: torch.Tensor):
        return torch.sqrt(((pred.float() - self.gt) ** 2).mean())


class MAELoss(Loss):
    """Mean absolute error."""

    def __init__(self, gt: torch.Tensor, **kwargs):
        super().__init__(**kwargs)
        self.gt = gt.float()

    def forward(self, pred: torch.Tensor):
        return (pred.float() - self.gt).abs().mean()


class MAPELoss(Loss):
    """Mean absolute percentage error."""

    def __init__(self, gt: torch.Tensor, eps: float = 1e-8, **kwargs):
        super().__init__(**kwargs)
        self.gt = gt.float()
        self.eps = eps

    def forward(self, pred: torch.Tensor):
        return (
            (pred.float() - self.gt).abs()
            / (self.gt.abs() + self.eps)
        ).mean()


class DirectionalAccuracy(Loss):
    """Fraction of times the predicted direction matches the true direction."""

    def __init__(self, gt: torch.Tensor, **kwargs):
        super().__init__(**kwargs)
        self.gt = gt.float()

    def forward(self, pred: torch.Tensor):
        sign_pred = torch.sign(pred.float())
        sign_true = torch.sign(self.gt)
        return (sign_pred == sign_true).float().mean()


class SignLoss(Loss):
    """Penalises wrong direction; 0 if correct direction, 1 if incorrect."""

    def __init__(self, gt: torch.Tensor, **kwargs):
        super().__init__(**kwargs)
        self.gt = gt.float()

    def forward(self, pred: torch.Tensor):
        return (torch.sign(pred.float()) != torch.sign(self.gt)).float().mean()


class QuantileLoss(Loss):
    """Pinball loss for quantile regression."""

    def __init__(self, gt: torch.Tensor, quantile: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.gt = gt.float()
        self.q = quantile

    def forward(self, pred: torch.Tensor):
        diff = self.gt - pred.float()
        return torch.max((self.q - 1) * diff, self.q * diff).mean()


class CRPSLoss(Loss):
    """
    Continuous ranked probability score.

    pred_samples: [N, S, D] (S = number of forecast samples)
    gt:         [N, D] — captured at init.
    """

    def __init__(self, gt: torch.Tensor, **kwargs):
        super().__init__(**kwargs)
        self.gt = gt.float()

    def forward(self, pred_samples: torch.Tensor):
        pred_samples = pred_samples.float()
        # Term 1: mean absolute error to true
        mae_term = (pred_samples - self.gt.unsqueeze(1)).abs().mean()

        # Term 2: sample-sample interactions
        diff = pred_samples.unsqueeze(2) - pred_samples.unsqueeze(1)  # [N, S, S, D]
        sample_term = diff.abs().mean() / 2

        return mae_term - sample_term


# Correlation, classification, robust regression, strategy-ratio losses
# Same forecasting contract: gt at __init__, pred at forward, scalar return.
# Bare-name classes return a higher-is-better score; *Loss-suffix classes are
# lower-is-better. Convention is a label only — the assembled evaluator sums
# value * weight without a direction flag, so callers control maximise/minimise
# via the sign of the per-reward weight.
class PearsonCorrelation(Loss):
    """Pearson r between predicted and true vectors. Higher is better; range [-1, 1]."""

    def __init__(self, gt: torch.Tensor, eps: float = 1e-8, **kwargs):
        super().__init__(**kwargs)
        self.gt = gt.float()
        self.eps = eps

    def forward(self, pred: torch.Tensor):
        x = pred.float().flatten()
        y = self.gt.flatten()
        x_c = x - x.mean()
        y_c = y - y.mean()
        denom = torch.sqrt((x_c ** 2).sum() * (y_c ** 2).sum()) + self.eps
        return (x_c * y_c).sum() / denom


class SpearmanCorrelation(Loss):
    """Rank correlation (Information Coefficient). Higher is better; range [-1, 1].

    Eval-only — argsort breaks differentiability; ties get sequential ranks
    rather than averaged ranks.
    """

    def __init__(self, gt: torch.Tensor, eps: float = 1e-8, **kwargs):
        super().__init__(**kwargs)
        self.gt = gt.float()
        self.eps = eps

    def forward(self, pred: torch.Tensor):
        x = pred.float().flatten().argsort().argsort().float()
        y = self.gt.flatten().argsort().argsort().float()
        x_c = x - x.mean()
        y_c = y - y.mean()
        denom = torch.sqrt((x_c ** 2).sum() * (y_c ** 2).sum()) + self.eps
        return (x_c * y_c).sum() / denom


class F1Score(Loss):
    """Macro-F1 of (pred > threshold) vs (gt > threshold). Higher is better."""

    def __init__(
        self,
        gt: torch.Tensor,
        threshold: float = 0.0,
        eps: float = 1e-8,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.gt = gt.float()
        self.threshold = threshold
        self.eps = eps

    def forward(self, pred: torch.Tensor):
        pred_pos = pred.float() > self.threshold
        true_pos = self.gt > self.threshold
        tp = (pred_pos & true_pos).float().sum()
        fp = (pred_pos & ~true_pos).float().sum()
        fn = (~pred_pos & true_pos).float().sum()
        precision = tp / (tp + fp + self.eps)
        recall = tp / (tp + fn + self.eps)
        return 2 * precision * recall / (precision + recall + self.eps)


class R2Score(Loss):
    """Coefficient of determination, 1 - SS_res / SS_tot. Higher is better; (-inf, 1]."""

    def __init__(self, gt: torch.Tensor, eps: float = 1e-8, **kwargs):
        super().__init__(**kwargs)
        self.gt = gt.float()
        self.eps = eps

    def forward(self, pred: torch.Tensor):
        ss_res = ((pred.float() - self.gt) ** 2).sum()
        ss_tot = ((self.gt - self.gt.mean()) ** 2).sum()
        return 1.0 - ss_res / (ss_tot + self.eps)


class HuberLoss(Loss):
    """Quadratic for |err| <= delta, linear beyond. Lower is better."""

    def __init__(self, gt: torch.Tensor, delta: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.gt = gt.float()
        self.delta = delta

    def forward(self, pred: torch.Tensor):
        err = pred.float() - self.gt
        abs_err = err.abs()
        return torch.where(
            abs_err <= self.delta,
            0.5 * err ** 2,
            self.delta * (abs_err - 0.5 * self.delta),
        ).mean()


class SharpeRatioLoss(Loss):
    """Sharpe of strategy returns sign(pred) * gt. Higher is better.

    Loss-form forecasting metric — consumes aligned tensors. Distinct from the
    curated-tasks-only ``TradingReward.SharpeRatio`` further below, which
    consumes per-step trade-history dicts.
    """

    def __init__(self, gt: torch.Tensor, eps: float = 1e-8, **kwargs):
        super().__init__(**kwargs)
        self.gt = gt.float()
        self.eps = eps

    def forward(self, pred: torch.Tensor):
        strat = (torch.sign(pred.float()) * self.gt).flatten()
        if strat.numel() < 2:
            return torch.zeros((), dtype=strat.dtype, device=strat.device)
        return strat.mean() / (strat.std(unbiased=False) + self.eps)


class SortinoRatio(Loss):
    """Mean strategy return / downside-only std. Higher is better."""

    def __init__(self, gt: torch.Tensor, eps: float = 1e-8, **kwargs):
        super().__init__(**kwargs)
        self.gt = gt.float()
        self.eps = eps

    def forward(self, pred: torch.Tensor):
        strat = (torch.sign(pred.float()) * self.gt).flatten()
        if strat.numel() < 1:
            return torch.zeros((), dtype=strat.dtype, device=strat.device)
        downside = torch.clamp(strat, max=0.0)
        return strat.mean() / (torch.sqrt((downside ** 2).mean()) + self.eps)


# Embedding-space Metrics (replacements for model-taking FID/KID)
class EmbeddingFID(Loss):
    """Frechet distance between real and fake embedding distributions.

    Decouples FID from any in-metric feature extractor: the caller precomputes
    embeddings (``[N, D]``) from real and fake data using whatever model is
    appropriate for the task, and this metric only consumes the embeddings.

    Reference statistics (``mu_real``, ``sigma_real``) are precomputed in
    ``__init__`` so repeated ``forward`` calls do not re-reduce the real set.
    """

    def __init__(self, real_emb, **kwargs):
        super().__init__(**kwargs)
        real_np = (
            to_numpy(real_emb)
            if isinstance(real_emb, torch.Tensor)
            else np.asarray(real_emb)
        )
        if real_np.ndim != 2:
            raise ValueError(
                f"EmbeddingFID.real_emb must be 2-D [N, D]; got shape {real_np.shape}"
            )
        self.mu_real = real_np.mean(axis=0)
        self.sigma_real = np.cov(real_np, rowvar=False)

    def forward(self, fake_emb):
        fake_np = (
            to_numpy(fake_emb)
            if isinstance(fake_emb, torch.Tensor)
            else np.asarray(fake_emb)
        )
        if fake_np.ndim != 2:
            raise ValueError(
                f"EmbeddingFID.fake_emb must be 2-D [N, D]; got shape {fake_np.shape}"
            )
        mu_fake = fake_np.mean(axis=0)
        sigma_fake = np.cov(fake_np, rowvar=False)
        return calculate_frechet_distance(
            self.mu_real, self.sigma_real, mu_fake, sigma_fake
        )


class EmbeddingKID(Loss):
    """Kernel Inception Distance over precomputed real/fake embeddings.

    Like :class:`EmbeddingFID` but using polynomial-kernel MMD. ``subset_size``
    is capped to the smaller of the two embedding batch sizes so sampling
    never fails on small datasets.
    """

    def __init__(self, real_emb, **kwargs):
        super().__init__(**kwargs)
        real_np = (
            to_numpy(real_emb)
            if isinstance(real_emb, torch.Tensor)
            else np.asarray(real_emb)
        )
        if real_np.ndim != 2:
            raise ValueError(
                f"EmbeddingKID.real_emb must be 2-D [N, D]; got shape {real_np.shape}"
            )
        self.real_np = real_np

    def forward(self, fake_emb):
        fake_np = (
            to_numpy(fake_emb)
            if isinstance(fake_emb, torch.Tensor)
            else np.asarray(fake_emb)
        )
        if fake_np.ndim != 2:
            raise ValueError(
                f"EmbeddingKID.fake_emb must be 2-D [N, D]; got shape {fake_np.shape}"
            )
        n = min(self.real_np.shape[0], fake_np.shape[0])
        subset_size = max(1, min(n, 1000))
        return polynomial_mmd_averages(fake_np, self.real_np, subset_size=subset_size)


# Structured trading rewards run once per episode and require no gradients.


class TradingReward(ABC):
    """Compute a reward from a list of (prediction, ground_truth) pairs.

    Subclasses must implement :meth:`compute_per_trade`.  The default
    :meth:`compute_aggregate` returns the mean of per-trade values;
    override it for metrics that are only meaningful in aggregate
    (e.g. Sharpe ratio).
    """

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def compute_per_trade(
        self,
        predictions: list[dict[str, Any]],
        ground_truths: list[dict[str, Any]],
    ) -> list[float]:
        """Return one score per (prediction, ground_truth) pair."""
        raise NotImplementedError

    def compute_aggregate(
        self,
        predictions: list[dict[str, Any]],
        ground_truths: list[dict[str, Any]],
    ) -> float:
        """Return a single aggregate score across all trades."""
        per_trade = self.compute_per_trade(predictions, ground_truths)
        if not per_trade:
            return 0.0
        return sum(per_trade) / len(per_trade)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"


def _pnl_list(predictions: list[dict], ground_truths: list[dict]) -> list[float]:
    """Signed PnL per trade, quantity-weighted when quantity is present.

    Backward compatible: if ``quantity`` is absent the default of 1.0
    preserves the original unit-size behaviour.
    """
    out: list[float] = []
    for pred, gt in zip(predictions, ground_truths):
        sign = 1.0 if pred["direction"] == "long" else -1.0
        quantity = float(pred.get("quantity", 1.0))
        out.append(sign * gt["actual_return"] * quantity)
    return out


def _nan_safe_mean(values: list[float]) -> float:
    """Mean of non-NaN values; NaN if all are NaN or list is empty."""
    valid = [v for v in values if not (isinstance(v, float) and math.isnan(v))]
    if not valid:
        return float("nan")
    return sum(valid) / len(valid)


def _derive_predicted_return(pred: dict, gt: dict) -> Optional[float]:
    """Return ``pred["predicted_return"]`` if present, else derive from
    ``(predicted_price - entry_price) / entry_price``.

    Returns ``None`` when neither path yields a value, so callers can
    decide how to handle the missing case (NaN for new metrics, 0.0
    legacy fallback for ``ReturnMAE``).
    """
    pr = pred.get("predicted_return")
    if pr is not None:
        return float(pr)
    pp = pred.get("predicted_price")
    entry = gt.get("entry_price")
    if pp is not None and entry not in (None, 0, 0.0):
        return (float(pp) - float(entry)) / float(entry)
    return None


class DirectionAccuracy(TradingReward):
    """1.0 when predicted direction matches actual, 0.0 otherwise."""

    def __init__(self) -> None:
        super().__init__("direction_accuracy")

    def compute_per_trade(self, predictions: list[dict], ground_truths: list[dict]) -> list[float]:
        results: list[float] = []
        for pred, gt in zip(predictions, ground_truths):
            actual_dir = "long" if gt["actual_return"] > 0 else "short"
            results.append(1.0 if pred["direction"] == actual_dir else 0.0)
        return results


class PnL(TradingReward):
    """Signed profit-and-loss assuming a unit-size position."""

    def __init__(self) -> None:
        super().__init__("pnl")

    def compute_per_trade(self, predictions: list[dict], ground_truths: list[dict]) -> list[float]:
        return _pnl_list(predictions, ground_truths)


class ReturnMAE(TradingReward):
    """Mean absolute error between predicted and actual return magnitude.

    Reads ``predicted_return`` directly when supplied; falls back to
    deriving it from ``predicted_price`` and ``entry_price`` so that
    price-only submissions also score on this metric. Legacy direction-
    only submissions (no return, no price) keep their historical 0.0
    fallback.
    """

    def __init__(self) -> None:
        super().__init__("return_mae")

    def compute_per_trade(self, predictions: list[dict], ground_truths: list[dict]) -> list[float]:
        out: list[float] = []
        for p, g in zip(predictions, ground_truths):
            pr = _derive_predicted_return(p, g)
            if pr is None:
                pr = 0.0
            out.append(abs(pr - g["actual_return"]))
        return out


class SharpeRatio(TradingReward):
    """Annualisation-free Sharpe: mean(PnL) / std(PnL)."""

    def __init__(self) -> None:
        super().__init__("sharpe_ratio")

    def compute_per_trade(self, predictions: list[dict], ground_truths: list[dict]) -> list[float]:
        return _pnl_list(predictions, ground_truths)

    def compute_aggregate(self, predictions: list[dict], ground_truths: list[dict]) -> float:
        pnls = self.compute_per_trade(predictions, ground_truths)
        if len(pnls) < 2:
            return 0.0
        mean_pnl = sum(pnls) / len(pnls)
        var = sum((p - mean_pnl) ** 2 for p in pnls) / len(pnls)
        std = math.sqrt(var)
        return (mean_pnl / std) if std > 0 else 0.0


class MaxDrawdown(TradingReward):
    """Worst peak-to-trough decline in cumulative PnL (negative number)."""

    def __init__(self) -> None:
        super().__init__("max_drawdown")

    def compute_per_trade(self, predictions: list[dict], ground_truths: list[dict]) -> list[float]:
        return _pnl_list(predictions, ground_truths)

    def compute_aggregate(self, predictions: list[dict], ground_truths: list[dict]) -> float:
        pnls = self.compute_per_trade(predictions, ground_truths)
        if not pnls:
            return 0.0
        cumulative = list(itertools.accumulate(pnls))
        peak = cumulative[0]
        max_dd = 0.0
        for val in cumulative:
            peak = max(peak, val)
            max_dd = min(max_dd, val - peak)
        return max_dd


class WinRate(TradingReward):
    """Fraction of trades with positive PnL."""

    def __init__(self) -> None:
        super().__init__("win_rate")

    def compute_per_trade(self, predictions: list[dict], ground_truths: list[dict]) -> list[float]:
        return _pnl_list(predictions, ground_truths)

    def compute_aggregate(self, predictions: list[dict], ground_truths: list[dict]) -> float:
        pnls = self.compute_per_trade(predictions, ground_truths)
        if not pnls:
            return 0.0
        return sum(1 for p in pnls if p > 0) / len(pnls)


class QuantityPnL(TradingReward):
    """PnL with explicit quantity * price-change, for realtime trading.

    Expected prediction dict keys (in addition to ``direction``):
        quantity       float  trade size
        action         str    "buy" | "sell" (optional, falls back to direction)

    Expected ground_truth dict keys:
        entry_price      float
        exit_price       float
        slippage_cost    float (optional, default 0)
        transaction_cost float (optional, default 0)
    """

    def __init__(self) -> None:
        super().__init__("quantity_pnl")

    def compute_per_trade(self, predictions: list[dict], ground_truths: list[dict]) -> list[float]:
        out: list[float] = []
        for pred, gt in zip(predictions, ground_truths):
            qty = float(pred.get("quantity", 1.0))
            entry = float(gt.get("entry_price", 0.0))
            exit_p = float(gt.get("exit_price", 0.0))
            action = pred.get("action", pred.get("direction", "long"))
            sign = 1.0 if action in ("buy", "long") else -1.0
            pnl = sign * qty * (exit_p - entry)
            pnl -= float(gt.get("slippage_cost", 0.0))
            pnl -= float(gt.get("transaction_cost", 0.0))
            out.append(pnl)
        return out


# Price metrics compare ``predicted_price`` with ``exit_price`` and derive
# return metrics from ``entry_price``. Missing inputs produce NaN so legacy
# direction-only records do not contaminate aggregates.


class PriceMSE(TradingReward):
    """Mean-squared error between predicted price and realised exit price.

    NaN per-trade when ``predicted_price`` or ``exit_price`` is missing;
    aggregate is NaN-skip mean.
    """

    def __init__(self) -> None:
        super().__init__("price_mse")

    def compute_per_trade(self, predictions: list[dict], ground_truths: list[dict]) -> list[float]:
        out: list[float] = []
        for pred, gt in zip(predictions, ground_truths):
            pp = pred.get("predicted_price")
            ep = gt.get("exit_price")
            if pp is None or ep is None:
                out.append(float("nan"))
                continue
            out.append((float(pp) - float(ep)) ** 2)
        return out

    def compute_aggregate(self, predictions: list[dict], ground_truths: list[dict]) -> float:
        return _nan_safe_mean(self.compute_per_trade(predictions, ground_truths))


class PriceRMSE(TradingReward):
    """Root-mean-squared error on price.

    Per-trade returns the squared error so the aggregate can compute
    ``sqrt(mean(squared_errors))`` over the full session. The per-trade
    list is intentionally not the per-trade RMSE (which would be the
    abs error and equal to PriceMAE) — the aggregation is what makes
    this class meaningful.
    """

    def __init__(self) -> None:
        super().__init__("price_rmse")

    def compute_per_trade(self, predictions: list[dict], ground_truths: list[dict]) -> list[float]:
        out: list[float] = []
        for pred, gt in zip(predictions, ground_truths):
            pp = pred.get("predicted_price")
            ep = gt.get("exit_price")
            if pp is None or ep is None:
                out.append(float("nan"))
                continue
            out.append((float(pp) - float(ep)) ** 2)
        return out

    def compute_aggregate(self, predictions: list[dict], ground_truths: list[dict]) -> float:
        mean_sq = _nan_safe_mean(self.compute_per_trade(predictions, ground_truths))
        if math.isnan(mean_sq):
            return float("nan")
        return math.sqrt(mean_sq)


class PriceMAE(TradingReward):
    """Mean-absolute error between predicted and exit price."""

    def __init__(self) -> None:
        super().__init__("price_mae")

    def compute_per_trade(self, predictions: list[dict], ground_truths: list[dict]) -> list[float]:
        out: list[float] = []
        for pred, gt in zip(predictions, ground_truths):
            pp = pred.get("predicted_price")
            ep = gt.get("exit_price")
            if pp is None or ep is None:
                out.append(float("nan"))
                continue
            out.append(abs(float(pp) - float(ep)))
        return out

    def compute_aggregate(self, predictions: list[dict], ground_truths: list[dict]) -> float:
        return _nan_safe_mean(self.compute_per_trade(predictions, ground_truths))


class PriceMAPE(TradingReward):
    """Mean-absolute percentage error on price.

    Scale-free; the only price metric safe for cross-symbol macro
    aggregation.  NaN when ``exit_price`` is below ``eps`` (avoids divide-
    by-zero blow-up on rare zero/near-zero-priced assets).
    """

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__("price_mape")
        self._eps = eps

    def compute_per_trade(self, predictions: list[dict], ground_truths: list[dict]) -> list[float]:
        out: list[float] = []
        for pred, gt in zip(predictions, ground_truths):
            pp = pred.get("predicted_price")
            ep = gt.get("exit_price")
            if pp is None or ep is None:
                out.append(float("nan"))
                continue
            denom = abs(float(ep))
            if denom < self._eps:
                out.append(float("nan"))
                continue
            out.append(abs(float(pp) - float(ep)) / denom)
        return out

    def compute_aggregate(self, predictions: list[dict], ground_truths: list[dict]) -> float:
        return _nan_safe_mean(self.compute_per_trade(predictions, ground_truths))


class PriceR2(TradingReward):
    """Coefficient of determination on (predicted_price, exit_price) pairs.

    Aggregate-only metric: ``compute_per_trade`` returns NaN per element
    because R² requires the full series. ``compute_aggregate`` rebuilds
    valid pairs and applies ``1 - SS_res / SS_tot``. Returns NaN when
    fewer than 2 valid pairs exist or ground truth has zero variance
    (denominator collapse).
    """

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__("price_r2")
        self._eps = eps

    def compute_per_trade(self, predictions: list[dict], ground_truths: list[dict]) -> list[float]:
        return [float("nan")] * len(predictions)

    def compute_aggregate(self, predictions: list[dict], ground_truths: list[dict]) -> float:
        pairs: list[tuple[float, float]] = []
        for pred, gt in zip(predictions, ground_truths):
            pp = pred.get("predicted_price")
            ep = gt.get("exit_price")
            if pp is None or ep is None:
                continue
            pairs.append((float(pp), float(ep)))
        if len(pairs) < 2:
            return float("nan")
        gts = [g for _, g in pairs]
        gt_mean = sum(gts) / len(gts)
        ss_tot = sum((g - gt_mean) ** 2 for g in gts)
        if ss_tot < self._eps:
            return float("nan")
        ss_res = sum((p - g) ** 2 for p, g in pairs)
        return 1.0 - ss_res / ss_tot


class PricePearson(TradingReward):
    """Pearson correlation between predicted and exit prices.

    Aggregate-only; same NaN-on-low-variance / fewer-than-2-pairs
    semantics as :class:`PriceR2`.
    """

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__("price_pearson")
        self._eps = eps

    def compute_per_trade(self, predictions: list[dict], ground_truths: list[dict]) -> list[float]:
        return [float("nan")] * len(predictions)

    def compute_aggregate(self, predictions: list[dict], ground_truths: list[dict]) -> float:
        pairs: list[tuple[float, float]] = []
        for pred, gt in zip(predictions, ground_truths):
            pp = pred.get("predicted_price")
            ep = gt.get("exit_price")
            if pp is None or ep is None:
                continue
            pairs.append((float(pp), float(ep)))
        if len(pairs) < 2:
            return float("nan")
        n = len(pairs)
        mean_p = sum(p for p, _ in pairs) / n
        mean_g = sum(g for _, g in pairs) / n
        cov = sum((p - mean_p) * (g - mean_g) for p, g in pairs)
        var_p = sum((p - mean_p) ** 2 for p, _ in pairs)
        var_g = sum((g - mean_g) ** 2 for _, g in pairs)
        denom = math.sqrt(var_p * var_g)
        if denom < self._eps:
            return float("nan")
        return cov / denom


class ReturnMSE(TradingReward):
    """MSE between predicted and actual return.

    ``predicted_return`` is read directly when supplied; otherwise
    derived from ``(predicted_price - entry_price) / entry_price`` so
    price-only submissions still produce a return-error reading.
    NaN-skip aggregation keeps direction-only submissions from poisoning
    the panel.
    """

    def __init__(self) -> None:
        super().__init__("return_mse")

    def compute_per_trade(self, predictions: list[dict], ground_truths: list[dict]) -> list[float]:
        out: list[float] = []
        for pred, gt in zip(predictions, ground_truths):
            pr = _derive_predicted_return(pred, gt)
            ar = gt.get("actual_return")
            if pr is None or ar is None:
                out.append(float("nan"))
                continue
            out.append((pr - float(ar)) ** 2)
        return out

    def compute_aggregate(self, predictions: list[dict], ground_truths: list[dict]) -> float:
        return _nan_safe_mean(self.compute_per_trade(predictions, ground_truths))


class ReturnRMSE(TradingReward):
    """RMSE between predicted and actual return — sqrt of mean ReturnMSE."""

    def __init__(self) -> None:
        super().__init__("return_rmse")

    def compute_per_trade(self, predictions: list[dict], ground_truths: list[dict]) -> list[float]:
        out: list[float] = []
        for pred, gt in zip(predictions, ground_truths):
            pr = _derive_predicted_return(pred, gt)
            ar = gt.get("actual_return")
            if pr is None or ar is None:
                out.append(float("nan"))
                continue
            out.append((pr - float(ar)) ** 2)
        return out

    def compute_aggregate(self, predictions: list[dict], ground_truths: list[dict]) -> float:
        mean_sq = _nan_safe_mean(self.compute_per_trade(predictions, ground_truths))
        if math.isnan(mean_sq):
            return float("nan")
        return math.sqrt(mean_sq)


# Convenience registry of all trading reward classes
ALL_TRADING_REWARDS: list[type[TradingReward]] = [
    DirectionAccuracy,
    PnL,
    ReturnMAE,
    SharpeRatio,
    MaxDrawdown,
    WinRate,
    QuantityPnL,
    PriceMSE,
    PriceRMSE,
    PriceMAE,
    PriceMAPE,
    PriceR2,
    PricePearson,
    ReturnMSE,
    ReturnRMSE,
]


# Binary-event rewards compare ``predicted_yes_probability`` with ``outcome``.
# The resolver drops ambiguous 0.5 outcomes before scoring.


# Clamp probabilities away from {0, 1} for log-loss to avoid log(0).
_LOG_LOSS_EPS = 1e-12


class EventReward(ABC):
    """Compute a reward over (probability, binary-outcome) pairs.

    Subclasses must implement :meth:`compute_aggregate`. Unlike
    ``TradingReward``, there is no per-trade variant — most calibration
    metrics (Brier, log-loss, ECE) are naturally aggregate.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def compute_aggregate(
        self,
        predictions: list[dict[str, Any]],
        ground_truths: list[dict[str, Any]],
    ) -> float:
        """Return a single aggregate score across all predictions.

        Implementations should return ``0.0`` when ``predictions`` is
        empty (no scoring possible — not a failure).
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"


def _event_pairs(
    predictions: list[dict[str, Any]],
    ground_truths: list[dict[str, Any]],
) -> list[tuple[float, float]]:
    """Zip into ``(p, y)`` pairs, skipping rows with NaN/None on either side.

    Defensive: even though the resolver should drop 0.5 outcomes
    upstream, also skip them here so a misuse can't crash scoring.
    """
    pairs: list[tuple[float, float]] = []
    for pred, gt in zip(predictions, ground_truths):
        p = pred.get("predicted_yes_probability")
        y = gt.get("outcome")
        if p is None or y is None:
            continue
        try:
            p_f = float(p)
            y_f = float(y)
        except (TypeError, ValueError):
            continue
        if math.isnan(p_f) or math.isnan(y_f):
            continue
        if abs(y_f - 0.5) < 1e-9:
            continue  # ambiguous; should have been dropped upstream
        pairs.append((p_f, y_f))
    return pairs


class EventBrierScore(EventReward):
    """Brier score: mean squared error between probability and outcome.

    Range: ``[0, 1]``. Lower is better. ``0`` = perfect prediction;
    ``0.25`` = uniform 0.5 prior on every market.
    """

    def __init__(self) -> None:
        super().__init__("brier_score")

    def compute_aggregate(
        self,
        predictions: list[dict[str, Any]],
        ground_truths: list[dict[str, Any]],
    ) -> float:
        pairs = _event_pairs(predictions, ground_truths)
        if not pairs:
            return 0.0
        return sum((p - y) ** 2 for p, y in pairs) / len(pairs)


class EventLogLoss(EventReward):
    """Binary cross-entropy: ``-mean(y·log(p) + (1-y)·log(1-p))``.

    Probabilities are clipped to ``[1e-12, 1 - 1e-12]`` so a
    confident-and-wrong prediction yields a finite (but large) loss
    rather than infinity. Lower is better.
    """

    def __init__(self) -> None:
        super().__init__("log_loss")

    def compute_aggregate(
        self,
        predictions: list[dict[str, Any]],
        ground_truths: list[dict[str, Any]],
    ) -> float:
        pairs = _event_pairs(predictions, ground_truths)
        if not pairs:
            return 0.0
        total = 0.0
        for p, y in pairs:
            p_clip = min(max(p, _LOG_LOSS_EPS), 1.0 - _LOG_LOSS_EPS)
            total += -(y * math.log(p_clip) + (1.0 - y) * math.log(1.0 - p_clip))
        return total / len(pairs)


class EventCalibrationError(EventReward):
    """Expected Calibration Error (ECE) over 10 equal-width probability bins.

    For each bin, computes ``|mean_predicted_prob - mean_outcome|`` and
    averages weighted by bin size. Lower is better. ``0`` = perfectly
    calibrated; e.g. of all predictions in the 0.7-0.8 bin, 75% should
    have outcome=1.

    Note: ECE has known limitations with small sample sizes and bin
    boundary effects — interpret alongside Brier and log-loss rather
    than as a single number.
    """

    _N_BINS = 10

    def __init__(self) -> None:
        super().__init__("expected_calibration_error")

    def compute_aggregate(
        self,
        predictions: list[dict[str, Any]],
        ground_truths: list[dict[str, Any]],
    ) -> float:
        pairs = _event_pairs(predictions, ground_truths)
        if not pairs:
            return 0.0
        n = len(pairs)
        # Bin index = floor(p * N_BINS), clipped to [0, N_BINS-1] so
        # p == 1.0 lands in the top bin rather than overflowing.
        bins: dict[int, list[tuple[float, float]]] = {}
        for p, y in pairs:
            idx = min(self._N_BINS - 1, max(0, int(p * self._N_BINS)))
            bins.setdefault(idx, []).append((p, y))
        ece = 0.0
        for bucket in bins.values():
            mean_p = sum(p for p, _ in bucket) / len(bucket)
            mean_y = sum(y for _, y in bucket) / len(bucket)
            weight = len(bucket) / n
            ece += weight * abs(mean_p - mean_y)
        return ece


# Convenience registry of all event reward classes — peers
# ``ALL_TRADING_REWARDS`` and is consumed by the resolver when scoring
# event-resolution sessions.
ALL_EVENT_REWARDS: list[type[EventReward]] = [
    EventBrierScore,
    EventLogLoss,
    EventCalibrationError,
]
