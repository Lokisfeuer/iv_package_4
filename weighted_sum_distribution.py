import numpy as np
from scipy import stats


def weighted_sum_distribution(prob_dists, weights=None, n_points=10000, quantiles=(1e-3, 1 - 1e-3)):
    """
    Compute the distribution of Y = sum_i (w_i * X_i), where each X_i is drawn from a given
    SciPy.stats frozen distribution and w_i are weights.

    This version fixes a bug in which the common grid for re-sampling the scaled PDFs was defined
    on the Minkowski sum of the effective supports. Instead, we now build a grid that spans the
    union of the individual supports so that no density is inadvertently truncated.

    Parameters:
      prob_dists : list
          A list of SciPy.stats frozen distributions (each must have .pdf and .ppf methods).
      weights : array-like, optional
          Weights for each distribution. If None, uniform weights are used.
          In any case, weights are normalized to sum to 1.
      n_points : int, optional
          Number of points used for discretizing each distribution’s effective support.
      quantiles : tuple, optional
          The quantiles (low, high) that define the effective support for each distribution.

    Returns:
      final_dist : scipy.stats frozen distribution
          An rv_histogram object representing the distribution of Y.

    Note:
      The density for a scaled variable Y = w * X transforms as
          pdf_Y(y) = (1 / |w|) * f_X(y / w),
      and the density for Y = sum_i Y_i is computed by iterative convolution.
    """

    n = len(prob_dists)

    # Normalize weights; use uniform weights if None.
    if weights is None:
        weights = np.ones(n) / n
    else:
        weights = np.array(weights, dtype=float)
        weights /= weights.sum()

    # For each distribution, use the quantiles to define the effective support for X,
    # then scale to Y = w * X. Also record the native grid spacing (dx).
    supports = []
    dxs = []
    for i, dist in enumerate(prob_dists):
        w = weights[i]
        lower_x = dist.ppf(quantiles[0])
        upper_x = dist.ppf(quantiles[1])
        lower_y = w * lower_x
        upper_y = w * upper_x
        supports.append((lower_y, upper_y))
        dxs.append((upper_y - lower_y) / (n_points - 1))

    # --- BUG FIX: Build a common grid that covers the union of all individual supports.
    # Instead of summing lower and upper bounds (Minkowski sum), we take the minimum lower_y and
    # maximum upper_y so that every distribution’s native support is fully included.
    common_lower = min(s[0] for s in supports)
    common_upper = max(s[1] for s in supports)
    common_dx = min(dxs)
    n_grid = int(np.ceil((common_upper - common_lower) / common_dx)) + 1
    common_grid = np.linspace(common_lower, common_upper, n_points)

    # Evaluate every scaled density on the common grid.
    # TODO: Understand function and its implementation
    # For Y = w * X: pdf_Y(y) = (1/|w|) * pdf_X(y/w) for y in [lower_y, upper_y] of that distribution.
    pdfs_common = []
    for i, dist in enumerate(prob_dists):
        w = weights[i]
        lower_y, upper_y = supports[i]
        pdf_vals = np.zeros_like(common_grid)
        mask = (common_grid >= lower_y) & (common_grid <= upper_y)
        pdf_vals[mask] = (1.0 / abs(w)) * dist.pdf(common_grid[mask] / w)
        pdfs_common.append(pdf_vals)

    # Iteratively convolve the densities.
    # Since all PDFs are sampled on the same common_grid (with spacing common_dx) the integration
    # is consistent in the convolution.
    current_pdf = pdfs_common[0]
    current_grid = common_grid.copy()  # grid for the first PDF

    for i in range(1, n):
        new_pdf = np.convolve(current_pdf, pdfs_common[i], mode='full') * common_dx
        # When convolving functions sampled on a grid,
        # the new support extends from (current_grid[0] + common_grid[0])
        # to (current_grid[-1] + common_grid[-1]).
        new_grid_lower = current_grid[0] + common_grid[0]
        new_grid_upper = current_grid[-1] + common_grid[-1]
        n_new = len(new_pdf)
        new_grid = np.linspace(new_grid_lower, new_grid_upper, n_new)
        current_pdf = new_pdf
        current_grid = new_grid

    # Normalize the final PDF.
    area = np.trapezoid(current_pdf, current_grid)
    current_pdf /= area

    # Build bin edges roughly centered on the grid points.
    dx_final = current_grid[1] - current_grid[0]
    bins = np.concatenate(([current_grid[0] - dx_final / 2],
                           current_grid + dx_final / 2))

    # Return the frozen distribution via rv_histogram.
    final_dist = stats.rv_histogram((current_pdf, bins))
    return final_dist


# --- Example usage ---
if __name__ == "__main__":
    from scipy import stats

    # For demonstration, combine a standard normal and a uniform distribution.
    normal_dist = stats.norm(loc=0, scale=1)
    uniform_dist = stats.uniform(loc=-1, scale=2)  # Uniform on [-1, 1]
    final = weighted_sum_distribution([normal_dist, uniform_dist], weights=[0.6, 0.4])
    xs = np.linspace(-4, 4, 200)
    pdf_vals = final.pdf(xs)
    print("x-value\tpdf")
    for x_val, p_val in zip(xs[::10], pdf_vals[::10]):
        print(f"{x_val:.2f}\t{p_val:.4f}")
