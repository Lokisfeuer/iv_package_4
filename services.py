import numpy as np
import scipy.stats as stats
from scipy.integrate import quad
from iv8 import IV
import warnings


def independent_validation(classifier, X, y, key="bacc", n=float("inf"), return_mean=True, plot=False,
                           iv_start_trainset_size=2,
                           iv_batch_size=None, iv_n_batches=None,
                           mcmc_num_samples=1000, mcmc_step_size=0.1,
                           mcmc_burn_in=10000, mcmc_thin=50, mcmc_random_seed=None):
    # TODO: Unify thinning default value for this function and iv method!
    """
    Perform Independent Validation (IV) on an estimator and extract accuracy metrics.

    This function serves as a high‐level wrapper that:
      1. Initializes an IV instance with the given features (X), labels (y), and classifier.
      2. Runs the incremental (independent) validation process.
      3. Computes the posterior distributions for the evolving accuracy via MCMC.
      4. Retrieves the requested accuracy distribution (or its mean), optionally plotting it.

    Parameters:
      classifier : estimator
          An estimator (classifier) implementing the fit and predict methods.
      X : array-like
          Feature matrix.
      y : array-like
          Target vector (labels).
      key : str or int, optional
          Specifies which accuracy distribution to extract:
            • "bacc" for balanced accuracy,
            • "acc" for overall (weighted) accuracy,
            • Or a single label (e.g. an int) for label-specific accuracy.
          Default is "bacc".
      n : int, float, or str, optional
          The training set size at which to evaluate the accuracy distribution. Use a numeric value
          (e.g. 50) or float("inf") to get the asymptotic accuracy. Alternatively, setting n to
          "development" (case insensitive) will trigger development analysis (over multiple training sizes).
          Default is float("inf").
      return_mean : bool, optional
          If True, the function returns the mean of the accuracy distribution.
          Note: When performing development analysis (n="development") or if multiple keys are provided,
          returning a single mean is not supported.
          Default is True.
      plot : bool or str, optional
          If True, the accuracy distribution is plotted in a new figure.
          If given as a string, that string is used as a filename to save the plot.
          Default is False.
      iv_start_trainset_size : int, optional
          The initial number of samples used for training in the IV process.
          Default is 2.
      iv_batch_size : int, optional
          Batch size to use when validating additional samples.
      iv_n_batches : int, optional
          Alternatively, specify the desired number of batches (only one of iv_batch_size or iv_n_batches
          may be provided). If iv_batch_size is omitted, it is computed as the length of X divided by iv_n_batches,
          rounded up.
      mcmc_num_samples : int, optional
          Number of MCMC samples to draw (after thinning and burn-in) for posterior computation.
          Default is 1000.
      mcmc_step_size : float, optional
          Step size used for the MCMC proposal distribution.
          Default is 0.05.
      mcmc_burn_in : int, optional
          Number of iterations to discard as burn-in for the MCMC chain.
          Default is 100.
      mcmc_thin : int, optional
          Thinning interval for the MCMC chain.
          Default is 50.
      mcmc_random_seed : int, optional
          Seed for making the MCMC process reproducible.
          Default is None.

    Returns:
      If return_mean is True:
          float
              The mean value of the requested accuracy distribution.
      Otherwise:
          A frozen SciPy distribution (rv_histogram) if a single key is specified,
          or a dictionary mapping (key, n) pairs to frozen distributions if multiple keys are used.

    Raises:
      ValueError:
          - When both iv_batch_size and iv_n_batches are provided.
          - When return_mean is requested but n is set for development analysis or the key is provided as a list.
    """
    # Validate IV batch parameters.
    if iv_batch_size is not None and iv_n_batches is not None:
        raise ValueError("Only one of iv_batch_size or iv_n_batches should be specified, not both.")
    if iv_batch_size is None and iv_n_batches is None:
        iv_n_batches = 10  # default number of batches if neither is set.
    if iv_batch_size is None:
        # Compute batch size as ceiling of len(X)/iv_n_batches.
        iv_batch_size = -(-len(X) // iv_n_batches)  # the double - is a rounding trick.

    # For returning a single mean, key must be a singular value and n must not indicate development analysis.
    if return_mean:
        if (isinstance(key, (list, tuple)) and len(key) != 1) or (isinstance(n, str) and n.lower() == "development"):
            raise ValueError("When return_mean=True, key must be a single value and n must be numeric (not 'development').")

    # Create an IV instance.
    # Note: IV requires (x_data, y_data, classifier) so we pass X, y, classifier in that order.
    iv_instance = IV(X, y, classifier)
    iv_instance.run_iv(start_trainset_size=iv_start_trainset_size, batch_size=iv_batch_size)

    # Determine the label parameter for the posterior computation.
    # If key indicates overall metrics ("bacc" or "acc"), compute posterior for all labels.
    if not isinstance(key, (list, tuple)):
        chosen_key = key
    else:
        # When multiple keys are provided, we default to computing posterior for all labels.
        chosen_key = None

    if isinstance(chosen_key, str) and chosen_key.lower() in ["bacc", "acc"]:
        posterior_label = None
    else:
        posterior_label = chosen_key  # This can be an int (label) or None.

    # Compute the posterior using the given MCMC parameters.
    iv_instance.compute_posterior(num_samples=mcmc_num_samples,
                                  step_size=mcmc_step_size,
                                  burn_in=mcmc_burn_in,
                                  thin=mcmc_thin,
                                  random_seed=mcmc_random_seed,
                                  label=posterior_label)

    # Retrieve the requested accuracy distribution.
    # If key was provided as a singular value, pass it as such.
    result = iv_instance.get(keys=key, n=n, plot=plot)

    # If a mean is requested, try to return the mean value.
    if return_mean:
        if hasattr(result, "mean") and callable(result.mean):
            return result.mean()
        else:
            raise ValueError("The obtained result does not support computing a mean.")
    return result


def dist_greater_than(dist1, dist2):
    """
    Compute the probability that a sample drawn from the first frozen distribution is greater
    than a sample drawn from the second frozen distribution.

    This is achieved by integrating the product of the first distribution's PDF and the second
    distribution's CDF over the entire real line.

    Parameters:
      dist1 : scipy.stats frozen distribution
          A frozen distribution (such as one returned by stats.rv_histogram) representing the first variable.
      dist2 : scipy.stats frozen distribution
          A frozen distribution representing the second variable.

    Returns:
      float
          The probability that a value drawn from dist1 exceeds a value drawn from dist2.

    Raises:
      TypeError:
          If either dist1 or dist2 does not have the required pdf and cdf methods.
    """
    # Verify that the inputs provide the necessary methods.
    if not all(hasattr(d, "pdf") for d in (dist1,)) or not all(hasattr(d, "cdf") for d in (dist2,)):
        raise TypeError("Both dist1 and dist2 must have 'pdf' and 'cdf' methods.")

    # Define the integrand: f_dist1(x) * F_dist2(x)
    integrand = lambda x: dist1.pdf(x) * dist2.cdf(x)

    # Integrate over the entire real line.
    probability, _ = quad(integrand, -np.inf, np.inf)
    return probability

# TODO: Signifikanzniveau, soll p wert zurückgeben

# --- Example (for testing or demonstration purposes) ---
if __name__ == "__main__":
    from sklearn.neighbors import KNeighborsClassifier
    import matplotlib.pyplot as plt

    # Generate some artificial data for demonstration.
    np.random.seed(42)
    num_samples = 500
    X = np.random.rand(num_samples, 2)
    # Create a simple binary classification target.
    y = (X[:, 0] + X[:, 1] > 1).astype(int)

    # Create an instance of a k-Nearest Neighbors classifier.
    clf = KNeighborsClassifier(n_neighbors=3)

    # Example 1: Retrieve the balanced accuracy mean.
    mean_bacc = independent_validation(clf, X, y, key="bacc", n=100, return_mean=True, plot=False)
    print("Mean Balanced Accuracy:", mean_bacc)

    # Example 2: Retrieve the overall accuracy distribution and plot it.
    acc_dist = independent_validation(clf, X, y, key="acc", n=100, return_mean=False, plot=True)
    xs = np.linspace(0, 1, 200)
    plt.figure()
    plt.plot(xs, acc_dist.pdf(xs))
    plt.xlabel("Accuracy")
    plt.ylabel("Density")
    plt.title("Overall Accuracy Distribution")
    plt.show()

    # Example 3: Compare two distributions using dist_greater_than.
    # Let’s compare the overall accuracy distribution to the balanced accuracy distribution.
    bacc_dist = independent_validation(clf, X, y, key="bacc", n=100, return_mean=False, plot=False)
    probability = dist_greater_than(bacc_dist, acc_dist)
    print("Probability that Balanced Accuracy > Overall Accuracy:", probability)
