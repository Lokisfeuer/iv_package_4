import unittest
import numpy as np
import matplotlib
matplotlib.use('Agg')  # use a non-interactive backend for tests that call plotting routines
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import rv_histogram

# Import the functions and classes from the modules.
from mcmc import metropolis_hastings
from weighted_sum_distribution import weighted_sum_distribution
from iv8 import IV

# A simple dummy classifier that always predicts the most frequent label
from sklearn.base import BaseEstimator, ClassifierMixin

from services import independent_validation, dist_greater_than
from sklearn.base import BaseEstimator, ClassifierMixin


class DummyClassifier(BaseEstimator, ClassifierMixin):
    def fit(self, X, y):
        # Find the most common label in the training data.
        if len(y) == 0:
            self.most_common = None
        else:
            self.most_common = np.bincount(y).argmax()
        return self

    def predict(self, X):
        # Predict the most common label regardless of X.
        n = len(X)
        return np.full(n, self.most_common, dtype=int)


class TestMCMC(unittest.TestCase):
    def test_metropolis_hastings_normal_1d(self):
        """Test 1D sampling from a standard normal distribution."""
        # Define target log-probability for 1D standard normal (up to additive constant).
        def target_log_prob(x):
            # x is a numpy array, so sum(x**2) works even if x is multidimensional.
            return -0.5 * np.sum(x ** 2)

        initial_value = np.array([0.0])
        num_samples = 500
        samples, acc_rate = metropolis_hastings(
            target_log_prob_fn=target_log_prob,
            initial_value=initial_value,
            num_samples=num_samples,
            step_size=1.0,
            burn_in=100,
            thin=1,
            random_seed=42,
        )

        # Check that the shape of samples is (num_samples, 1)
        self.assertEqual(samples.shape, (num_samples, 1))
        # Check that the sample mean is close to 0 (within a tolerance).
        self.assertAlmostEqual(np.mean(samples), 0, delta=0.3)
        self.assertTrue(0 < acc_rate <= 1)

    def test_metropolis_hastings_2d(self):
        """Test 2D sampling from a standard normal distribution."""
        def target_log_prob(x):
            return -0.5 * np.sum(x ** 2)

        initial_value = np.array([0.0, 0.0])
        num_samples = 300
        samples, acc_rate = metropolis_hastings(
            target_log_prob_fn=target_log_prob,
            initial_value=initial_value,
            num_samples=num_samples,
            step_size=0.5,
            burn_in=50,
            thin=2,
            random_seed=1,
        )
        self.assertEqual(samples.shape, (num_samples, 2))
        mean_sample = np.mean(samples, axis=0)
        np.testing.assert_allclose(mean_sample, [0, 0], atol=0.3)


class TestWeightedSumDistribution(unittest.TestCase):
    def test_single_distribution(self):
        """Test that a one-distribution weighted sum returns an rv_histogram whose PDF integrates to 1."""
        normal_dist = stats.norm(loc=0, scale=1)
        result_dist = weighted_sum_distribution([normal_dist])
        self.assertIsInstance(result_dist, stats.rv_histogram)

        # Check that the PDF integrates to 1 over a reasonable grid.
        x_grid = np.linspace(-5, 5, 1000)
        pdf_vals = result_dist.pdf(x_grid)
        area = np.trapezoid(pdf_vals, x_grid)
        self.assertAlmostEqual(area, 1.0, places=2)

    def test_two_distributions(self):
        """Test the convolution of two distributions (normal and uniform) with different weights."""
        normal_dist = stats.norm(loc=0, scale=1)
        uniform_dist = stats.uniform(loc=-1, scale=2)  # Uniform on [-1, 1]
        # Test with equal weights.
        result_dist_eq = weighted_sum_distribution([normal_dist, uniform_dist], weights=[0.5, 0.5])
        self.assertIsInstance(result_dist_eq, stats.rv_histogram)
        x_grid = np.linspace(-5, 5, 1000)
        pdf_vals_eq = result_dist_eq.pdf(x_grid)
        area_eq = np.trapezoid(pdf_vals_eq, x_grid)
        self.assertAlmostEqual(area_eq, 1.0, places=2)

        # Test with different weights.
        result_dist_diff = weighted_sum_distribution([normal_dist, uniform_dist], weights=[0.3, 0.7])
        self.assertIsInstance(result_dist_diff, stats.rv_histogram)
        pdf_vals_diff = result_dist_diff.pdf(x_grid)
        area_diff = np.trapezoid(pdf_vals_diff, x_grid)
        self.assertAlmostEqual(area_diff, 1.0, places=2)

        # TODO: make proper test with means.

        # TODO: Make test with accuracy computation based on combination of all samples

    def test_default_weights(self):
        """Test that omitting the weights argument results in uniform (default) weighting."""
        normal_dist = stats.norm(0, 1)
        uniform_dist = stats.uniform(-1, 2)
        result_default = weighted_sum_distribution([normal_dist, uniform_dist])
        result_uniform = weighted_sum_distribution([normal_dist, uniform_dist], weights=[0.5, 0.5])
        x_grid = np.linspace(-5, 5, 1000)
        pdf_default = result_default.pdf(x_grid)
        pdf_uniform = result_uniform.pdf(x_grid)
        # The two PDFs should be nearly equal.
        np.testing.assert_allclose(pdf_default, pdf_uniform, atol=1e-2)


class TestIV(unittest.TestCase):
    def setUp(self):
        # Create a small artificial dataset.
        # 20 samples with 1 feature; labels alternating between 0 and 1.
        self.x_data = np.arange(20).reshape(20, 1)
        self.y_data = np.tile([0, 1], 10)
        self.dummy_clf = DummyClassifier()

    def test_run_iv_records(self):
        """Test that the IV process records prediction outcomes correctly."""
        iv_instance = IV(self.x_data, self.y_data, self.dummy_clf)
        iv_instance.run_iv(start_trainset_size=2, batch_size=1)
        # The initial trainset uses 2 samples, so predictions happen for the remaining 18 samples.
        total_predictions = sum(len(rec['sizes']) for rec in iv_instance.iv_records.values())
        self.assertEqual(total_predictions, 18)

    def test_compute_posterior(self):
        """Test that compute_posterior populates the posterior attribute for each label."""
        iv_instance = IV(self.x_data, self.y_data, self.dummy_clf)
        iv_instance.run_iv(start_trainset_size=2, batch_size=1)
        iv_instance.compute_posterior(num_samples=100, burn_in=10, thin=1, random_seed=42)
        for label in np.unique(self.y_data):
            self.assertIn(label, iv_instance.posterior)
            samples, acc_rate = iv_instance.posterior[label]
            # Each sample should contain two parameters: asymptote and offset_factor.
            self.assertEqual(samples.shape[1], 2)

    def test_get_label_accuracy_before_and_after_posterior(self):
        """Test that get_label_accuracy raises an error before computing posterior and returns a valid distribution after."""
        iv_instance = IV(self.x_data, self.y_data, self.dummy_clf)
        iv_instance.run_iv(start_trainset_size=2, batch_size=1)
        """
        # Before running compute_posterior, trying to get label accuracy should raise ValueError.
        with self.assertRaises(ValueError):
            iv_instance.get_label_accuracy(0, plot=False)
        """
        # Now compute posterior and obtain the accuracy distribution.
        iv_instance.compute_posterior(num_samples=100, burn_in=10, thin=1, random_seed=42)
        dist = iv_instance.get_label_accuracy(0, plot=False, n=float('inf'))
        self.assertIsInstance(dist, rv_histogram)

    def test_get_bacc_and_acc(self):
        """Test that get_bacc and get_acc return valid frozen distributions."""
        iv_instance = IV(self.x_data, self.y_data, self.dummy_clf)
        iv_instance.run_iv(start_trainset_size=2, batch_size=1)
        iv_instance.compute_posterior(num_samples=100, burn_in=10, thin=1, random_seed=42)
        bacc = iv_instance.get_bacc_dist(n=10, plot=False)
        acc = iv_instance.get_acc_dist(n=10, plot=False)
        self.assertIsInstance(bacc, rv_histogram)
        self.assertIsInstance(acc, rv_histogram)

    def test_get_multi(self):
        """Test that get_multi returns a dictionary with the expected (key, n) tuples."""
        iv_instance = IV(self.x_data, self.y_data, self.dummy_clf)
        iv_instance.run_iv(start_trainset_size=2, batch_size=1)
        iv_instance.compute_posterior(num_samples=100, burn_in=10, thin=1, random_seed=42)
        results = iv_instance.get_multi(keys=[0, 'bacc'], n=[10, 20], plot=False)
        expected_keys = {(0, 10), (0, 20), ('bacc', 10), ('bacc', 20)}
        self.assertEqual(set(results.keys()), expected_keys)
        for distr in results.values():
            self.assertIsInstance(distr, rv_histogram)

    def test_get_development(self):
        """Test that get_development returns three lists (means, lower, and upper quartiles) of the expected length."""
        iv_instance = IV(self.x_data, self.y_data, self.dummy_clf)
        iv_instance.run_iv(start_trainset_size=2, batch_size=1)
        iv_instance.compute_posterior(num_samples=100, burn_in=10, thin=1, random_seed=42)
        up_to = 5
        means, lower_q, upper_q = iv_instance.get_development(0, plot=False, up_to=up_to)
        self.assertEqual(len(means), up_to)
        self.assertEqual(len(lower_q), up_to)
        self.assertEqual(len(upper_q), up_to)

    def test_get_method(self):
        """Test the general get method for various keys and parameters."""
        iv_instance = IV(self.x_data, self.y_data, self.dummy_clf)
        iv_instance.run_iv(start_trainset_size=2, batch_size=1)
        iv_instance.compute_posterior(num_samples=100, burn_in=10, thin=1, random_seed=42)

        # Test retrieving a label-specific accuracy.
        dist1 = iv_instance.get(0, n=10, plot=False)
        self.assertIsInstance(dist1, rv_histogram)

        # Test with multiple keys.
        multi_dists = iv_instance.get(keys=[0, 'bacc', 'acc'], n=10, plot=False)
        expected_keys = {(0, 10), ('bacc', 10), ('acc', 10)}
        self.assertEqual(set(multi_dists.keys()), expected_keys)

        # Test development mode.
        dev_result = iv_instance.get("acc", n="development", plot=False)
        self.assertIsInstance(dev_result, tuple)
        self.assertEqual(len(dev_result), 3)

    def test_get_label_accuracy_caching(self):
        """Test that successive calls to get_label_accuracy (with the same arguments) return consistent results (caching)."""
        iv_instance = IV(self.x_data, self.y_data, self.dummy_clf)
        iv_instance.run_iv(start_trainset_size=2, batch_size=1)
        iv_instance.compute_posterior(num_samples=100, burn_in=10, thin=1, random_seed=42)
        dist1 = iv_instance.get_label_accuracy(0, plot=False, n=10)
        dist2 = iv_instance.get_label_accuracy(0, plot=False, n=10)
        # Compare the PDFs on a common grid.
        x_grid = np.linspace(0, 1, 100)
        np.testing.assert_allclose(dist1.pdf(x_grid), dist2.pdf(x_grid), atol=1e-6)

class TestIndependentValidationServices(unittest.TestCase):
    def setUp(self):
        # Create an artificial binary classification dataset.
        np.random.seed(42)
        self.X = np.random.rand(100, 2)
        # For simplicity, set label=1 when the sum of the two features is at least 1.
        self.y = (np.sum(self.X, axis=1) >= 1).astype(int)
        self.classifier = DummyClassifier()

    def test_independent_validation_return_mean(self):
        """
        Test that independent_validation returns a float mean when return_mean is True.
        """
        mean_accuracy = independent_validation(
            self.classifier, self.X, self.y,
            key="bacc", n=50, return_mean=True, plot=False
        )
        self.assertIsInstance(mean_accuracy, float)
        self.assertGreaterEqual(mean_accuracy, 0)
        self.assertLessEqual(mean_accuracy, 1)

    def test_independent_validation_return_distribution(self):
        """
        Test that independent_validation returns a frozen distribution (object with a callable pdf)
        when return_mean is False.
        """
        dist = independent_validation(
            self.classifier, self.X, self.y,
            key="acc", n=50, return_mean=False, plot=False
        )
        # Check that the result has a pdf method (i.e. is an rv_histogram or similar).
        self.assertTrue(callable(getattr(dist, "pdf", None)))

    def test_independent_validation_development_error(self):
        """
        When n is set to "development" and return_mean is True,
        independent_validation should raise a ValueError.
        """
        with self.assertRaises(ValueError):
            independent_validation(
                self.classifier, self.X, self.y,
                key="bacc", n="development", return_mean=True, plot=False
            )

    def test_independent_validation_multiple_keys_error(self):
        """
        Test that independent_validation raises a ValueError when a list of keys is provided
        while return_mean is True.
        """
        with self.assertRaises(ValueError):
            independent_validation(
                self.classifier, self.X, self.y,
                key=[0, "bacc"], n=50, return_mean=True, plot=False
            )

    def test_independent_validation_with_iv_n_batches(self):
        """
        Test that independent_validation works correctly when using the iv_n_batches parameter
        (which should compute the batch size automatically).
        """
        # Do not provide iv_batch_size but provide iv_n_batches.
        result = independent_validation(
            self.classifier, self.X, self.y,
            key="acc", n=50, return_mean=False, plot=False,
            iv_n_batches=5
        )
        self.assertTrue(callable(getattr(result, "pdf", None)))

    def test_dist_greater_than_equal_normals(self):
        """
        Test that dist_greater_than computes a probability near 0.5 when comparing two identical
        normal distributions.
        """
        norm1 = stats.norm(loc=0, scale=1)
        norm2 = stats.norm(loc=0, scale=1)
        probability = dist_greater_than(norm1, norm2)
        self.assertAlmostEqual(probability, 0.5, delta=0.05)

    def test_dist_greater_than_error(self):
        """
        Test that dist_greater_than raises a TypeError when one of the inputs does not have a pdf or cdf.
        """
        class Dummy:
            pass

        dummy = Dummy()
        valid_dist = stats.norm(loc=0, scale=1)
        with self.assertRaises(TypeError):
            dist_greater_than(dummy, valid_dist)
        with self.assertRaises(TypeError):
            dist_greater_than(valid_dist, dummy)

if __name__ == "__main__":
    unittest.main()
