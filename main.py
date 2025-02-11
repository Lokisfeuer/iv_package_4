import collections
from sklearn.utils.multiclass import unique_labels
import numpy as np
import pandas as pd
from scipy import stats
from scipy.integrate import quad
from scipy.signal import convolve


def dist_greater_than(dist1, dist2):
    """
    Compute the probability that a sample drawn from the first frozen distribution
    (dist1) is greater than a sample drawn from the second frozen distribution (dist2).

    Parameters:
        dist1 : A frozen distribution (e.g., from scipy.stats.rv_histogram)
        dist2 : A frozen distribution (e.g., from scipy.stats.rv_histogram)

    Returns:
        float: The probability that a sample from dist1 is greater than a sample from dist2.
    """
    # Define the integrand: f_dist1(x) * F_dist2(x)
    integrand = lambda x: dist1.pdf(x) * dist2.cdf(x)

    # Integrate over the whole real line.
    probability, _ = quad(integrand, -np.inf, np.inf)
    return probability

class MultiIV:
    def __init__(self, classifiers, datasets):
        # Parameters are dicts with {name: clf} or {name: dataset}
        """
        Initialize a MultiIV object with classifiers and datasets.
        :param classifiers: dictionary of names: classifiers
        :param datasets: dictionary of names: datasets
        """
        self.classifiers = classifiers
        self.datasets = datasets
        self.ivs = {}
        for clf_name in classifiers.keys():
            for dataset_name in datasets.keys():
                self.ivs[(clf_name, dataset_name)] = None

    def get(self, clf_name, dataset_name):
        """
        Get the IV for the given classifier and dataset.
        :param clf_name: name of the classifier
        :param dataset_name: name of the dataset
        :return: IV for the given classifier and dataset
        """
        if self.ivs[(clf_name, dataset_name)] is None:
            clf = self.classifiers[clf_name]
            dataset = self.datasets[dataset_name]
            self.ivs[(clf_name, dataset_name)] = IV(clf, dataset)
        return self.ivs[(clf_name, dataset_name)]

    def __getitem__(self, item):
        clf_name, dataset_name = item
        return self.get(clf_name, dataset_name)

def independent_validation():
    """
    Run IV and return acc / bacc mean and std
    :return:
    """
    pass

TupleHit = collections.namedtuple('TupleHit', ('n', 'hit_bool'))
ABSamples = collections.namedtuple('ABSamples', ('a', 'b'))

class IV:
    def __init__(self, clf, x, y):
        """
        Initialize IV object with a classifier and a dataset.
        :param clf: classifier
        :param dataset: dataset
        """
        self.clf = clf
        self.x = x
        self.y = y
        self.tuple_hits = set() # find better nam
        self.labels = unique_labels(y)
        self.samples = {i: pd.DataFrame() for i in self.labels}

    def run_iv(self):
        """
        Run IV for the classifier and dataset.
        Generate tuple_hits
        """
        if self.tuple_hits != set():
            raise ValueError
        # this is a placeholder
        for n, (x, y) in enumerate(zip(self.x, self.y)):
            hit_bool = self.clf.predict(x) == y
            self.tuple_hits.add(TupleHit(n=n, hit_bool=hit_bool))

    def mcmc(self):
        """
        Run MCMC to get posterior from tuple_hits
        """
        # this is a placeholder
        for label in self.labels:
            asymptotes = []
            offset_factors = []
            for i in range(10):
                asymptotes.append(0.8)
                offset_factors.append(0.5)
            self.samples[label] = pd.DataFrame({'asymptote': asymptotes, 'offset_factor': offset_factors})

    def get(self, label):
        if label == 'bacc':  # 'bacc' should not be in self.labels
            return self.get_bacc()
        elif label == 'acc':  # neither should acc be in there.
            return self.get_acc()
        if label not in self.labels:
            raise ValueError
        hist = np.histogram(self.samples[label].asymptote, bins="auto", density=True)
        return stats.rv_histogram(hist, density=True)

    def get_bacc(self, grid_points=500):
        """
        Compute the pdf of Y = (A + B + C + ...)/n using convolutions,
        and return also a frozen distribution representing the average.

        Each individual pdf is evaluated on a common grid in [0,1]. After convolving
        the individual densities (which yields the pdf of the sum X = A+B+...),
        the average pdf is obtained using the change of variables:

            Y = X/n  =>  f_Y(y) = n * f_X(ny)

        We then construct a frozen distribution using rv_histogram with bins that
        match the computed grid.

        Parameters:
            grid_points (int): Number of points to define the grid over [0, 1].

        Returns:
            frozen_avg (rv_histogram): A frozen distribution approximating the average pdf.
        """
        # Set up a common grid on [0, 1] for evaluating each pdf
        grid = np.linspace(0, 1, grid_points)
        dx = grid[1] - grid[0]
        n = len(self.labels)

        # Evaluate the pdf for each label using the get() method
        pdfs = []
        for label in self.labels:
            distribution = self.get(label)
            pdf_vals = distribution.pdf(grid)
            # Normalize to ensure area = 1
            pdf_vals /= np.trapz(pdf_vals, grid)
            pdfs.append(pdf_vals)

        # Convolve the pdfs to obtain the pdf of the sum X = A+B+...
        pdf_sum = pdfs[0]
        for pdf in pdfs[1:]:
            pdf_sum = np.convolve(pdf_sum, pdf) * dx

        # The domain of the sum X is [0, n]. Create the corresponding grid.
        grid_sum = np.linspace(0, n, len(pdf_sum))

        # Transform from the sum X to the average Y = X/n.
        # Change-of-variables gives: f_Y(y) = n * f_X(ny)
        new_grid = grid  # Y is defined on [0,1]
        pdf_avg = n * np.interp(n * new_grid, grid_sum, pdf_sum)
        pdf_avg /= np.trapz(pdf_avg, new_grid)  # Ensure normalization

        # To create a frozen distribution, define bin edges matching new_grid.
        # Here we approximate new_grid as bin centers. We'll use evenly spaced bins.
        bin_edges = np.linspace(0, 1, grid_points + 1)
        # Approximate the "counts" for each bin. Since pdf_avg approximates density,
        # counts ≈ density * bin width.
        counts = pdf_avg * dx
        histogram_data = (counts, bin_edges)
        frozen_avg = stats.rv_histogram(histogram_data, density=True)

        return frozen_avg


