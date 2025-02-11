"""
This module implements an Independent Validation (IV) process for classification.
The IV process is designed to assess the evolving accuracy of a classifier as new data
is incrementally incorporated into the training set. [Additional documentation omitted for brevity...]
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import rv_histogram
from mcmc import metropolis_hastings
from weighted_sum_distribution import weighted_sum_distribution
from sklearn.base import clone
import warnings
import math


class IV:
    # TODO: Prior für den offset factor zwischen 0 und 10 legen, makes sense da ab 10 sicher die bedingung aus dem eckert paper erfüllt ist.
        # - Eckert Paper genau nachlesen.
        # - im paper korrigieren.
    def __init__(self, x_data, y_data, classifier):

        # Shuffle the dataset (x_data and y_data remain aligned)
        indices = np.random.permutation(len(x_data))

        # Save data and classifier (a clone is stored to avoid external modifications)
        self.x_data = x_data[indices]
        self.y_data = y_data[indices]
        self.classifier = clone(classifier)

        # The unique labels in the data.
        self.labels = np.unique(y_data)

        # For each label, record lists of:
        #   'sizes': the training set size at prediction time
        #   'outcomes': 1 if prediction correct, 0 otherwise.
        self.iv_records = {label: {'sizes': [], 'outcomes': []} for label in self.labels}

        # To hold posterior MCMC samples for each label: {label: (samples, acceptance_rate)}
        self.posterior = {}

        # Caches for distributions
        self.accuracy_cache = {}  # key: (label, n)
        self.bacc_cache = {}      # key: n
        self.acc_cache = {}       # key: n
        self.multi_cache = {}     # key: (key, n)
        self.development_cache = {}  # key: key -> dict

        # Frequency for each label in the full dataset for weighted overall accuracy.
        total = len(y_data)
        self.label_frequencies = {}
        for label in self.labels:
            self.label_frequencies[label] = np.sum(y_data == label) / total

    def run_iv(self, start_trainset_size=2, batch_size=1):
        """
        Runs the Independent Validation (IV) process.

        The process:
          1. The classifier is initially trained on the first 'start_trainset_size' samples.
          2. For the remaining samples, in batches of 'batch_size':
             - The current classifier predicts the label(s) for the new sample(s).
             - For each sample, the outcome (1 for a correct prediction; 0 otherwise)
               is recorded along with the current training set size.
             - The new sample(s) are then added to the training set and the classifier is retrained.

        The outcomes are stored in self.iv_records, organized by true label.
        """
        n_total = len(self.x_data)
        if start_trainset_size >= n_total:
            raise ValueError("start_trainset_size must be less than the total number of samples.")

        # Initial training set: use the first start_trainset_size samples.
        train_indices = list(range(start_trainset_size))

        # Process remaining samples in order, batch-by-batch.
        current_index = start_trainset_size

        # Extend until all classes are inside trainingsset
        while len(np.unique(self.y_data[train_indices])) < len(self.labels) and current_index < n_total:
            batch_indices = list(range(current_index, min(current_index + batch_size, n_total)))
            x_batch = self.x_data[batch_indices]
            y_batch = self.y_data[batch_indices]
            current_train_size = len(train_indices)
            # guessing prediction
            predictions = np.random.choice(self.labels, size=len(batch_indices)) 
            for i, true_label in enumerate(y_batch):
                outcome = 1 if predictions[i] == true_label else 0
                self.iv_records[true_label]['sizes'].append(current_train_size)
                self.iv_records[true_label]['outcomes'].append(outcome)
            train_indices.extend(batch_indices)
            current_index += batch_size

        # Initial Fit
        x_train = self.x_data[train_indices]  
        y_train = self.y_data[train_indices] 
        self.classifier.fit(x_train, y_train) 

        while current_index < n_total:
            batch_indices = list(range(current_index, min(current_index + batch_size, n_total)))
            x_batch = self.x_data[batch_indices]
            y_batch = self.y_data[batch_indices]

            # Current training set size (used for computing accuracy function)
            current_train_size = len(train_indices)

            # Predict for the batch.
            predictions = self.classifier.predict(x_batch)

            # For each sample, record the prediction outcome in the IV records.
            for i, true_label in enumerate(y_batch):
                outcome = 1 if predictions[i] == true_label else 0
                self.iv_records[true_label]['sizes'].append(current_train_size)
                self.iv_records[true_label]['outcomes'].append(outcome)

            # Add the batch samples to the training set and retrain.
            train_indices.extend(batch_indices)
            x_train = self.x_data[train_indices]
            y_train = self.y_data[train_indices]
            self.classifier.fit(x_train, y_train)

            current_index += batch_size

    def compute_posterior(self, num_samples=1000, step_size=0.05, burn_in=100, thin=1, random_seed=None, label=None):
        """
        Computes the posterior distribution for the parameters (asymptote, offset_factor)
        based on the IV records via Markov Chain Monte Carlo (MCMC).

        For each label, given the recorded training set sizes and outcomes, the likelihood is:
            For a sample with training size s:
                p_correct = asymptote - offset_factor / s
            and its contribution:
                outcome * log(p_correct) + (1 - outcome) * log(1 - p_correct)
        Parameter constraints:
            - asymptote must lie in (0, 1).
            - offset_factor must be non-negative.
            - Also, p_correct must lie in (0, 1) for each sample, else the parameter set is rejected.

        The MCMC sampler (metropolis_hastings) from mcmc.py is used with the provided MCMC parameters.

        Parameters:
            num_samples : number of MCMC samples to return (after burn-in and thinning).
            step_size   : step size for the proposal distribution.
            burn_in     : number of initial samples to discard.
            thin        : interval for thinning the chain.
            random_seed : seed for reproducibility.
            label       : if specified, compute the posterior only for this label.
                          Otherwise, compute for all labels.
        """
        # NEW: Check if IV records have been computed. If not, warn and automatically call run_iv.
        total_records = sum(len(record['sizes']) for record in self.iv_records.values())
        if total_records == 0:
            batch_size = math.ceil(len(self.x_data) / 10)
            start_trainset_size = 1
            warnings.warn(
                f"compute_posterior was called before run_iv; no IV records exist. "
                f"Automatically running run_iv(start_trainset_size={start_trainset_size}, batch_size={batch_size})."
            )
            self.run_iv(start_trainset_size=start_trainset_size, batch_size=batch_size)

        def target_log_prob(theta, sizes, outcomes):
            asymptote, offset_factor = theta
            if not (0 < asymptote < 1) or offset_factor < 0:
                return -np.inf
            log_prob = 0.0
            for s, outcome in zip(sizes, outcomes):
                p = asymptote - offset_factor / s
                if p <= 0 or p >= 1:
                    return -np.inf
                log_prob += outcome * np.log(p) + (1 - outcome) * np.log(1 - p)
            return log_prob

        labels_to_process = [label] if label is not None else self.labels

        for lbl in labels_to_process:
            sizes = np.array(self.iv_records[lbl]['sizes'])
            outcomes = np.array(self.iv_records[lbl]['outcomes'])
            initial_value = np.array([0.9, 1.0])

            def target(theta):
                return target_log_prob(theta, sizes, outcomes)

            samples, acceptance_rate = metropolis_hastings(
                target_log_prob_fn=target,
                initial_value=initial_value,
                num_samples=num_samples,
                step_size=step_size,
                burn_in=burn_in,
                thin=thin,
                random_seed=random_seed
            )
            self.posterior[lbl] = (samples, acceptance_rate)

    def get_label_accuracy(self, label, plot=False, n=float('inf')):
        """
        Returns the accuracy distribution for a given label at training set size n.

        The accuracy is computed from each MCMC sample as:
              accuracy(n) = asymptote - offset_factor / n
        (For n == infinity, only the asymptote is returned.)
        The set of accuracy values is used to build a histogram, which is then frozen
        as a scipy.stats distribution (rv_histogram).

        Parameters:
            label : label from the dataset.
            plot  : if True, the PDF of the distribution is plotted;
                    if a string, it is interpreted as a filename to save the plot.
            n     : training set size for which the accuracy distribution is evaluated.

        Returns:
            A frozen distribution (rv_histogram) representing the accuracy distribution.
        """
        cache_key = (label, n)
        if cache_key in self.accuracy_cache:
            dist, raw_samples = self.accuracy_cache[cache_key]
        else:
            # NEW: Instead of raising an error if the posterior for the label is missing,
            # issue a warning and automatically compute it for that label.
            if label not in self.posterior:
                warnings.warn(
                    f"Posterior for label {label} not computed. Automatically computing posterior using default parameters."
                )
                self.compute_posterior(label=label)
            samples, _ = self.posterior[label]  # Samples of shape (num_samples, 2)
            if n == float('inf'):
                accuracy_samples = samples[:, 0]
            else:
                accuracy_samples = samples[:, 0] - samples[:, 1] / n
            accuracy_samples = np.clip(accuracy_samples, 0, 1)
            num_bins = 50
            hist, bin_edges = np.histogram(accuracy_samples, bins=num_bins, density=True)
            dist = rv_histogram((hist, bin_edges))
            self.accuracy_cache[cache_key] = (dist, accuracy_samples)
            raw_samples = accuracy_samples

        if plot:
            x_grid = np.linspace(0, 1, 500)
            y_vals = dist.pdf(x_grid)
            plt.figure()
            plt.plot(x_grid, y_vals, label=f'Accuracy dist for label {label} (n={n})')
            plt.xlabel('Accuracy')
            plt.ylabel('Density')
            plt.legend()
            if isinstance(plot, str):
                plt.savefig(plot)
                plt.close()
            else:
                plt.show()
        return dist

    # TODO: get_bacc(mean/map) and get_acc(mean/map)
        # - make this same function as get_bacc_dist with a parameter

    # TODO: Get alternative to weighted_sum_distribution with assumption beta is normal distribution and then simpler formular.

    def get_bacc_dist(self, n=float('inf'), plot=False):
        """
        Returns the balanced accuracy (bacc) distribution, computed as the convolution
        of the individual label accuracy distributions (for a given training set size n) using equal weights.

        Parameters:
            n    : training set size for which the accuracy distributions are evaluated.
            plot : if True, the resulting PDF is plotted; if a string, used as filename.

        Returns:
            A frozen distribution (rv_histogram) representing the balanced accuracy.
        """
        if n in self.bacc_cache:
            bacc_dist = self.bacc_cache[n]
        else:
            distributions = []
            for label in self.labels:
                dist = self.get_label_accuracy(label, plot=False, n=n)
                distributions.append(dist)
            weights = [1 / len(self.labels)] * len(self.labels)
            bacc_dist = weighted_sum_distribution(distributions, weights=weights)
            self.bacc_cache[n] = bacc_dist

        if plot:
            x_grid = np.linspace(0, 1, 200)
            y_vals = bacc_dist.pdf(x_grid)
            plt.figure()
            plt.plot(x_grid, y_vals, label=f'Balanced Accuracy (n={n})')
            plt.xlabel('Accuracy')
            plt.ylabel('Density')
            plt.legend()
            if isinstance(plot, str):
                plt.savefig(plot)
                plt.close()
            else:
                plt.show()
        return bacc_dist

    def get_acc_dist(self, n=float('inf'), plot=False):
        """
        Returns the overall accuracy (acc) distribution which is computed as the convolution
        of the individual label accuracy distributions weighted by the frequency of the label
        in the full dataset.

        Parameters:
            n    : training set size for which the accuracy distributions are evaluated.
            plot : if True, the PDF is plotted; if a string, interpreted as filename.

        Returns:
            A frozen distribution (rv_histogram) representing the overall accuracy.
        """
        if n in self.acc_cache:
            acc_dist = self.acc_cache[n]
        else:
            distributions = []
            weights = []
            for label in self.labels:
                dist = self.get_label_accuracy(label, plot=False, n=n)
                distributions.append(dist)
                weights.append(self.label_frequencies[label])
            acc_dist = weighted_sum_distribution(distributions, weights=weights)
            self.acc_cache[n] = acc_dist

        if plot:
            x_grid = np.linspace(0, 1, 200)
            y_vals = acc_dist.pdf(x_grid)
            plt.figure()
            plt.plot(x_grid, y_vals, label=f'Overall Accuracy (n={n})')
            plt.xlabel('Accuracy')
            plt.ylabel('Density')
            plt.legend()
            if isinstance(plot, str):
                plt.savefig(plot)
                plt.close()
            else:
                plt.show()
        return acc_dist

    def get_multi(self, keys, n=float('inf'), plot=False):
        """
        Returns a dictionary of accuracy distributions based on provided keys and training set size(s).

        Parameters:
            keys : A label, 'bacc', 'acc', or a list of such keys.
            n    : A scalar or list of scalars representing the training set size.
            plot : If True, all distributions are plotted in a single graph.
                   If a string, it is interpreted as a filename.

        For each key and each n value, the appropriate method is called:
            • For key 'bacc', get_bacc(n) is used.
            • For key 'acc', get_acc(n) is used.
            • For a label key, get_label_accuracy is used.

        Returns:
            A dictionary mapping (key, n) tuples to the corresponding frozen distribution.
        """
        if not isinstance(keys, list):
            keys = [keys]
        if not isinstance(n, list):
            n = [n]

        result = {}
        if plot:
            plt.figure()

        for key in keys:
            for n_val in n:
                cache_key = (key, n_val)
                if cache_key in self.multi_cache:
                    dist = self.multi_cache[cache_key]
                else:
                    if key == 'bacc':
                        dist = self.get_bacc_dist(n=n_val, plot=False)
                    elif key == 'acc':
                        dist = self.get_acc_dist(n=n_val, plot=False)
                    elif key in self.labels:
                        dist = self.get_label_accuracy(key, plot=False, n=n_val)
                    else:
                        raise ValueError(f"Key {key} not recognized. Must be a label, 'bacc', or 'acc'.")
                    self.multi_cache[cache_key] = dist
                result[(key, n_val)] = dist

                if plot:
                    x_grid = np.linspace(0, 1, 200)
                    y_vals = dist.pdf(x_grid)
                    plt.plot(x_grid, y_vals, label=f'{key} (n={n_val})')

        if plot:
            plt.xlabel('Accuracy')
            plt.ylabel('Density')
            plt.legend()
            if isinstance(plot, str):
                plt.savefig(plot)
                plt.close()
            else:
                plt.show()
        return result

    def get_development(self, key, plot=False, up_to=100):
        """
        Computes the development of the accuracy distribution for a given key (a label, 'bacc', or 'acc')
        as the training set size (n) increases from 1 to up_to.

        For each integer n in this range, the following statistics are computed:
            • Mean of the accuracy distribution.
            • Lower quartile (25th percentile).
            • Upper quartile (75th percentile).

        If plot is True, a line plot is generated with:
            • The mean curve.
            • Lines marking the lower and upper quartiles.
            • A shaded area between the quartiles.

        Parameters:
            key    : A label or the strings 'bacc' or 'acc'.
            plot   : If True, show a plot; if a string, use it as the filename.
            up_to  : The maximum training set size to consider (starting from n=1).

        Returns:
            Three lists: means, lower_quartiles, and upper_quartiles, each of length up_to.
        """
        means = []
        lower_quartiles = []
        upper_quartiles = []
        n_values = list(range(1, up_to + 1))

        for n_val in n_values:
            if key == 'bacc':
                dist = self.get_bacc_dist(n=n_val, plot=False)
            elif key == 'acc':
                dist = self.get_acc_dist(n=n_val, plot=False)
            elif key in self.labels:
                dist = self.get_label_accuracy(key, plot=False, n=n_val)
            else:
                raise ValueError(f"Key {key} not recognized. Must be a label, 'bacc', or 'acc'.")

            samples = dist.rvs(size=10000)
            mean_val = np.mean(samples)
            # TODO: make the percentiles parameters
                # - std /
                # - credible interval
                # - percentile
            lower_q = np.percentile(samples, 25)
            upper_q = np.percentile(samples, 75)

            # Return
            means.append(mean_val)
            lower_quartiles.append(lower_q)
            upper_quartiles.append(upper_q)

        if plot:
            plt.figure()
            plt.plot(n_values, means, label='Mean Accuracy')
            plt.fill_between(n_values, lower_quartiles, upper_quartiles, color='gray', alpha=0.3,
                             label='25-75 Percentile')
            plt.xlabel('Training Set Size (n)')
            plt.ylabel('Accuracy')
            plt.legend()
            if isinstance(plot, str):
                plt.savefig(plot)
                plt.close()
            else:
                plt.show()
        return means, lower_quartiles, upper_quartiles

    def get(self, keys="bacc", n=float('inf'), plot=False):
        """
        General get function to obtain accuracy distributions.

        Parameters:
            keys : A label, 'bacc', 'acc', or a list of such keys (default: 'bacc').
            n    : A scalar, a list of scalars, or the string 'development' (default: float('inf')).
            plot : If True, the result is plotted; if a string, it is interpreted as a filename.

        Behavior:
            • If n equals "development", get_development is called for each key.
            • If a single key and a scalar n are provided:
                - For 'bacc', get_bacc is used.
                - For 'acc', get_acc is used.
                - For a label, get_label_accuracy is used.
            • Otherwise, get_multi is invoked.

        Returns:
            The output from the corresponding method.
        """
        if n == "development":
            if not isinstance(keys, list):
                return self.get_development(keys, plot=plot)
            result = {}
            for key in (keys if isinstance(keys, list) else [keys]):
                result[key] = self.get_development(key, plot=plot)
                if isinstance(plot, str):
                    # TODO: If plot is a string this overwrites itself. Improve the below warning.
                    warnings.warn(
                        f'With multiple given keys and n=="development" multiple plots will be generated.'
                        f"Plots will overwrite each other, only the last one will be permanently saved."
                        f"If you want to save individual plots, save them from the returned result."
                    )
            return result
        if not isinstance(keys, list) and not isinstance(n, list):
            if keys == 'bacc':
                return self.get_bacc_dist(n=n, plot=plot)
            elif keys == 'acc':
                return self.get_acc_dist(n=n, plot=plot)
            elif keys in self.labels:
                return self.get_label_accuracy(keys, plot=plot, n=n)
            else:
                raise ValueError(f"Key {keys} not recognized. Must be a label, 'bacc', or 'acc'.")
        return self.get_multi(keys, n=n, plot=plot)

    # TODO: Try with bigger mcmc, individual labels acc dist should approach beta.

if __name__ == "__main__":
    import numpy as np
    from sklearn.neighbors import KNeighborsClassifier
    import matplotlib.pyplot as plt

    print('Run Program')
    random_seed = 42

    # Generate artificial data.
    num_samples = 1500
    proportions = [0.1, 0.1, 0.8]
    counts = [int(num_samples * p) for p in proportions]

    # Create 2-dimensional features drawn from different normal distributions per label.
    np.random.seed(random_seed)
    x_data = []
    y_data = []

    # Label 0
    x_data.append(np.random.normal(loc=3, scale=1, size=(counts[1], 1)))
    y_data.extend([0] * counts[0])
    # Label 1
    x_data.append(np.random.normal(loc=3, scale=1, size=(counts[0], 1)))
    y_data.extend([1] * counts[1])
    # Label 2
    x_data.append(np.random.normal(loc=6, scale=0.2, size=(counts[2], 1)))
    y_data.extend([2] * counts[2])

    x_data = np.vstack(x_data)
    y_data = np.array(y_data)

    # Shuffle the dataset so that the order of x_data and y_data is randomized in unison.
    indices = np.random.permutation(x_data.shape[0])
    x_data = x_data[indices]
    y_data = y_data[indices]

    # Create a K-Nearest Neighbors classifier.
    clf = KNeighborsClassifier()
    print('Creating IV object')
    iv_instance = IV(x_data, y_data, clf)

    print('Running IV')
    # Run the IV process (this could be skipped by the user, in which case compute_posterior will call it automatically)
    iv_instance.run_iv(start_trainset_size=10, batch_size=5)

    print('Computing Posterior')
    iv_instance.compute_posterior(num_samples=250, step_size=0.05, burn_in=500, thin=10, random_seed=random_seed)

    print("Calling get methods")
    bacc_dist = iv_instance.get("bacc", plot="plots/bacc")
    acc_dist = iv_instance.get("acc", plot="plots/acc")
    label0_dist = iv_instance.get(0, plot="plots/label0")
    label1_dist = iv_instance.get(1, plot="plots/label1")
    label2_dist = iv_instance.get(2, plot="plots/label2")
    bacc_dist50 = iv_instance.get("bacc", n=50, plot="plots/bacc_50")
    acc_dist50 = iv_instance.get("acc", n=50, plot="plots/acc_50")
    means, lower_quartiles, upper_quartiles = iv_instance.get("bacc", n="development", plot="plots/development_bacc")
    means, lower_quartiles, upper_quartiles = iv_instance.get("acc", n="development", plot="plots/development_acc")

    # plt.show()

