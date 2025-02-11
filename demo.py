"""
Demo Script for the IV Project

This script demonstrates the following features:
  • Using the independent_validation service (with plotting and distribution comparison).
  • Building and using an IV object on artificial (multi‐class) data.
  • Running a direct MCMC sampling via metropolis_hastings.
  • Combining distributions using weighted_sum_distribution.
  • Running experiments on real data from Kaggle (Titanic) after some preprocessing.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import modules from your project.
from services import independent_validation, dist_greater_than
from iv8 import IV
from mcmc import metropolis_hastings
from weighted_sum_distribution import weighted_sum_distribution

# Import classifiers from scikit-learn.
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression


def demo_artificial_data_services():
    print("\n=== Demo: Artificial Data with independent_validation (services.py) ===")
    np.random.seed(42)
    num_samples = 200

    # Generate 2D features uniformly and a simple binary target.
    X = np.random.rand(num_samples, 2)
    y = (X[:, 0] + X[:, 1] > 1).astype(int)

    # Use k-Nearest Neighbors classifier.
    clf = KNeighborsClassifier(n_neighbors=3)

    # 1. Run independent_validation to get the mean balanced accuracy and plot distribution.
    mean_bacc = independent_validation(clf, X, y, key="bacc", n=100, iv_start_trainset_size=3, return_mean=True, plot="demo/synthetic/bacc100_1")
    print("Mean Balanced Accuracy (artificial data):", mean_bacc)

    # 2. Get the overall accuracy distribution (and show its plot).
    acc_dist = independent_validation(clf, X, y, key="acc", n=100, return_mean=False, iv_start_trainset_size=3, plot="demo/synthetic/acc100_1")

    # 3. Compare the two distributions: probability that bacc > overall accuracy.
    bacc_dist = independent_validation(clf, X, y, key="bacc", n=100, return_mean=False, iv_start_trainset_size=3, plot=False)
    prob = dist_greater_than(bacc_dist, acc_dist)
    print("Probability that Balanced Accuracy > Overall Accuracy (artificial data):", prob)
    plt.pause(1)  # pause briefly to allow plots to render


def demo_artificial_data_iv():
    print("\n=== Demo: IV Class on Artificial Multi-class Data ===")
    np.random.seed(123)
    num_samples = 150
    # Create 2D artificial data with three classes (labels 0, 1, 2).
    X = np.random.rand(num_samples, 2)
    y = np.random.choice([0, 1, 2], size=num_samples)

    # Use a k-NN classifier.
    clf = KNeighborsClassifier(n_neighbors=3)

    # Create an IV object and run the independent validation process.
    iv = IV(X, y, clf)
    iv.run_iv(start_trainset_size=10, batch_size=5)

    # Compute the posterior distributions via MCMC.
    iv.compute_posterior(num_samples=250, step_size=0.05, burn_in=50, thin=10, random_seed=42)
    print("Posterior computed for labels:", iv.posterior.keys())

    # Plot the balanced and overall accuracy distributions for training size n=50.
    print("Plotting balanced accuracy distribution (n=50)...")
    bacc_dist = iv.get("bacc", n=50, plot="demo/synthetic/bacc50_1")
    print("Plotting overall accuracy distribution (n=50)...")
    acc_dist = iv.get("acc", n=50, plot="demo/synthetic/acc50_1")

    # Plot the “development” curves (accuracy as a function of training set size).
    print("Plotting development curves for balanced accuracy...")
    means_bacc, lower_bacc, upper_bacc = iv.get("bacc", n="development", plot="demo/synthetic/baccDevelopment_1")
    plt.pause(1)


def demo_mcmc():
    print("\n=== Demo: Metropolis-Hastings Sampling (mcmc.py) ===")

    # Define a target log-probability for a standard normal distribution.
    def target_log_prob(x):
        # Since x is a NumPy array (even if 1D), return the log-density (up to constant).
        return -0.5 * np.sum(x ** 2)

    initial_value = np.array([0.0])
    num_samples = 1000
    samples, acc_rate = metropolis_hastings(
        target_log_prob_fn=target_log_prob,
        initial_value=initial_value,
        num_samples=num_samples,
        step_size=1.0,
        burn_in=100,
        thin=1,
        random_seed=42
    )
    print("Metropolis-Hastings acceptance rate:", acc_rate)

    # Plot a histogram of the samples and compare with the analytic standard normal density.
    plt.figure()
    plt.hist(samples, bins=30, density=True, alpha=0.7, label="MCMC samples")
    x_grid = np.linspace(-4, 4, 200)
    normal_pdf = np.exp(-0.5 * x_grid ** 2) / np.sqrt(2 * np.pi)
    plt.plot(x_grid, normal_pdf, "r-", lw=2, label="Standard Normal PDF")
    plt.xlabel("x")
    plt.ylabel("Density")
    plt.title("Metropolis-Hastings Sampling of Standard Normal")
    plt.legend()
    plt.show()


def demo_weighted_sum_distribution():
    print("\n=== Demo: Weighted Sum Distribution (weighted_sum_distribution.py) ===")
    from scipy.stats import norm, uniform

    # Create two distributions:
    normal_dist = norm(loc=0, scale=1)
    # A uniform distribution on [-1, 1].
    uniform_dist = uniform(loc=-1, scale=2)

    # Combine them using weights 0.6 and 0.4.
    combined_dist = weighted_sum_distribution([normal_dist, uniform_dist], weights=[0.6, 0.4])

    # Plot the resulting density.
    plt.figure()
    x_vals = np.linspace(-4, 4, 300)
    plt.plot(x_vals, combined_dist.pdf(x_vals), label="Weighted Sum Distribution")
    plt.xlabel("x")
    plt.ylabel("Density")
    plt.title("Convolution of Normal and Uniform PDFs")
    plt.legend()
    plt.show()


def demo_real_data():
    print("\n=== Demo: Real Data from Kaggle (Titanic Dataset) ===")
    try:
        # Attempt to load the Titanic dataset. (Place the file in a folder called "data".)
        data = pd.read_csv("data/train_data.csv")
    except Exception as e:
        print(
            "Could not load 'data/train_data.csv'. Please ensure the Kaggle Titanic dataset is present. Skipping real-data demo.")
        return

    # Preprocess the Titanic data.
    # Drop rows with missing values in key columns.
    cols_of_interest = ["Pclass_1" , "Pclass_2", "Pclass_3", "Sex", "Age", "Fare", "Survived"]
    data = data.dropna(subset=cols_of_interest).copy()

    # Encode categorical variable 'Sex' to numerical values.
    # data["Sex"] = data["Sex"].map({"male": 0, "female": 1})

    # Use features: Pclass, Sex, Age, Fare and target: Survived.
    X = data[["Pclass_1" , "Pclass_2", "Pclass_3", "Sex", "Age", "Fare"]].values
    y = data["Survived"].values

    # Choose a classifier – here we use Logistic Regression.
    clf = LogisticRegression(max_iter=200)

    # Run independent_validation on the Titanic data.
    print("Running independent_validation on Titanic data...")
    mean_bacc = independent_validation(clf, X, y, key="bacc", n=50, iv_batch_size=5, return_mean=True, plot="demo/titanic/bacc50_1")
    print("Mean Balanced Accuracy (Titanic data):", mean_bacc)

    # Use the IV object to record the evolving performance.
    iv_real = IV(X, y, clf)
    iv_real.run_iv(start_trainset_size=10, batch_size=5)
    iv_real.compute_posterior(num_samples=1000, step_size=0.05, burn_in=100, thin=50, random_seed=42)
    print("Posterior computed for Titanic data, labels:", iv_real.posterior.keys())

    # Plot balanced and overall accuracy distributions.
    print("Plotting balanced accuracy (Titanic data, n=50)...")
    iv_real.get("bacc", n=50, plot="demo/titanic/bacc50_2")
    print("Plotting overall accuracy (Titanic data, n=50)...")
    iv_real.get("acc", n=50, plot="demo/titanic/acc50")

    # Plot the evolution of overall accuracy as training set size increases.
    print("Plotting development curve for overall accuracy on Titanic data...")
    iv_real.get("acc", n="development", plot="demo/titanic/accDevelopment")
    plt.pause(1)


def main():
    # demo_artificial_data_services()
    # demo_artificial_data_iv()
    # demo_mcmc()
    # demo_weighted_sum_distribution()
    demo_real_data()


# TODO: Why the hell is acc between the two different baccs for titanic data?
# TODO: Make synthetic case with unbalanced dataset

# TODO: How can acc be so bad for 50 and so good for 100. Formular doesn't make sense

if __name__ == "__main__":
    main()
