import numpy as np


def metropolis_hastings(
        target_log_prob_fn,
        initial_value,
        num_samples,
        step_size=0.1,
        proposal_fn=None,
        burn_in=0,
        thin=1,
        random_seed=None
):
    """
    Performs MCMC sampling using the Metropolis-Hastings algorithm.

    Parameters:
    -----------
    target_log_prob_fn : function
        A function that takes a state (or parameter vector) and returns the log probability of the state.
    initial_value : array_like
        The starting state for the MCMC chain. Can be a scalar or numpy array.
    num_samples : int
        The number of samples to return (after burn-in and thinning).
    step_size : float or array_like, optional
        Standard deviation for the Gaussian proposal. If initial_value is multi-dimensional,
        step_size can be a scalar (used for all dimensions) or an array of standard deviations (one per dimension).
        Default is 1.0.
    proposal_fn : function, optional
        An alternative proposal function. It should take the current state and step_size (if needed)
        and return a new proposed state. If None, a symmetric Gaussian proposal is used.
    burn_in : int, optional
        The number of initial samples to discard (burn-in period). Default is 0.
    thin : int, optional
        Interval for thinning the chain. For example, thin=2 will keep every 2nd sample.
        Default is 1.
    random_seed : int, optional
        Seed for reproducibility. Default is None.

    Returns:
    --------
    samples : np.ndarray
        An array of shape (num_samples, ...) containing the sampled states.
    acceptance_rate : float
        The proportion of proposals that were accepted (excluding burn-in).
    """

    # Set random seed for reproducibility if provided.
    if random_seed is not None:
        np.random.seed(random_seed)

    # Initialize the MCMC chain
    current_state = np.array(initial_value)
    current_log_prob = target_log_prob_fn(current_state)

    samples = []
    accepted = 0
    total_proposals = 0

    # Total iterations: extra iterations might be needed to account for thinning, etc.
    total_iterations = burn_in + num_samples * thin

    for iteration in range(total_iterations):
        # Generate a proposal. Use the custom proposal_fn if provided.
        if proposal_fn is None:
            # Making sure step_size has the right shape. If scalar given, expand it.
            proposal = current_state + np.random.normal(
                loc=0,
                scale=step_size if np.ndim(current_state) > 0 else float(step_size),
                size=current_state.shape
            )
        else:
            proposal = proposal_fn(current_state, step_size)

        proposal_log_prob = target_log_prob_fn(proposal)

        # Acceptance probability (using the log probabilities for numerical stability)
        # TODO: check balancing with web.
        log_accept_ratio = proposal_log_prob - current_log_prob
        if np.log(np.random.rand()) < log_accept_ratio:
            # Accept the proposal
            current_state = proposal
            current_log_prob = proposal_log_prob
            accepted += 1  # count accepted proposals
        total_proposals += 1

        # Record sample if beyond burn-in and respects thinning interval.
        if iteration >= burn_in and ((iteration - burn_in) % thin == 0):
            samples.append(current_state.copy())

    samples = np.array(samples)
    acceptance_rate = accepted / total_proposals

    return samples, acceptance_rate

# TODO: Does this still need balancing out because of the normal distributed step (instead of a uniform random new guess)
