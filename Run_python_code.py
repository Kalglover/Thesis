# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Define functions for cost calculation, equilibrium finding, etc.

# Your Python code here...
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


# Define the cost functions
def leader_cost(p, q, a, b):
    """ Cost function for the leader, where p is leader's power level, q is follower's power level. """
    return a * p ** 2 - b * np.log(1 + p / q)


def follower_cost(q, p, c, d):
    """ Cost function for the follower, where q is follower's power level, p is leader's power level. """
    return c * q ** 2 - d * np.log(1 + q / p)


# Find the best response for the follower
def follower_response(p, c, d, initial_guess=0.1):
    result = minimize(lambda q: follower_cost(q, p, c, d), initial_guess, bounds=[(0.01, None)])
    return result.x


# Stackelberg Equilibrium finder
def find_stackelberg_equilibrium(a, b, c, d):
    def objective(p):
        q = follower_response(p, c, d)
        return leader_cost(p, q, a, b)

    initial_guess = 0.1
    bounds = [(0.01, None)]
    result = minimize(objective, initial_guess, bounds=bounds)
    optimal_p = result.x
    optimal_q = follower_response(optimal_p, c, d)
    return optimal_p, optimal_q


# Parameters (You may adjust these based on empirical data or detailed system modeling)
a, b, c, d = 1, 2, 1, 2

# Robustness Parameters
noise_std_dev = 0.1  # Standard deviation of Gaussian noise added to costs


# Function to add noise to costs
def add_noise(cost, std_dev):
    if isinstance(cost, (int, float)):  # If cost is scalar
        return cost + np.random.normal(0, std_dev)
    else:  # If cost is an array
        return cost + np.random.normal(0, std_dev, size=cost.shape)


# Function to evaluate equilibrium robustness
def evaluate_robustness(a, b, c, d, noise_std_dev, num_samples=100):
    robustness = []
    for _ in range(num_samples):
        # Add noise to the costs
        a_noisy = add_noise(a, noise_std_dev)
        b_noisy = add_noise(b, noise_std_dev)
        c_noisy = add_noise(c, noise_std_dev)
        d_noisy = add_noise(d, noise_std_dev)

        # Find equilibrium with noisy costs
        p_star, q_star = find_stackelberg_equilibrium(a_noisy, b_noisy, c_noisy, d_noisy)
        robustness.append((p_star, q_star))
    return robustness


# Performance Metrics
def calculate_performance_metrics(robustness_results):
    # Example: Calculate average power levels and deviation
    p_values = [p for p, _ in robustness_results]
    q_values = [q for _, q in robustness_results]
    avg_p = np.mean(p_values)
    avg_q = np.mean(q_values)
    p_deviation = np.std(p_values)
    q_deviation = np.std(q_values)
    return avg_p, avg_q, p_deviation, q_deviation


# Evaluate Robustness
robustness_results = evaluate_robustness(a, b, c, d, noise_std_dev)

# Calculate Performance Metrics
avg_p, avg_q, p_deviation, q_deviation = calculate_performance_metrics(robustness_results)

# Output Performance Metrics
print("Performance Metrics:")
print(f"Average Leader Power: {avg_p}")
print(f"Average Follower Power: {avg_q}")
print(f"Leader Power Deviation: {p_deviation}")
print(f"Follower Power Deviation: {q_deviation}")

# Plotting the results might be a good way to visualize the strategy outcomes
plt.figure(figsize=(10, 6))
powers = np.linspace(0.01, 3, 100)
responses = [follower_response(p, c, d) for p in powers]
plt.plot(powers, responses, label='Follower response')
plt.scatter([avg_p], [avg_q], color='red', label='Average Equilibrium Point')
plt.errorbar(avg_p, avg_q, xerr=p_deviation, yerr=q_deviation, fmt='o', color='green', label='Standard Deviation')
plt.title('Follower Response vs Leader Power Level with Robustness Metrics')
plt.xlabel('Leader Power Level (p)')
plt.ylabel('Follower Power Level (q)')
plt.legend()
plt.grid(True)
plt.show()
