import argparse
import numpy as np
from markov_chain_inference.analysis.stationary_distribution import StationaryDistribution
from markov_chain_inference.analysis.temperature_effects import TemperatureEffects
from markov_chain_inference.visualization.plot_transition_matrix import plot_transition_matrix
from markov_chain_inference.visualization.convergence_dynamics import plot_convergence_dynamics



def main(transition_matrix_path):
    transition_matrix = np.load(transition_matrix_path)

    # Plot the original transition matrix
    plot_transition_matrix(transition_matrix)

    # Calculate stationary distribution
    stationary = StationaryDistribution(transition_matrix)
    stationary_distribution = stationary.calculate()
    plot_convergence_dynamics(stationary_distribution)

    # Temperature adjustment
    temp_effects = TemperatureEffects(transition_matrix)
    adjusted_matrix = temp_effects.adjust_temperature(temperature=0.7)
    plot_transition_matrix(adjusted_matrix, title="Adjusted Transition Matrix (T=0.7)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Analysis on Transition Matrix")
    parser.add_argument('--transition_matrix_path', type=str, required=True,
                        help="Path to transition matrix file (npy format)")
    args = parser.parse_args()
    main(args.transition_matrix_path)
