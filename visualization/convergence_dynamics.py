import matplotlib.pyplot as plt as plt

def plot_convergence_dynamics(stationary_distribution, title="Convergence Dynamics"):
    plt.figure(figsize=(10, 6))
    plt.plot(stationary_distribution, marker='o')
    plt.xlabel("State")
    plt.ylabel("Probability")
    plt.title(title)
    plt.show()
