import matplotlib.pyplot as plt as plt
import numpy as np as np

def plot_transition_matrix(transition_matrix, title="Transition Matrix"):
    plt.figure(figsize=(10, 8))
    plt.imshow(transition_matrix, cmap="viridis")
    plt.colorbar(label="Transition Probability")
    plt.xlabel("Next State")
    plt.ylabel("Current State")
    plt.title(title)
    plt.show()
