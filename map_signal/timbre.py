import numpy as np
import matplotlib.pyplot as plt

import math

def discrete_normal_distribution(array, mean, std_dev):
    probabilities = []
    total_prob = 0

    for num in array:
        prob = 1 / (std_dev * math.sqrt(2*math.pi)) * math.exp(-((num-mean)**2) / (2*std_dev**2))
        probabilities.append(prob)
        total_prob += prob

    probabilities = [p/total_prob for p in probabilities] # Normalize probabilities so they sum to 1

    return probabilities

if __name__ == "__main__":
    array = np.array([100, 200, 300, 400, 500, 600, 700, 800, 900])
    mean = 300,
    std_dev = 200
    probabilities = discrete_normal_distribution(array, mean, std_dev)
    print(probabilities)
    # plot probabilities
    plt.plot(array, probabilities)
    plt.show()

