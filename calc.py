import numpy as np
import matplotlib.pyplot as plt

import pickle


def image(p):
    return 1 / (p ** 2 + 1)


def kfnst(t):
    with open('10nodes.pkl', 'rb') as file:
        pairs = pickle.load(file)
    ans = 0+0j
    print(pairs)
    for elem in pairs:
        ans += elem[1] * (elem[0] / t * image(elem[0] / t))
    return ans


if __name__ == "__main__":
    x = np.linspace(0.01, 7, 1000)
    y = kfnst(x)
    plt.plot(x, y)
    plt.title("Simple Line Graph")
    plt.xlabel("X-axis Label")
    plt.ylabel("Y-axis Label")
    plt.grid(True)  # Optional: adds a grid
    plt.show()  # Displays the graph
