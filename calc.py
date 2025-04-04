import numpy as np
import matplotlib.pyplot as plt

import pickle


def image(p):
    return 1 / (p ** 2 + 1)


def kfnst(t):
    with open('40n9nodes.pkl', 'rb') as file:
        pairs = list(pickle.load(file))
        pairs.sort(key=lambda x:x[0])
    ans = 0+0j
    for elem in pairs:
        print(elem[0],elem[1])
        ans += elem[1] * (elem[0] / t * image(elem[0] / t))
        #ans += elem[1] *  image(elem[0] / t)
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
