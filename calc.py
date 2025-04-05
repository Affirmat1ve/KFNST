import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from mpmath import mp
import pickle


mp.dps =80
filename = '14n14nodes.pkl'
filepath = os.path.join('results', filename)
with open(filepath, 'rb') as file:
    pairs = list(pickle.load(file))
    for q in pairs:
        print(q[0],q[1])


def image(p):
    return 1 / (p ** 2 + 1)


def kfnst(t):
    ans = mp.mpc(0,0)
    for elem in pairs:
        ans += elem[1] * (elem[0] / t * image(elem[0] / t))
        #ans += elem[1] *  image(elem[0] / t)
    return ans


if __name__ == "__main__":
    x = np.linspace(0.01, 40, 100)
    y=[]
    for q in tqdm(x,desc="processing points on graph"):
        y.append(float(kfnst(q).real))
    plt.plot(x, y)
    plt.title("Simple Line Graph")
    plt.xlabel("X-axis Label")
    plt.ylabel("Y-axis Label")
    plt.grid(True)  # Optional: adds a grid
    plt.show()  # Displays the graph
