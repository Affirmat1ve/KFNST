import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from mpmath import mp
import pickle


mp.dps =80
n=100
filename = 'e'+str(n)+'n'+str(n)+'nodes.pkl'
filepath = os.path.join('results', filename)
with open(filepath, 'rb') as file:
    pairs = list(pickle.load(file))
    pairs.sort(key=lambda x:x[0].real)
    for q in pairs:
        print(q[0],q[1])


def image(p):
    return 1 / (p - 1)


def kfnst(t):
    ans = mp.mpc(0,0)
    for elem in pairs:
        ans += elem[1] * (elem[0] / t * image(elem[0] / t))
        #ans += elem[1] *  image(elem[0] / t)
    return ans


if __name__ == "__main__":
    x = np.linspace(0.01, 10, 1000)
    y=[]
    for q in tqdm(x,desc="processing points on graph"):
        y.append(float(kfnst(q).real))
    plt.plot(x, y)
    plt.title("Graph for n = "+str(n))
    plt.xlabel("X-axis Label")
    plt.ylabel("Y-axis Label")
    plt.grid(True)  # Optional: adds a grid
    plt.show()  # Displays the graph
