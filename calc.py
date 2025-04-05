import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from mpmath import mp
import pickle


mp.dps =10


def image(p):
    return 1 / (p ** 2 + 1)


def kfnst(t):
    with open('14n14nodes.pkl', 'rb') as file:
        pairs = list(pickle.load(file))
    ans = mp.mpc(0,0)
    for elem in pairs:
        #print(elem[0],elem[1])
        ans += elem[1] * (elem[0] / t * image(elem[0] / t))
        #ans += elem[1] *  image(elem[0] / t)
    #print(ans[0].real)
    return ans


if __name__ == "__main__":
    x = np.linspace(0.01, 7, 1000)
    y=[]
    for q in tqdm(x,desc="processing points on graph"):
        y.append(float(kfnst(x)[0].real))
    plt.plot(x, y)
    plt.title("Simple Line Graph")
    plt.xlabel("X-axis Label")
    plt.ylabel("Y-axis Label")
    plt.grid(True)  # Optional: adds a grid
    plt.show()  # Displays the graph
