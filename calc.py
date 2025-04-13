import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from mpmath import mp
import pickle


def image_with_gap(p):
    return mp.exp(-p) / p


def image_bessel(p):
    return 1 / mp.sqrt(p ** 2 + 1)


def image_sin(p):
    return 1 / (p ** 2 + 1)


def draw(n, chosen_image=image_sin, stop=100, gp=1000, s=1):
    mp.dps = n // 2 + 100
    filename = 's' + str(s) + 'n' + str(n) + '.pkl'
    filepath = os.path.join('new_results', filename)
    with open(filepath, 'rb') as file:
        pairs = list(pickle.load(file))

    def kfnst(t, image):
        ans = mp.mpc(0, 0)
        for elem in pairs:
            ans += elem[1] * (mp.power(elem[0] / t, s) * image(elem[0] / t))
        ans *= mp.power(t, s - 1)
        return ans

    x = np.linspace(0.01, stop, num=gp)
    y = []
    z = []
    d = []
    for q in tqdm(x, desc="processing points on graph"):
        y.append(float(kfnst(q, chosen_image).real))
        #z.append(mp.besselj(0, q))
        z.append(mp.sin(q))
        #d.append(float(kfnst(q, chosen_image).real) - mp.besselj(0, q))
        d.append(float(kfnst(q, chosen_image).real) - mp.sin(q))
    plt.plot(x, y)
    plt.plot(x, z)
    plt.plot(x, d)
    plt.title("Graph for n = " + str(n))
    plt.xlabel("X-axis Label")
    plt.ylabel("Y-axis Label")
    plt.grid(True)  # Optional: adds a grid
    plt.show()  # Displays the graph


if __name__ == "__main__":
    right_margin = 250  # задать до какого значения t нарисовать график
    number_of_points = 1000  # количество точек на графике
    s_value = 3
    n_value = 50
    draw(n=n_value, chosen_image=image_sin, stop=right_margin, gp=number_of_points, s=s_value)
