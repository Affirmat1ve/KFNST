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


def draw(n, chosen_image=image_sin,start=0.01, stop=100, gp=1000, s=1, folder='new_results'):
    mp.dps = n // 2 + 100
    filename = 's' + str(s) + 'n' + str(n) + '.pkl'
    filepath = os.path.join(folder, filename)
    with open(filepath, 'rb') as file:
        pairs = list(pickle.load(file))

    def kfnst(t, image):
        ans = mp.mpc(0, 0)
        for elem in pairs:
            ans += elem[1] * (mp.power(elem[0] / t, s) * image(elem[0] / t))
        ans *= mp.power(t, s - 1)
        return ans

    x = np.linspace(start+1e-3, stop, num=gp)
    y = []
    for q in tqdm(x, desc="processing points on graph"):
        y.append(float(kfnst(q, chosen_image).real))
    plt.plot(x, y)
    plt.title("Graph for n = " + str(n)+" s = "+str(s))
    plt.xlabel("X-axis Label")
    plt.ylabel("Y-axis Label")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    left_margin = 0    # начало_отрезка_на_котором_вычисляется_оригинал
    right_margin = 10  # конец_отрезка
    number_of_points = 200  # количество_точек_на_отрезке
    s_value = 1
    n_value = 180
    draw(n=n_value, chosen_image=image_sin,start=left_margin, stop=right_margin, gp=number_of_points, s=s_value, folder='new_results')