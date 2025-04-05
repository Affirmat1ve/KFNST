import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root

# Определяем функцию для вычисления модуля
def modulus(z):
    return np.abs(np.exp(np.sqrt(1 + 1 / z ** 2)) / (z * (1 + np.sqrt(1 + 1 / z ** 2))))

    # Создаем массив значений z в комплексной плоскости


def get_contour(tolerance, net_steps, visualize=False):

    r = np.linspace(-2, -0.01, net_steps)
    i = np.linspace(-1.5, 1.5, net_steps)

    re, im = np.meshgrid(r, i)
    z = re + 1j * im

    # Вычисляем модуль
    mod = modulus(z)

    # Находим пересечения, где модуль равен 1
    # tolerance = 0.0009  # Допуск для нахождения пересечений
    indices = np.where(np.abs(mod - 1) < tolerance)

    # Извлекаем координаты пересечений
    intersection_re = re[indices]
    intersection_im = im[indices]

    if visualize:
        # Визуализируем результаты
        plt.figure(figsize=(10, 8))
        plt.contour(re, im, mod, levels=20, cmap='viridis', alpha=0.7)
        plt.scatter(intersection_re, intersection_im, color='red', label='Пересечения (|Z|=1)', zorder=5)
        plt.title('Пересечения функции в комплексной плоскости')
        plt.xlabel('Re(Z)')
        plt.ylabel('Im(Z)')
        plt.xticks(np.arange(-1.7, 0.1, 0.05))  # X-axis ticks from 0 to 10 with a step of 1
        plt.yticks(np.arange(-1.3, 1.3, 0.05))
        plt.axhline(0, color='black', lw=0.5, ls='--')
        plt.axvline(0, color='black', lw=0.5, ls='--')
        plt.colorbar(label='Модуль функции')
        plt.legend()
        plt.grid()
        plt.show()
    print(len(intersection_re))
    return zip(intersection_re, intersection_im)

    # Выводим координаты пересечений


def get_contour_split(tolerance, desired = 500,visualize=False):
    area_split_29 = [(-0.2, 0, 1, 1.1), (-0.4, -0.2, 1.05, 1.15), (-0.6, -0.4, 1.1, 1.125),
                     (-0.75, -0.6, 1.05, 1.125), (-0.9, -0.75, 1, 1.1), (-1.05, -0.9, 0.9, 1.05),
                     (-1.2, -1.05, 0.75, 0.95), (-1.3, -1.2, 0.65, 0.8), (-1.35, -1.3, 0.55, 0.7),
                     (-1.4, -1.35, 0.4, 0.6), (-1.45, -1.4, 0.35, 0.5), (-1.5, -1.45, 0.25, 0.4),
                     (-1.525, -1.475, 0.15, 0.25), (-1.525, -1.475, 0.05, 0.15),
                     (-1.525, -1.5, -0.05, 0.05),
                     (-1.525, -1.475, -0.15, -0.05), (-1.525, -1.475, -0.25, -0.15), (-1.5, -1.45, -0.4, -0.25),
                     (-1.45, -1.4, -0.5, -0.35), (-1.4, -1.35, -0.6, -0.4), (-1.35, -1.3, -0.7, -0.55),
                     (-1.3, -1.2, -0.8, -0.65), (-1.2, -1.05, -0.95, -0.75), (-1.05, -0.9, -1.05, -0.9),
                     (-0.9, -0.75, -1.1, -1), (-0.75, -0.6, -1.125, -1.05), (-0.6, -0.4, -1.125, -1.1),
                     (-0.4, -0.2, -1.15, -1.05), (-0.2, 0, -1.1, -1)]
    net_steps = 100
    for q in area_split_29:
        net_steps = 100
        intersection_re=[]
        while(len(intersection_re)<desired/29):
            net_steps*=1.5
            net_steps=int(net_steps)
            r = np.linspace(q[0], q[1], net_steps)
            i = np.linspace(q[2], q[3], net_steps)

            re, im = np.meshgrid(r, i)
            z = re + 1j * im

            # Вычисляем модуль
            mod = modulus(z)

            # Находим пересечения, где модуль равен 1
            # tolerance = 0.0009  # Допуск для нахождения пересечений
            indices = np.where(np.abs(mod - 1) < tolerance)

            # Извлекаем координаты пересечений
            intersection_re = re[indices]
            intersection_im = im[indices]

        if visualize:
            # Визуализируем результаты
            plt.figure(figsize=(10, 8))
            plt.contour(re, im, mod, levels=20, cmap='viridis', alpha=0.7)
            plt.scatter(intersection_re, intersection_im, color='red', label='Пересечения (|Z|=1)', zorder=5)
            plt.title('Пересечения функции в комплексной плоскости')
            plt.xlabel('Re(Z)')
            plt.ylabel('Im(Z)')
            plt.xticks(np.arange(-1.7, 0.1, 0.05))  # X-axis ticks from 0 to 10 with a step of 1
            plt.yticks(np.arange(-1.3, 1.3, 0.05))
            plt.axhline(0, color='black', lw=0.5, ls='--')
            plt.axvline(0, color='black', lw=0.5, ls='--')
            plt.colorbar(label='Модуль функции')
            plt.legend()
            plt.grid()
            plt.show()
        print(len(intersection_re))
        yield zip(intersection_re, intersection_im)

def vert_func(t,a):
    z=t+1j*a
    return np.abs(np.exp(np.sqrt(1 + 1 / z ** 2)) / (z * (1 + np.sqrt(1 + 1 / z ** 2))))-1

def hor_func(t,a):
    z=a+1j*t
    return np.abs(np.exp(np.sqrt(1 + 1 / z ** 2)) / (z * (1 + np.sqrt(1 + 1 / z ** 2))))-1

def get_contour_exact(tolerance, desired = 500, visualize=False):
    area_split_vertical = [(-1.2, -1.05, 0.75, 0.95), (-1.3, -1.2, 0.65, 0.8), (-1.35, -1.3, 0.55, 0.7),
     (-1.4, -1.35, 0.4, 0.6), (-1.45, -1.4, 0.35, 0.5), (-1.5, -1.45, 0.25, 0.4),
     (-1.525, -1.475, 0.15, 0.25), (-1.525, -1.475, 0.05, 0.15),
     (-1.525, -1.5, -0.05, 0.05),
     (-1.525, -1.475, -0.15, -0.05), (-1.525, -1.475, -0.25, -0.15), (-1.5, -1.45, -0.4, -0.25),
     (-1.45, -1.4, -0.5, -0.35), (-1.4, -1.35, -0.6, -0.4), (-1.35, -1.3, -0.7, -0.55),
     (-1.3, -1.2, -0.8, -0.65), (-1.2, -1.05, -0.95, -0.75)]
    area_split_horizontal = [(-0.2, 0, 1, 1.1), (-0.4, -0.2, 1.05, 1.15), (-0.6, -0.4, 1.1, 1.125),
                     (-0.75, -0.6, 1.05, 1.125), (-0.9, -0.75, 1, 1.1), (-1.05, -0.9, 0.9, 1.05),(-1.05, -0.9, -1.05, -0.9),
                     (-0.9, -0.75, -1.1, -1), (-0.75, -0.6, -1.125, -1.05), (-0.6, -0.4, -1.125, -1.1),
                     (-0.4, -0.2, -1.15, -1.05), (-0.2, 0, -1.1, -1)]


    net_steps = 100
    intersection_re = []
    intersection_im = []
    for q in area_split_vertical:
        net_steps = desired//29
        for w in np.linspace(q[2],q[3],net_steps):
            solution = root(vert_func, (q[0]+q[1])/2, args=(w,))
            intersection_re.append(solution.x[0])
            intersection_im.append(w)
    for q in area_split_horizontal:
        net_steps = desired // 29
        for w in np.linspace(q[0], q[1], net_steps):
            solution = root(hor_func, (q[2] + q[3]) / 2, args=(w,))
            intersection_re.append(w)
            intersection_im.append(solution.x[0])
    if visualize:
        # Визуализируем результаты
        plt.figure(figsize=(10, 8))
        plt.scatter(intersection_re, intersection_im, color='red', label='Пересечения (|Z|=1)')
        plt.title('Пересечения функции в комплексной плоскости')
        plt.xlabel('Re(Z)')
        plt.ylabel('Im(Z)')
        plt.xticks(np.arange(-1.7, 0.1, 0.1))  # X-axis ticks from 0 to 10 with a step of 1
        plt.yticks(np.arange(-1.3, 1.3, 0.1))
        plt.axhline(0, color='black', lw=0.5, ls='--')
        plt.axvline(0, color='black', lw=0.5, ls='--')
        plt.colorbar(label='Модуль функции')
        plt.legend()
        plt.grid()
        plt.show()

    yield zip(intersection_re, intersection_im)

def res_print(intersection_re, intersection_im):
    print("Координаты пересечений (|Z|=1):")
    for x, y in zip(intersection_re, intersection_im):
        print(f"({x:.2f}, {y:.2f})")


if __name__ == "__main__":
    for q in get_contour_exact(1e-8, desired=1000, visualize=True):
        print(q)
