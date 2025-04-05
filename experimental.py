import numpy as np
import math
import pickle
from tqdm import tqdm
from scipy.optimize import newton, fsolve
from scipy.special import gamma, comb
from area import get_contour, get_contour_split, get_contour_exact
from mpmath import mp

# Задаем параметры
n = 14  # Порядок КФНСТ
s = 1  # Параметр s
mp.dps = 50  # точность вычислений


# Функция для вычисления символа Похгаммера (a)_k
def pochhammer(a, k):
    if k == 0:
        return 1
    else:
        return np.prod([a + i for i in range(k)])


# Функция P_n^s(x)
def p_n_s(x):
    total = mp.mpc(0,0)
    for k in range(n + 1):
        coeff = (-1) ** (n - k) * comb(n, k) * pochhammer(n + s - 1, k)
        total += coeff * x ** k
    return total


def p_n1_s(x):
    total = mp.mpc(0,0)
    for k in range(n):
        coeff = (-1) ** (n-1 - k) * comb(n-1, k) * pochhammer(n-1 + s - 1, k)
        total += coeff * x ** k
    return total

def p_n_s_prime(x):
    total = mp.mpc(0,0)
    for k in range(1, n + 1):
        coeff = (-1) ** (n - k) * comb(n, k) * pochhammer(n + s - 1, k)
        total += k * coeff * x ** (k - 1)
    return total


def p_n_s_prime2(x):
    total = mp.mpc(0,0)
    for k in range(2, n + 1):
        coeff = (-1) ** (n - k) * comb(n, k) * pochhammer(n + s - 1, k)
        total += k * (k - 1) * coeff * x ** (k - 2)
    return total


def newton_method(f, fprime, x0, max_iter=100, tol=1e-10):
    z = x0
    for i in range(max_iter):
        f_value = f(z)
        df_value = fprime(z)

        if df_value == 0:
            raise ValueError("Derivative is zero. No solution found.")

        z_next = z - f_value / df_value

        # Check for convergence
        if abs(z_next - z) < tol:
            return z_next

        z = z_next

    raise ValueError("Maximum iterations reached without convergence.")


def get_nodes_alternative(cont, need_alphas=False):
    # Основной алгоритм
    x_alphas = []
    possible_nodes = []
    coefficients = []
    skip_n = 0

    for dot in tqdm(cont, desc="processing points on contour"):
        try:
            # 1. Находим точку z_0
            z_0 = mp.mpc(dot[0],dot[1])  # z_0 из контура
            # 2. Полагаем x_0^α = -z_0
            x_0_alpha = -z_0/(2*n+s-2)
            # 3. Реализуем метод Ньютона для нахождения корня
            x_alpha = newton_method(p_n_s, fprime=p_n_s_prime, x0=x_0_alpha, max_iter=3000, tol=1e-9)
            #print(x_alpha)

            # проверка на корректность найденного корня
            if np.abs(p_n_s(x_alpha)) > 1e-6:
                print("skip")
                continue

            # 2 проверка на корректность найденного корня
            if x_alpha.real < 0:
                continue
            #print(f"x_alpha: {x_alpha}")
            # 4. Вычисляем искомый узел КФНСТ
            p_kn = 1/x_alpha

            # 5. Вычисляем коэффициент КФНСТ
            coefficients_A_kn = ((-1) ** (n + 1) * math.factorial(n) * (2 * n + s - 2) ** 2) / (
                    n ** 2 * gamma(n + s - 1) * p_kn ** 2 * (p_n1_s(1 / p_kn)) ** 2)

            # print(f"result n{n_iter}:p_kn: {p_kn}, coef: {coefficients_A_kn}")
            # Сохраняем результаты
            x_alphas.append(x_alpha)
            possible_nodes.append(p_kn)
            coefficients.append(coefficients_A_kn)
        except Exception as e:
            print(e)

    nodes_and_coeffs = set()
    dif_ans = 1e-3
    k = 0
    # Выводим результаты
    for i in range(len(possible_nodes)):
        flag = True
        '''
        if np.isnan(possible_nodes[i]) or np.isnan(coefficients[i]):
            print("NaN")
            continue
        '''
        for q in nodes_and_coeffs:
            if np.abs(possible_nodes[i] - q[0]) < dif_ans:
                flag = False
                break

        if flag:
            k += 1
            nodes_and_coeffs.add((possible_nodes[i], coefficients[i]))
            print(f"N {k}: {possible_nodes[i]}")
            print(f"C {k}: {coefficients[i]}")
            print(f"R {k}: {x_alphas[i]}")

    if need_alphas:
        out2_name = str(k) + "x_alphas.pkl"
        with open(out2_name, 'wb') as out_file:
            pickle.dump(x_alphas, out_file)
    return nodes_and_coeffs


def save_to_file(data):
    k = len(data)
    out_name = "" + str(n) + "n" + str(k) + "nodes" + ".pkl"
    with open(out_name, 'wb') as out_file:
        pickle.dump(data, out_file)
        print("saved", k, "nodes to:", out_name)


if __name__ == "__main__":
    points_amount = 1000
    for q in get_contour_exact(1e-8, desired=points_amount):
        print("Starting with", points_amount, "points")
        ans = get_nodes_alternative(q)

        save_to_file(ans)
