import numpy as np
import math
import pickle
from tqdm import tqdm
from scipy.optimize import newton,fsolve
from scipy.special import gamma, comb
from area import get_contour, get_contour_split, get_contour_exact

# Задаем параметры
n = 12  # Порядок КФНСТ
s = 1  # Параметр s

def pochhammer(a, k):
    if k == 0:
        return 1
    else:
        return np.prod([a + i for i in range(k)])


# Функция P_n^s(x)
def p_n_s(x):
    total = 0 + 0j
    for k in range(n + 1):
        coeff = (-1) ** (n - k) * comb(n, k) * gamma(n + s+k - 1)/gamma(n+s-1)
        print(coeff)
        total += coeff * x ** k
    return total

def p_n_s2(x):
    total = 0 + 0j
    for k in range(n + 1):
        coeff = (-1) ** (n - k) * comb(n, k) * pochhammer(n + s - 1, k)
        print(coeff)
        total += coeff * x ** k
    return total

def p_n_s_prime(x):
    total = 0 + 0j
    for k in range(1, n + 1):
        coeff = (-1) ** (n - k) * comb(n, k) * gamma(n + s+k - 1)/gamma(n+s-1)
        total += k * coeff * x ** (k - 1)
    return total

def get_nodes_alternative(cont, need_alphas=False):
    # Основной алгоритм
    x_alphas = []
    possible_nodes = []
    coefficients = []
    skip_n = 0

    for dot in tqdm(cont, desc="processing points on contour"):
        try:
            # 1. Находим точку z_0
            z_0 = dot[0] + 1j * dot[1]  # z_0 из контура
            for m in range(10):
                # 2. Полагаем x_0^α = -z_0
                x_0_alpha = -z_0
                x_0_alpha*= 0.2*m

                # 3. Реализуем метод Ньютона для нахождения корня
                x_alpha = newton(p_n_s, x0=x_0_alpha, fprime=p_n_s_prime, maxiter=7000, tol=1e-8)
                #print(x_alpha)
                # проверка на корректность найденного корня
                if np.abs(p_n_s(x_alpha)) > 1e-6:
                    continue

                # print(f"{n_iter}:z_0: {z_0}, x_alpha: {x_0_alpha}")
                # 4. Вычисляем искомый узел КФНСТ
                p_kn = (2 * n + s - 2) * x_alpha

                # 5. Вычисляем коэффициент КФНСТ
                coefficients_A_kn = ((-1) ** (n + 1) * math.factorial(n) * (2 * n + s - 2) ** 2) / (
                        n ** 2 * gamma(n + s - 1) * p_kn ** 2 * (p_n_s(1 / p_kn)) ** 2)

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
        if np.isnan(possible_nodes[i]) or np.isnan(coefficients[i]):
            continue
        for q in nodes_and_coeffs:
            if np.abs(possible_nodes[i] - q[0]) < dif_ans:
                flag = False
                break

        if flag:
            k += 1
            nodes_and_coeffs.add((possible_nodes[i], coefficients[i]))
            print(f"Узел {k}: {possible_nodes[i]}, Коэффициент: {coefficients[i]}")

    if need_alphas:
        out2_name = str(k) + "x_alphas.pkl"
        with open(out2_name, 'wb') as out_file:
            pickle.dump(x_alphas, out_file)
    return nodes_and_coeffs

def save_to_file(data):
    k = len(data)
    out_name = "Maple"+str(n)+"n"+str(k)+"nodes" + ".pkl"
    with open(out_name, 'wb') as out_file:
        pickle.dump(data, out_file)
        print("saved", k, "nodes to:", out_name)

if __name__ == "__main__":
    p_n_s(0.1)
'''    points_amount = 500
    for q in get_contour_exact(1e-8, desired=points_amount):
        print("Starting with", points_amount, "points")
        ans = get_nodes_alternative(q)
        save_to_file(ans)'''