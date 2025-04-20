import numpy as np
import math
import pickle
from tqdm import tqdm # это_библиотека_для_прогресс-бара
import time
import os
from scipy.special import gamma, comb, beta
from area import get_contour_exact # это_мой_метод_приближения_контура
from mpmath import mp


def calculate_pairs(n,s=1,filename_prefix="",directory = 'new_results'):
    accuracy = int(3.4*n)
    mp.dps = accuracy  # точность_вычислений
    points_amount = int(2.4*n)+10  # количество_точек_на_контуре
    cont_tol = mp.power(10,-(accuracy//1.25)-10)  # точность_контура
    newton_tol = mp.power(10,-(accuracy//1.25)-10) # точность_метода_ньютона


    # Функция_для_вычисления_символа_Похгаммера_(a)_k
    def pochhammer(a, k):
        if k == 0:
            return mp.mpc(1, 0)
        else:
            return math.prod([a + i for i in range(k)])


    # Функция_P_n^s(x)
    def p_n_s(x):
        total = mp.mpc(0, 0)
        lx = mp.mpc(1, 0)
        for k in range(n + 1):
            coeff = (-1) ** (n - k) * comb(n, k, exact=True) * pochhammer(n + s - 1, k)
            total += coeff * lx
            lx *= x
        return total


    # Функция_P_(n-1)^s(x)
    def p_n1_s(x):
        total = mp.mpc(0, 0)
        lx = mp.mpc(1, 0)
        for k in range(n):
            coeff = (-1) ** (n - 1 - k) * comb(n - 1, k, exact=True) * pochhammer(n - 1 + s - 1, k)
            total += coeff * lx
            lx *= x
        return total


    def get_nodes(cont, need_alphas=False, method='mnewton', print_roots=False):
        # Основной_алгоритм
        x_alphas = []
        possible_nodes = []
        coefficients = []
        error_count = 0

        for dot in tqdm(cont, desc="processing points on contour"):
            try:
                # 1. Берем_точку_на_контуре_гамма_и_масштабируем
                x_0_alpha = mp.mpc(-dot[0]/ (2 * n + s - 2), -dot[1]/ (2 * n + s - 2))

                # 2. Используем_метод_Ньютона_для_нахождения_корня
                x_alpha = mp.findroot(p_n_s, x_0_alpha, method=method, maxiter=3000, tol=newton_tol, full_output=False)

                # 3. Проверка_на_корректность_найденного_корня
                if np.abs(p_n_s(x_alpha)) > 1e-6: continue

                if x_alpha.real < 0: continue

                # 4. Вычисляем_искомый_узел_КФНСТ_и_его_коэффициент
                p_kn = mp.mpc(1 / x_alpha)
                if s==1:
                    coefficient_a_kn = mp.mpc((-1) ** (n%2 + 1) * n * (2 * n + s - 2) ** 2)
                    coefficient_a_kn /= n ** 2 * p_kn ** 2 * (p_n1_s(1 / p_kn)) ** 2
                else:
                    coefficient_a_kn = mp.mpc((-1) ** (n % 2 + 1)*n*beta(n,s-1)  * (2 * n + s - 2) ** 2)
                    coefficient_a_kn /= n ** 2  *gamma(s-1)* p_kn ** 2 * (p_n1_s(1 / p_kn)) ** 2

                # 5. Проверка_корректности_вычисленного_узла_и_коэффициента
                if mp.isnan(p_kn):
                    print("p_kn NaN")
                    continue
                if mp.isnan(coefficient_a_kn):
                    print("A_kn NaN")
                    continue
                # Сохраняем_результаты
                x_alphas.append(x_alpha)
                possible_nodes.append(p_kn)
                coefficients.append(coefficient_a_kn)
            except Exception as e:
                error_count += 1

        nodes_and_coeffs = set()
        dif_ans = 1e-15  # точность_различения_корней
        k = 0
        # Отсееваем_повторы,_выводим_результаты,_если_требуется
        for i in range(len(possible_nodes)):
            ok_indicator = True
            for q in nodes_and_coeffs:
                if np.abs(possible_nodes[i] - q[0]) < dif_ans:
                    ok_indicator = False
                    break
            if ok_indicator:
                k += 1
                nodes_and_coeffs.add((possible_nodes[i], coefficients[i]))
                if print_roots:
                    print(f"N {k}: {possible_nodes[i]}")
                    print(f"C {k}: {coefficients[i]}")
                    print(f"R {k}: {x_alphas[i]}\n")

        # Верифицируем_результат_по_сумме_коэффициентов
        if len(nodes_and_coeffs)==n:
            ok_flag = True
            for m in tqdm(range(n), desc="Verifying results"):
                summ = 0
                for z in nodes_and_coeffs: summ += z[1] * z[0] ** (-m)
                summ -= 1 / gamma(s + m)
                if np.abs(summ) > 1e-5:
                    ok_flag = False
                    print(f"Results are incorrect S {np.abs(summ)}")
                    break
            if ok_flag:
                print("\033[32m Results are Correct!\033[0m")
                # Сохраняем_результаты
                save_to_file(nodes_and_coeffs)
        else: print(f"Results are incorrect N {len(nodes_and_coeffs)}")

        # сохраняем_полученные_корни_многочлена,_если_требуется
        if need_alphas:
            out2_name = str(k) + "x_alphas.pkl"
            with open(out2_name, 'wb') as out_file:
                pickle.dump(x_alphas, out_file)

        return nodes_and_coeffs

    def save_to_file(data):
        k = len(data)

        filename = filename_prefix+"s"+str(s)+"n" + str(k) + ".pkl"
        filepath = os.path.join(directory, filename)
        # создаем_папку_с_результатами,_если_её_нет
        os.makedirs(directory, exist_ok=True)
        with open(filepath, 'wb') as outfile:
            pickle.dump(data, outfile)  # сохраняем_массив_узлов_и_коэффициентов
            print("saved", k, "nodes to:", filename)

    start_time = time.time()
    countour = get_contour_exact(cont_tol, desired=points_amount)
    print("Starting with approx", points_amount, "points.",f"Contour calc time: {time.time() - start_time:.2f} seconds")
    get_nodes(countour, print_roots=False)
    print(f"Full work time: {time.time() - start_time:.2f} seconds")




if __name__ == "__main__":

    # Задаем_порядок_КФНСТ
    kfnst_n = 101
    # Задаем_второстепенные_параметры
    save_dir = 'new_results' # папка_для_результатов
    calculate_pairs(kfnst_n, directory=save_dir, s=1)