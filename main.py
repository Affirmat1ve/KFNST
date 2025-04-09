import numpy as np
import math
import pickle
from tqdm import tqdm #это библиотека для прогресс-бара
import time
import os
from scipy.special import gamma, comb
from area import get_contour_exact # это мой метод приближения контура
from mpmath import mp


def calculate_pairs(n,s=1,filename_prefix="",directory = 'new_results'):
    accuracy = int(3.4*n)
    mp.dps = accuracy  # точность вычислений
    points_amount = int(2.4*n)+10  # точек на контуре
    cont_tol = mp.power(10,-(accuracy//1.25)-10)  # точность контура
    newton_tol = mp.power(10,-(accuracy//1.25)-10) # точность метода ньютона

    # создаем контур из заданного количества точек


    # Функция для вычисления символа Похгаммера (a)_k
    def pochhammer(a, k):
        if k == 0:
            return mp.mpc(1, 0)
        else:
            return math.prod([a + i for i in range(k)])


    # Функция P_n^s(x)
    def p_n_s(x):
        total = mp.mpc(0, 0)
        lx = mp.mpc(1, 0)
        for k in range(n + 1):
            coeff = (-1) ** (n - k) * comb(n, k, exact=True) * pochhammer(n + s - 1, k)
            total += coeff * lx
            lx *= x
        return total


    # Функция P_(n-1)^s(x)
    def p_n1_s(x):
        total = mp.mpc(0, 0)
        lx = mp.mpc(1, 0)
        for k in range(n):
            coeff = (-1) ** (n - 1 - k) * comb(n - 1, k, exact=True) * pochhammer(n - 1 + s - 1, k)
            total += coeff * lx
            lx *= x
        return total


    def get_nodes(cont, need_alphas=False, method='mnewton', print_roots=False):
        # Основной алгоритм
        x_alphas = []
        possible_nodes = []
        coefficients = []
        error_count = 0

        for dot in tqdm(cont, desc="processing points on contour"):
            try:
                z_0 = mp.mpc(dot[0], dot[1])  # 1. Находим точку z_0
                x_0_alpha = -z_0 / (2 * n + s - 2)  # 2. Полагаем x_0^α = -z_0 и масштабируем
                # print(x_0_alpha)
                # 3. Реализуем метод Ньютона для нахождения корня
                x_alpha = mp.findroot(p_n_s, x_0_alpha, method=method, maxiter=3000, tol=newton_tol, full_output=False)

                # проверка на корректность найденного корня
                if np.abs(p_n_s(x_alpha)) > 1e-6: continue

                # 2 проверка на корректность найденного корня
                if x_alpha.real < 0: continue

                # 4. Вычисляем искомый узел КФНСТ и его коэффициент
                p_kn = mp.mpc(1 / x_alpha)
                coefficient_a_kn = mp.mpc((-1) ** (n%2 + 1) * math.factorial(n) * (2 * n + s - 2) ** 2)
                coefficient_a_kn /= n ** 2 * gamma(n + s - 1) * p_kn ** 2 * (p_n1_s(1 / p_kn)) ** 2
                if mp.isnan(coefficient_a_kn) or mp.isnan(p_kn): continue
                # Сохраняем результаты
                x_alphas.append(x_alpha)
                possible_nodes.append(p_kn)
                coefficients.append(coefficient_a_kn)
            except Exception as e:
                # print(e)
                error_count += 1

        nodes_and_coeffs = set()
        dif_ans = 1e-15  # точность различения корней
        k = 0
        # Отсееваем повторы, выводим результаты если требуется
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

        # Верифицируем результат
        if len(nodes_and_coeffs)==n:
            ok_flag = True
            for m in range(n):
                summ = 0
                for z in nodes_and_coeffs: summ += z[1] * z[0] ** (-m)
                summ -= 1 / gamma(s + m)
                if np.abs(summ) > 1e-5:
                    ok_flag = False
                    break
            if ok_flag:
                print(f"\033[32m Results are Correct!\033[0m")
                save_to_file(nodes_and_coeffs)
            else:
                print("Results are incorrect")
        else: print("Results are incorrect")

        # сохраняем полученные корни многочлена
        if need_alphas:
            out2_name = str(k) + "x_alphas.pkl"
            with open(out2_name, 'wb') as out_file:
                pickle.dump(x_alphas, out_file)

        # печатаем количество несошедшихся вызовов метода ньютона
        print("Errors occured", error_count)
        return nodes_and_coeffs

    def save_to_file(data):
        k = len(data)

        filename = filename_prefix+"n" + str(k) + ".pkl"
        filepath = os.path.join(directory, filename)
        # создаем папку с результатами, если её нет
        os.makedirs(directory, exist_ok=True)
        with open(filepath, 'wb') as outfile:
            pickle.dump(data, outfile)  # сохраняем массив узлов и коэффициентов
            print("saved", k, "nodes to:", filename)


    countour = get_contour_exact(cont_tol, desired=points_amount)
    print("Starting with approx", points_amount, "points")
    start_time = time.time()
    ans = get_nodes(countour, print_roots=False)
    print(f"Elapsed time: {time.time() - start_time:.4f} seconds")
    print(n, "n", len(ans))
    #if input("type q to skip")!='q':
        #save_to_file(ans)




if __name__ == "__main__":

    # Задаем важный параметр
    kfnst_n = 10 # Порядок КФНСТ
    # Задаем второстепенные параметры
    save_dir = 'new_results' # папка для результатов
    for q in range(15,150,5):
        calculate_pairs(q, directory=save_dir)
