import numpy as np
import math
import pickle
from tqdm import tqdm #это библиотека для прогресс-бара
import time
import os
from scipy.special import gamma, comb
from area import get_contour_exact # это мой метод приближения контура
from mpmath import mp

RED = "\033[31m"
RESET = "\033[0m"
GREEN = "\033[32m"

mp.dps = 80
s=1
'''
n = 30
filename = '' + str(n) + 'n' + str(n) + 'nodes.pkl'
filepath = os.path.join('results', filename)
with open(filepath, 'rb') as file:
    pairs = list(pickle.load(file))
    pairs.sort(key=lambda x: x[0].real)
    for q in pairs:
        print(q[0], q[1])
'''

if __name__ == "__main__":

    # Specify the directory
    directory = 'results'

    # List to hold .pkl files
    pkl_files = []

    # Iterate through the directory
    for filename in os.listdir(directory):
        if filename.endswith('.pkl'):
            pkl_files.append(os.path.join(directory, filename))
    for q in pkl_files:
        with open(q, 'rb') as file:
            pairs = list(pickle.load(file))
        n = len(pairs)
        ok_flag = True
        for m in range(n):
            sum=0
            for z in pairs: sum+=z[1]*z[0]**(-m)
            sum-=1/gamma(s+m)
            if np.abs(sum)>1e-5:
                ok_flag = False
                break
        if ok_flag: print(f"{GREEN}{q}{RESET}")
        else: print(f"\033[31m{q}{RESET}")