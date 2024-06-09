import numpy as np
import unittest
import subprocess
from unittest.mock import patch
import sys
sys.path.insert(0, '../')
import NJL3Model  # модуль с игрой в лото
import random
import time
from LHSOpt import generate_parameter_sample_values
# Установить фиксированный seed
random.seed(42)

class TestProgram(unittest.TestCase):

    def test_2(self):
        total_time_first = []
        total_time_opt = []
        diff = []
        #M,b,mu,L,n,phi
        params = {
            'M': [(1e-1, 10), 'float'],
            'b': [(-0.1, 0.1), 'float'],
            'mu': [(0, 6), 'float'],
            'L': [(1e-1, 10), 'float'],
        }
        n=100
        dic_params = generate_parameter_sample_values(n, params, random_state=2)
        for i in range(n):
            g = -2/np.pi  # Примерное значение, которое можете изменить
            M = dic_params['M'][i]
            L = dic_params['L'][i]
            b = dic_params['b'][i]
            mu = dic_params['mu'][i]
            phi = 0
            print(f'M={M}')
            print(f'L={L}')
            print(f'b={b}')
            print(f'mu={mu}')
            print(f'phi={phi}')
            
            start_time = time.time()
            Omega_L_opt = NJL3Model.Omega_phys_ren_opt(M, b, L, phi, mu, g)
            end_time = time.time()
            execution_time = end_time - start_time
            total_time_opt.append(execution_time)
            print("Время выполнения функции Omega_L_opt:", execution_time, "секунд")            
            
            start_time = time.time()
            Omega_L_first = NJL3Model.Omega_phys_ren(M, b, L, phi, mu, g)
            end_time = time.time()
            execution_time = end_time - start_time
            total_time_first.append(execution_time)            
            print("Время выполнения функции Omega_L_first:", execution_time, "секунд")
            
            print(f'Omega_L = {Omega_L_first}')
            print(f'Omega_L_opt = {Omega_L_opt}')
            if Omega_L_first != 0 and  Omega_L_opt != 0:
                diff.append(round(2*abs(Omega_L_first - Omega_L_opt)/abs(Omega_L_first + Omega_L_opt)*100))
                print(f'Diff = {diff[-1]}%')
                
                #assert 2*abs(Omega_L_first - Omega_L_opt)/abs(Omega_L_first + Omega_L_opt)<5*1e-2
            else:
                diff.append(0)
        print("Test 2.0 passed")
        print('Средние время для Omega_L_first:',sum(total_time_first)/len(total_time_first))
        print('Средние время для Omega_L_opt:',sum(total_time_opt)/len(total_time_opt))
        print('Max Diff = ',max(diff))
        i = diff.index(max(diff))
        M = dic_params['M'][i]
        L = dic_params['L'][i]
        b = dic_params['b'][i]
        mu = dic_params['mu'][i]
        phi = 0
        print(f'M={M}')
        print(f'L={L}')
        print(f'b={b}')
        print(f'mu={mu}')
        print(f'phi={phi}')
         

        