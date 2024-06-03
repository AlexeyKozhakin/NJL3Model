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
    # def test_1(self):
        # total_time_first = []
        # total_time_exp = []
        # total_time_opt = []
        # #M,b,mu,L,n,phi
        # for i in range(1):
            # M = random.uniform(0, 100)
            # L = random.uniform(1e-5, 100)
            # b = random.uniform(-100, 100)
            # mu = random.uniform(0, 100)
            # phi = 0
            # print(f'M={M}')
            # print(f'L={L}')
            # print(f'b={b}')
            # print(f'mu={mu}')
            # print(f'phi={phi}')
            
            # start_time = time.time()
            # Omega_mu_L_opt = NJL3Model.Omega_mu_L_opt(M, b, L, phi, mu)
            # end_time = time.time()
            # execution_time = end_time - start_time
            # total_time_opt.append(execution_time)
            # print("Время выполнения функции Omega_mu_L_opt:", execution_time, "секунд")            
            
            # start_time = time.time()
            # Omega_mu_L_first = NJL3Model.Omega_mu_L(M, b, L, phi, mu, p1_points=1000, n_points=1000)
            # end_time = time.time()
            # execution_time = end_time - start_time
            # total_time_first.append(execution_time)            
            # print("Время выполнения функции Omega_mu_L_first:", execution_time, "секунд")
            
            # start_time = time.time()
            # Omega_mu_L_exp = NJL3Model.int_F_n_sum(M,b,mu,L,phi)
            # end_time = time.time()
            # execution_time = end_time - start_time
            # print("Время выполнения функции Omega_mu_L_exp:", execution_time, "секунд")
            # total_time_exp.append(execution_time)
            
            # print(f'Omega_mu_L = {Omega_mu_L_first}')
            # print(f'Omega_mu_L_my = {Omega_mu_L_exp}')
            # print(f'Omega_mu_L_opt = {Omega_mu_L_opt}')
            # assert abs(Omega_mu_L_first - Omega_mu_L_exp)<1e-2
        # print("Test 1.0 passed")
        # print('Средние время для Omega_mu_L_first:',sum(total_time_first)/len(total_time_first))
        # print('Средние время для Omega_mu_L_exp:',sum(total_time_exp)/len(total_time_exp))
        # print('Средние время для Omega_mu_L_opt:',sum(total_time_opt)/len(total_time_opt))         

    # def test_2(self):
        # total_time_first = []
        # total_time_opt = []
        # diff = []
        # #M,b,mu,L,n,phi
        # params = {
            # 'M': [(1e-5, 100), 'float'],
            # 'b': [(-100, 100), 'float'],
            # 'mu': [(0, 100), 'float'],
            # 'L': [(1e-5, 100), 'float'],
        # }
        # n=2
        # dic_params = generate_parameter_sample_values(n, params, random_state=2)
        # for i in range(n):
            # M = dic_params['M'][i]
            # L = dic_params['L'][i]
            # b = dic_params['b'][i]
            # mu = dic_params['mu'][i]
            # phi = 0
            # print(f'M={M}')
            # print(f'L={L}')
            # print(f'b={b}')
            # print(f'mu={mu}')
            # print(f'phi={phi}')
            
            # start_time = time.time()
            # Omega_mu_L_opt = NJL3Model.Omega_mu_L_opt(M, b, L, phi, mu)
            # end_time = time.time()
            # execution_time = end_time - start_time
            # total_time_opt.append(execution_time)
            # print("Время выполнения функции Omega_mu_L_opt:", execution_time, "секунд")            
            
            # start_time = time.time()
            # Omega_mu_L_first = NJL3Model.Omega_mu_L(M, b, L, phi, mu, p1_points=1000, n_points=1000)
            # end_time = time.time()
            # execution_time = end_time - start_time
            # total_time_first.append(execution_time)            
            # print("Время выполнения функции Omega_mu_L_first:", execution_time, "секунд")
            
            # print(f'Omega_mu_L = {Omega_mu_L_first}')
            # print(f'Omega_mu_L_opt = {Omega_mu_L_opt}')
            # if Omega_mu_L_first != 0 and  Omega_mu_L_opt != 0:
                # diff.append(round(2*abs(Omega_mu_L_first - Omega_mu_L_opt)/abs(Omega_mu_L_first + Omega_mu_L_opt)*100))
                # print(f'Diff = {diff[-1]}%')
                # assert 2*abs(Omega_mu_L_first - Omega_mu_L_opt)/abs(Omega_mu_L_first + Omega_mu_L_opt)<5*1e-2
            # else:
                # diff.append(0)
        # print("Test 2.0 passed")
        # print('Средние время для Omega_mu_L_first:',sum(total_time_first)/len(total_time_first))
        # print('Средние время для Omega_mu_L_opt:',sum(total_time_opt)/len(total_time_opt))
        # print('Max Diff = ',max(diff))
        # i = diff.index(max(diff))
        # M = dic_params['M'][i]
        # L = dic_params['L'][i]
        # b = dic_params['b'][i]
        # mu = dic_params['mu'][i]
        # phi = 0
        # print(f'M={M}')
        # print(f'L={L}')
        # print(f'b={b}')
        # print(f'mu={mu}')
        # print(f'phi={phi}')
        
    def test_3(self):
        total_time_first = []
        total_time_exp = []
        total_time_opt = []
        M=8.64
        L=7
        mu=17
        phi=0
        #M,b,mu,L,n,phi
        for i in range(10):
            b = random.uniform(8, 15)

            phi = 0
            print(f'M={M}')
            print(f'L={L}')
            print(f'b={b}')
            print(f'mu={mu}')
            print(f'phi={phi}')
            
            start_time = time.time()
            Omega_mu_L_opt = NJL3Model.Omega_mu_L_opt(M, b, L, phi, mu)
            end_time = time.time()
            execution_time = end_time - start_time
            total_time_opt.append(execution_time)
            print("Время выполнения функции Omega_mu_L_opt:", execution_time, "секунд")            
            
            start_time = time.time()
            Omega_mu_L_first = NJL3Model.Omega_mu_L(M, b, L, phi, mu, p1_points=1000, n_points=1000)
            end_time = time.time()
            execution_time = end_time - start_time
            total_time_first.append(execution_time)            
            print("Время выполнения функции Omega_mu_L_first:", execution_time, "секунд")
            
            start_time = time.time()

            
            print(f'Omega_mu_L = {Omega_mu_L_first}')

            print(f'Omega_mu_L_opt = {Omega_mu_L_opt}')
            assert abs(Omega_mu_L_first - Omega_mu_L_opt)<1e-2
        print("Test 3.0 passed")
        print('Средние время для Omega_mu_L_first:',sum(total_time_first)/len(total_time_first))
        print('Средние время для Omega_mu_L_opt:',sum(total_time_opt)/len(total_time_opt))         

        