import streamlit as st
import numpy as np
import NJL3Model
import NJL3Model_v2
import pandas as pd

def plot(x_val,y_val,name_x='x',name_y='y',title='fun'):
    # Отображаем графики
    data=pd.DataFrame({
    name_x:x_val,
    name_y:y_val})
    st.subheader(title)
    st.line_chart(data,x=name_x,y=name_y)

# Ползунки для параметров
L = st.slider('L', 0.01, 10.0, 1.0, step=0.01)
mu = st.slider('mu', 0.0, 20.0, 0.1)
phi = st.slider('phi', -12.0, 12.0, 0.0)
g = st.slider('g', -10.0, 10.0, -2/np.pi, step=0.01)

ratio = st.radio("Выберите ratio", ["b", "M"])

if ratio == "b":
    # Добавляем ползунок для b с конкретным значением
    b = st.slider('b', -5.0, 5.0, 0.0)
    M_step = st.number_input('Шаг по М:', min_value=0.01, max_value=10.0, value=0.1, step=0.01)
    # Добавляем слайдер с двумя палзунками для выбора диапазона значений с шагом
    M_values_range = st.slider('Диапазон M:', 0.01, 100.0, (0.1, 10.0), step=M_step)
    M_values = np.arange(M_values_range[0], M_values_range[1], M_step)  # Измените этот диапазон по вашему усмотрению
    if st.button('Нажмите для обновления графиков'):
        # Вычисление значений функции для каждого значения M
        x_values = [(M, b, L, phi, mu) for M in M_values]
        Omega_mu_L_values = [NJL3Model.calculate_fun(NJL3Model_v2.Omega_mu_L_phys, *x)
                            for x in x_values]
        plot(M_values,Omega_mu_L_values,name_x='M',name_y='y',title=r'${\Omega_{{\mu} L,phys}(M,b)}$')
#======================================================================
        # Вычисление значений функции для каждого значения M
        x_values = [(M, b, L, phi) for M in M_values]
        Omega_L_values = [NJL3Model.calculate_fun(NJL3Model.Omega_L_phys_opt, *x)
                            for x in x_values]
        plot(M_values,Omega_L_values,name_x='M',name_y='y',title=r'${\Omega_{L,phys}(M,b)}$')
#======================================================================
        # Вычисление значений функции для каждого значения M
        x_values = [(M, b) for M in M_values]
        dU_values = [NJL3Model.calculate_fun(NJL3Model.delta_U_phys_opt, *x)
                            for x in x_values]
        plot(M_values,dU_values,name_x='M',name_y='y',title=r'$\Delta U_{phys}{(M,b)}$')
#======================================================================
        # Вычисление значений функции для каждого значения M
        x_values = [(M, b, L, phi, mu, g) for M in M_values]
        Omega_phys_ren_values = [NJL3Model.calculate_fun(NJL3Model.Omega_phys_ren, *x)
                            for x in x_values]
        plot(M_values,Omega_phys_ren_values,name_x='M',name_y='y',title=r'$\Omega_{phys}^{ren}$')
else:
    # Добавляем ползунок для b с конкретным значением
    M = st.slider('M', 0.01, 100.0, 5.0)
    b_step = st.number_input('Шаг по b:', min_value=-100.0, max_value=100.0, value=1.0, step=0.01)
    # Добавляем слайдер с двумя палзунками для выбора диапазона значений с шагом
    b_values_range = st.slider('Диапазон b:', -100.0, 100.0, (-50.0, 50.0), step=b_step)
    b_values = np.arange(b_values_range[0], b_values_range[1], b_step)  # Измените этот диапазон по вашему усмотрению
    # Кнопка для обновления графиков
    if st.button('Нажмите для обновления графиков'):
        # Вычисление значений функции для каждого значения M
        x_values = [(M, float(b), L, phi, mu) for b in b_values]
        Omega_mu_L_values = [NJL3Model.calculate_fun(NJL3Model_v2.Omega_mu_L_phys, *x)
                            for x in x_values]
        plot(b_values,Omega_mu_L_values,name_x='x',name_y='y',title=r'${\Omega_{{\mu} L,phys}(M,b)}$')
#======================================================================
        # Вычисление значений функции для каждого значения M
        x_values = [(M, b, L, phi) for b in b_values]
        Omega_L_values = [NJL3Model.calculate_fun(NJL3Model.Omega_L_phys_opt, *x)
                            for x in x_values]
        plot(b_values,Omega_L_values,name_x='x',name_y='y',title=r'${\Omega_{L,phys}(M,b)}$')
#======================================================================
        # Вычисление значений функции для каждого значения M
        x_values = [(M, b) for b in b_values]
        dU_values = [NJL3Model.calculate_fun(NJL3Model.delta_U_phys_opt, *x)
                            for x in x_values]
        plot(b_values,dU_values,name_x='x',name_y='y',title=r'$\Delta U_{phys}{(M,b)}$')
#======================================================================
#======================================================================
        # Вычисление значений функции для каждого значения M
        x_values = [(M, b, L, phi, mu, g) for b in b_values]
        Omega_phys_ren_values = [NJL3Model.calculate_fun(NJL3Model.Omega_phys_ren, *x)
                            for x in x_values]
        plot(b_values,Omega_phys_ren_values,name_x='b',name_y='y',title=r'$\Omega_{phys}^{ren}$')