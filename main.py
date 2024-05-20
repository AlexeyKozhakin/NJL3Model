import streamlit as st
import numpy as np
import NJL3Model
import pandas as pd


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
    M_values = np.arange(b_values_range[0], b_values_range[1], b_step)  # Измените этот диапазон по вашему усмотрению
else:
    # Добавляем ползунок для b с конкретным значением
    M = st.slider('M', 0.01, 100.0, 0.1)
    b_step = st.number_input('Шаг по b:', min_value=-100.0, max_value=100.0, value=1.0, step=0.01)
    # Добавляем слайдер с двумя палзунками для выбора диапазона значений с шагом
    b_values_range = st.slider('Диапазон b:', -100.0, 100.0, (-50.0, 50.0), step=b_step)
    b_values = np.arange(b_values_range[0], b_values_range[1], b_step)  # Измените этот диапазон по вашему усмотрению
    # Кнопка для обновления графиков
if st.button('Нажмите для обновления графиков'):
        # Вычисление значений функции для каждого значения M
        omega_values = [NJL3Model.Omega_mu_L_phys(M, b, L, phi, mu) 
                        for b in b_values]    
        # Отображаем графики
        data_omega_mu_L_phys=pd.DataFrame({
        'y':omega_values,
        'x':b_values})
        st.subheader(r'$\Omega_{{\mu}L,phys}{(M,b)}$')
        #st.subheader(r'$\Omega_{\mu L}{(M,b)}={-\frac 1L\int\frac{dp_1}{2\pi}\sum_{n=-\infty}^\infty\Bigg[\left(\mu-\sqrt{(E_1+b)^2+\frac{4\pi^2}{L^2}(n+\phi)^2}\right)\theta\left(\mu-\sqrt{(E_1+b)^2+\frac{4\pi^2}{L^2}(n+\phi)^2}\right)\nonumber\\&+&\left(\mu-\sqrt{(E_1-b)^2+\frac{4\pi^2}{L^2}(n+\phi)^2}\right)\theta\left(\mu-\sqrt{(E_1-b)^2+\frac{4\pi^2}{L^2}(n+\phi)^2}\right)\Bigg]}$')
        #st.subheader(r'$\Omega_{\mu L}{(M,b)}={-\frac 1L}$')        
        st.line_chart(data_omega_mu_L_phys,x='x',y='y')

#======================================================================
        # Вычисление значений функции для каждого значения M
        omega_values = [NJL3Model.Omega_L_phys(M, b, L, phi) 
                        for b in b_values]    
        # Отображаем графики
        data_omega_L_phys=pd.DataFrame({
        'y':omega_values,
        'x':b_values})
        st.subheader(r'$\Omega_{L,phys}{(M,b)}$')
        #st.subheader(r'$\Omega_{\mu L}{(M,b)}={-\frac 1L\int\frac{dp_1}{2\pi}\sum_{n=-\infty}^\infty\Bigg[\left(\mu-\sqrt{(E_1+b)^2+\frac{4\pi^2}{L^2}(n+\phi)^2}\right)\theta\left(\mu-\sqrt{(E_1+b)^2+\frac{4\pi^2}{L^2}(n+\phi)^2}\right)\nonumber\\&+&\left(\mu-\sqrt{(E_1-b)^2+\frac{4\pi^2}{L^2}(n+\phi)^2}\right)\theta\left(\mu-\sqrt{(E_1-b)^2+\frac{4\pi^2}{L^2}(n+\phi)^2}\right)\Bigg]}$')
        #st.subheader(r'$\Omega_{\mu L}{(M,b)}={-\frac 1L}$')        
        st.line_chart(data_omega_L_phys,x='x',y='y')
