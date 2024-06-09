import numpy as np
from math import floor, ceil
from scipy.integrate import quad

def Fpn_plus(n, p1, M, b, L, phi, mu):
    E1 = np.sqrt(M**2 + p1**2)
    return (mu - np.sqrt((E1 + b)**2 + (2 * np.pi / L * (n + phi))**2))/(2 * np.pi)

def Fpn_minus(n, p1, M, b, L, phi, mu):
    E1 = np.sqrt(M**2 + p1**2)
    return (mu - np.sqrt((E1 - b)**2 + (2 * np.pi / L * (n + phi))**2))/(2 * np.pi)

def n_max(M,b,L,mu,phi,Fpn1):
    #print('тут перед if')
    if abs(b)>M and b<0:
      #print('теперь тут внутри if')
      if Fpn1(0, np.sqrt(b**2-M**2), M, b, L, phi, mu)<=0:
        n = -1
      else:
        n = floor(L*mu/(2*np.pi)-phi)
    elif mu**2<= (M+b)**2:
      n=-1
    else:
      n = floor(L/(2*np.pi)*np.sqrt(mu**2-(M+b)**2)-phi)
    return n

def integration_limits(M,b,mu,L,n,phi,Fpn1):
    # print(b**2-M**2)
    #print('M,b,mu,L,n,phi',M,b,mu,L,n,phi)
    SQ = (mu**2-4*(np.pi/L)**2*(n+phi)**2)**(0.5)
    #print('SQ =',SQ)
    if (b>0):
        p_left = 0
        p_right = ((-b+SQ)**2 - M**2)**(0.5)
    elif (abs(b)<=M):
        p_left = 0
        p_right = ((-b+SQ)**2 - M**2)**(0.5)
    elif (Fpn1(n, 0, M, b, L, phi, mu)>=0 and \
        Fpn1(n, np.sqrt(b**2-M**2), M, b, L, phi, mu)>0):
        p_left = 0
        p_right = ((-b+SQ)**2 - M**2)**(0.5)

    elif (Fpn1(n, 0, M, b, L, phi, mu)<0 and \
        Fpn1(n, np.sqrt(b**2-M**2), M, b, L, phi, mu)>0):
            p_left = ((-b-SQ)**2 - M**2)**(0.5)
            p_right = ((-b+SQ)**2 - M**2)**(0.5)
    else:
            assert  print('Возникла херня');print('M,b,mu,L,n,phi',M,b,mu,L,n,phi)
    return p_left, p_right

def n_steps_integration(M,L,n,phi, p_left, p_right, tol=0.01):
    # print('M =', M)
    ddF_max = ddF(M,L,n,phi,p_left)
    #print('ddF_max =',ddF_max)
    n = ceil((p_right-p_left)*ddF_max**(0.5)/(2*np.sqrt(3*tol)))
    if n>100:
      n=100
    elif n<1:
      n=2
    return n

def ddF(M,L,n,phi,x):
  #Функция расчета второй производной в точке
  a = 2*np.pi/L*(n+phi)
  if M==0 and a==0:
     M=1e-2

  return (M**2+a**2)*np.sqrt(M**2+x**2+a**2)/(M**2+x**2+a**2)**2


def F_integral(M,b,mu,L,n,phi,Fpn1,method='trapz'):
    if method == 'scipy':
      # Используем lambda функцию для перестановки аргументов
      result, error = quad(
          lambda p1: Fpn1(n, p1, M, b, L, phi, mu)*np.heaviside(Fpn1(n, p1, M, b, L, phi, mu),1),
          -np.inf, np.inf)
    else:
        N_MAX = n_max(M,b,L,mu,phi,Fpn1)
        # print('N_MAX =',N_MAX)
        if N_MAX == -1:
            result = 0.0
        else:
            p_left, p_right = integration_limits(M,b,mu,L,n,phi,Fpn1)
            n_points = n_steps_integration(M,L,n,phi, p_left, p_right, tol=0.01)
            if n_points < 20:
               n_points = 20
            if b<0 and b**2>M**2:
                p_max = np.sqrt(b**2-M**2)
                if p_max<p_right and n_points<20:
                  p1_1 = np.linspace(p_left, p_max, n_points+1)
                  p1_2 = np.linspace(p_max, p_right, n_points+1)[1:]
                  # Объединение массивов
                  p1 = np.concatenate((p1_1, p1_2))
                else:
                  p1 = np.linspace(p_left, p_right, n_points)
            else:
                p1 = np.linspace(p_left, p_right, n_points)
            # print('p_left',p_left)
            # print('p_right',p_right)
            # print('число точек',n_points)
            if method == 'scipy_lim':
              result, error = quad(
                lambda p: Fpn1(n, p, M, b, L, phi, mu)*np.heaviside(Fpn1(n, p, M, b, L, phi, mu),1),
                p_left, p_right)
              result *=2
            else:
              integrand_values = Fpn1(n, p1, M, b, L, phi, mu)
              # print('integ_values =',integrand_values)
              # print('p1 =',p1)
              result = 2*np.trapz(integrand_values, p1)
    return result


# Function to compute Omega_mu_L
def Omega_mu_L_v2(M, b, L, phi, mu):

    N_max = n_max(M,b,L,mu,phi,Fpn_plus)
    #print('N_max=',N_max)
    if N_max<0:
        fun_n_plus = 0.0
    elif N_max == 0:
        fun_n_plus = F_integral(M,b,mu,L,0,phi,Fpn_plus,method='trapz')
    else:
      n_range = np.arange(1, N_max + 1)
      #print('n_range=', n_range)
      f_vectorized = np.vectorize(F_integral)

      fun_n_plus = 2*np.sum(f_vectorized(M,b,mu,L,n_range,phi,Fpn_plus,method='trapz'))
      fun_n_plus += F_integral(M,b,mu,L,0,phi,Fpn_plus,method='trapz')

    N_max = n_max(M,-b,L,mu,phi,Fpn_plus)
    #print('N_max=',N_max)
    if N_max<0:
        fun_n_minus = 0.0
    elif N_max == 0:
        fun_n_minus = F_integral(M,-b,mu,L,0,phi,Fpn_plus,method='trapz')
    else:
        n_range = np.arange(1, N_max + 1)
        #print('n_range=', n_range)
        f_vectorized = np.vectorize(F_integral)

        fun_n_minus = 2*np.sum(f_vectorized(M,-b,mu,L,n_range,phi,Fpn_plus,method='trapz'))
        fun_n_minus += F_integral(M,-b,mu,L,0,phi,Fpn_plus,method='trapz')

    result = -(fun_n_plus + fun_n_minus)/L

    return result
    
    
 # Physical version function
def Omega_mu_L_phys(M, b, L, phi, mu):
    return (Omega_mu_L_v2(M, b, L, phi, mu)
            - Omega_mu_L_v2(0, b, L, phi, mu)
            + Omega_mu_L_v2(0, 0, L, phi, mu))