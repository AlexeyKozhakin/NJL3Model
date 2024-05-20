import numpy as np
import time

def calculate_fun(fun, *x):
    return fun(*x)
#========================================Omega_mu_L_phys Start =============================================
# Define E_1
def E_1(p1, M):
    return np.sqrt(M**2 + p1**2)

# Define the summand function for vectorized operations
def summand_vectorized(n, p1, M, b, L, phi, mu):
    E1 = E_1(p1, M)
    term1 = np.sqrt((E1 + b)**2 + (2 * np.pi / L * (n + phi))**2)
    term2 = np.sqrt((E1 - b)**2 + (2 * np.pi / L * (n + phi))**2)

    theta1 = np.maximum(mu - term1, 0)
    theta2 = np.maximum(mu - term2, 0)

    return (theta1 + theta2) / (2 * np.pi)

# Define the integrand for Omega_mu_L
def integrand_mu(p1, M, b, L, phi, mu, n_range):
    summation = np.sum(summand_vectorized(n_range[:, np.newaxis], p1, M, b, L, phi, mu), axis=0)
    return summation

# Define the limits for p1 integration
p1_min, p1_max = -10, 10  # Example integration limits for p1

# Function to compute Omega_mu_L
def Omega_mu_L(M, b, L, phi, mu, p1_points=200, n_points=200):
    p1 = np.linspace(p1_min, p1_max, p1_points)
    n_range = np.arange(-n_points, n_points + 1)
    
    integrand_values = integrand_mu(p1, M, b, L, phi, mu, n_range)
    result = np.trapz(integrand_values, p1)
    return -result / L

# Physical version function
def Omega_mu_L_phys(M, b, L, phi, mu, p1_points=100, n_points=100):
    return (Omega_mu_L(M, b, L, phi, mu, p1_points, n_points)
            - Omega_mu_L(0, b, L, phi, mu, p1_points, n_points)
            + Omega_mu_L(0, 0, L, phi, mu, p1_points, n_points))
#========================================Omega_L_phys Start =============================================
# Define E_1, B_pm functions
def E_1(p1, M):
    return np.sqrt(M**2 + p1**2)

def B_pm(M, p1, p3, b, sign):
    E1 = E_1(p1, M)
    return np.sqrt(p3**2 + (E1 + sign * b)**2)

# Define the integrand
def integrand(p1, p3, L, M, b, phi):
    B_plus = B_pm(M, p1, p3, b, 1)
    B_minus = B_pm(M, p1, p3, b, -1)
    
    term1 = 1 - 2 * np.cos(2 * np.pi * phi) * np.exp(-L * B_plus) + np.exp(-2 * L * B_plus)
    term2 = 1 - 2 * np.cos(2 * np.pi * phi) * np.exp(-L * B_minus) + np.exp(-2 * L * B_minus)
    
    return -np.log(term1 * term2) / (2 * np.pi)**2

# Define Omega_L_phys function
def omega_L(M, b, L, phi):
  # Define the limits for p1 and p3 integration
    p1_min, p1_max = -200, 200  # Example integration limits for p1
    p3_min, p3_max = -200, 200  # Example integration limits for p3
    p1_points = 200  # Number of points for p1
    p3_points = 200  # Number of points for p3

    # Create a meshgrid for p1 and p3
    p1 = np.linspace(p1_min, p1_max, p1_points)
    p3 = np.linspace(p3_min, p3_max, p3_points)
    P1, P3 = np.meshgrid(p1, p3)
    # Compute the integrand on the grid
    integrand_values = integrand(P1, P3, L, M, b, phi)

    # Perform the double integration using trapezoidal rule
    result_p1 = np.trapz(integrand_values, p1, axis=1)
    result = np.trapz(result_p1, p3)
    return result

# Define Omega_L_phys function
def Omega_L_phys(M, b, L, phi):
    return omega_L(M, b, L, phi) - omega_L(0, b, L, phi) + omega_L(0, 0, L, phi)
#========================================Omega_L_phys End ========================================
#========================================dU_phys Start =============================================
# Define the integrand function
# def integrand_dU(p, phi, M, b):
    # term1 = -M**2 / p
    # term2 = np.sqrt(b**2 + p**2 + 2 * b * p * np.cos(phi))
    # term3 = np.sqrt(b**2 + p**2 - 2 * b * p * np.cos(phi))
    # term4 = np.sqrt(M**2 + b**2 + p**2 + 2 * b * np.sqrt(M**2 + p**2 * np.cos(phi)**2))
    # term5 = np.sqrt(M**2 + b**2 + p**2 - 2 * b * np.sqrt(M**2 + p**2 * np.cos(phi)**2))
    # return (-term1 - term2 - term3 + term4 + term5) / (np.pi**2)

# def delta_U_phys(M, b):
    # # Define the limits for p integration
    # p_min, p_max = 1e-2, 1000  # Integration limits for p
    # p_points = 100  # Number of points for p

    # # Create an array of phi values
    # phi_values = np.linspace(0, np.pi/2, p_points)

    # # Compute the integrand on the grid
    # p_values = np.linspace(p_min, p_max, p_points)
    # p_mesh, phi_mesh = np.meshgrid(p_values, phi_values)
    # integrand_values = integrand_dU(p_mesh, phi_mesh, M, b)

    # # Perform the double integration using trapezoidal rule
    # result_phi = np.trapz(integrand_values, phi_values, axis=1)
    # result = np.trapz(result_phi, p_values)
    # return result
#=================================================================================================
# from scipy.integrate import dblquad
# from scipy.optimize import minimize

# # Определение функции Delta U_phys
# def delta_U_phys(M, b):
    # # Подынтегральная функция
    # def integrand(p, phi):
        # return -(-M**2/p - np.sqrt(b**2 + p**2 + 2*b*p*np.cos(phi))
                # - np.sqrt(b**2 + p**2 - 2*b*p*np.cos(phi))
                # + np.sqrt(M**2 + b**2 + p**2 + 2*b*np.sqrt(M**2 + p**2*np.cos(phi)**2))
                # + np.sqrt(M**2 + b**2 + p**2 - 2*b*np.sqrt(M**2 + p**2*np.cos(phi)**2))) * p / (np.pi**2)
        # #return -(-M**2/(np.pi**2))
    # # Вычисление интеграла
    # result, _ = dblquad(integrand, 0, np.pi/2, lambda p: 0, lambda p: np.inf)

    # return result    
#==========================================
def delta_U_phys(M, b, num_points=100):
    # Create arrays for phi and p values
    phi_values = np.linspace(0, np.pi / 2, num_points)
    p_values = np.linspace(0, 100, num_points)  # Set upper limit for p integration

    # Create a meshgrid for phi and p
    phi_mesh, p_mesh = np.meshgrid(phi_values, p_values)
    
    # Compute the integrand using vectorized operations
    term1 = -M**2
    term2 = np.sqrt(b**2 + p_mesh**2 + 2 * b * p_mesh * np.cos(phi_mesh))
    term3 = np.sqrt(b**2 + p_mesh**2 - 2 * b * p_mesh * np.cos(phi_mesh))
    term4 = np.sqrt(M**2 + b**2 + p_mesh**2 + 2 * b * np.sqrt(M**2 + p_mesh**2 * np.cos(phi_mesh)**2))
    term5 = np.sqrt(M**2 + b**2 + p_mesh**2 - 2 * b * np.sqrt(M**2 + p_mesh**2 * np.cos(phi_mesh)**2))
    
    integrand_values = -term1/(np.pi**2) - (-term2 - term3 + term4 + term5) * p_mesh / (np.pi**2)

    # Perform the double integration using the trapezoidal rule
    result_phi = np.trapz(integrand_values, phi_values, axis=0)
    result = np.trapz(result_phi, p_values)

    return result   
#========================================dU_phys End =============================================
def Omega_phys_ren(M, b, L, phi, mu, g):
    return M**2 / (2 * g) + delta_U_phys(M, b) + Omega_L_phys(M, b, L, phi) + Omega_mu_L_phys(M, b, L, phi, mu)