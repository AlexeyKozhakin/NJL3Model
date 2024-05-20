import numpy as np
import time
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
def Omega_mu_L(M, b, L, phi, mu, p1_points=100, n_points=200):
    p1 = np.linspace(p1_min, p1_max, p1_points)
    n_range = np.arange(-n_points, n_points + 1)
    
    integrand_values = integrand_mu(p1, M, b, L, phi, mu, n_range)
    result = np.trapz(integrand_values, p1)
    return -result / L

# Physical version function
def Omega_mu_L_phys(M, b, L, phi, mu, p1_points=100, n_points=200):
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
    p1_min, p1_max = -100, 100  # Example integration limits for p1
    p3_min, p3_max = -100, 100  # Example integration limits for p3
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
#========================================Omega_L_phys End =============================================
