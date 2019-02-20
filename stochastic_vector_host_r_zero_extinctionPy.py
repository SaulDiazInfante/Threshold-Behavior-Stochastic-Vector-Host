import numpy as np
import datetime
import matplotlib.pyplot as plt
from stochastic_vector_host_numerics import \
    NumericsStochasticVectorHostDynamics

k = 6
p = 1
r = p
T0 = 0.0
T = 600
#
sigma_v = 0.0139  # Vector noise intensity
sigma_h = 2.50620936  # Host noise intensity
lambda_h = 114.286  # Whole host population
lambda_v = 21000.0  # Vector birth rate
#
mu_v = 2.1  # Vector mortality rate
mu_h = 1.0 / 70.0  # Host mortality rate
alpha = 150.0
alpha_hat = mu_v / mu_h
#
x_zero = np.array([900.0, 90, 650, 50.0])

n_v_inf = lambda_v / mu_v
n_h_inf = lambda_h / mu_h
#
beta_h = alpha * mu_h / n_h_inf
beta_v = (mu_h * (1.0 + alpha_hat) - mu_v) / n_v_inf

svh = NumericsStochasticVectorHostDynamics()
svh.initialize_mesh(k, p, r, T0, T)
file_name = 'parameters.yml'
svh.load_parameters(file_name)
svh.set_parameters_stochastic_vector_host_dynamics(mu_v, beta_v, lambda_v,
                                                   mu_h, beta_h, lambda_h,
                                                   sigma_v, sigma_h, x_zero)
"""
x_det = svh.deterministic_linear_steklov()
xst = svh.linear_steklov()
currentDT = datetime.datetime.now()
postfix_time = currentDT.strftime("%Y-%m-%d-%H:%M:%S")
file_name = 'r_zero_figure' + postfix_time + '.png'
svh.plotting(file_name)
"""
t = svh.t
tk = svh.dt * svh.tau
svh.extinction_conditions()
# svh.save_parameters()
# svh.load_parameters(file_name)
# r_zero = svh.r_zero()
#
#
a = (beta_h * n_h_inf + mu_h) / (2 * mu_v * mu_h)
b = (beta_v * n_v_inf + mu_v) / (2 * mu_v * mu_h)
# Reproductive numbers
deterministic_r_zero = (beta_v * beta_h * n_v_inf * n_h_inf) / \
                       (mu_v * mu_h)
sigma_aster = (((a / b - mu_h / (beta_v * n_v_inf)) * sigma_v) ** 2
               +
               ((b / a - mu_v / (
                       beta_h * n_h_inf)) * sigma_h) ** 2) \
              / (mu_v * mu_h)
sigma_aster_a = ((a / b - mu_h / (beta_v * n_v_inf)) * sigma_v) ** 2
sigma_aster_b = ((b / a - mu_v / (beta_h * n_h_inf)) * sigma_h) ** 2

print('\n\n\t--------------------------------------------')
#
print('\t mu_h/(beta_v n_v):\t %5.64f' % (mu_h / (beta_v * n_v_inf)))
print('\t a / b:\t\t\t %5.64f' % (a / b))
print('\t mu_v/(beta_h n_h):\t %5.64f' % (mu_v / (beta_h * n_h_inf)))
print('\t b / a:\t\t\t %5.64f' % (b / a))
print('\n\n\t============================================')
print('\t beta_v n_v:\t %5.64f' % (beta_v * n_v_inf))
print('\t beta_h n_h:\t %5.64f' % (beta_h * n_h_inf))
print('\t sigma_aster_a:\t %5.64f' % sigma_aster_a)
print('\t sigma_aster_b:\t %5.64f' % sigma_aster_b)
print('\t sigma_aster:\t %5.64f' % (0.25 * sigma_aster))
print('\n\n\n')
