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
sigma_v = 1.01227223  # Vector noise intensity
sigma_h = 1.019  # Host noise intensity
lambda_h = 114.286  # Whole host population
lambda_v = 21000.0  # Vector birth rate
#
mu_v = 2.1  # Vector mortality rate
mu_h = 1.0 / 70.0  # Host mortality rate
alpha = 1.05
alpha_hat = mu_v / mu_h
#
x_zero = np.array([900.0, 90, 650, 50.0])

n_v_inf = lambda_v / mu_v
n_h_inf = lambda_h / mu_h
#
beta_h = alpha * mu_h / n_h_inf
beta_v = mu_v / n_v_inf

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
