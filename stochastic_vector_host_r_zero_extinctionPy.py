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

sigma_v = 0.1  # Vector noise intensity
sigma_h = 0.1  # Host noise intensity
lambda_h = 114.286  # Whole host population
lambda_v = 21000.0  # Vector birth rate

mu_v = 2.1  # Vector mortality rate
mu_h = 0.0142857  # Host mortality rate
x_zero = np.array([2000.0, 1.0, 3500.0, 150.0])
n_v = lambda_v / mu_v
n_h = x_zero[2] + x_zero[3]
beta_v = mu_v / n_v
beta_h = mu_h / n_h
mu_v = 2.1 - .01

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
#
t = svh.t
tk = svh.dt * svh.tau
r_zero = svh.r_zero()
svh.extinction_conditions()
# svh.save_parameters()
svh.load_parameters(file_name)
