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
scale = 1.0e-2
#
sigma_v = 0.04965 * scale  # Vector noise intensity
sigma_h = 0.005 * scale  # Host noise intensity
lambda_h = 114.286 / 365.0  # Whole host population
lambda_v = 21000.0 / 365.0  # Vector birth rate

mu_v = 1.8 / 365.0  # Vector mortality rate
mu_h = 1.0 / (80.0 * 365.0)  # Host mortality rate
x_zero = np.array([2000.0, 1.0, 3500.0, 150.0])
n_v = lambda_v / mu_v
n_h = lambda_h / mu_h
eps = 1.0e-9
beta_v = mu_v / n_v + eps
beta_h = mu_h / n_h + eps

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
# r_zero = svh.r_zero()
svh.extinction_conditions()
# svh.save_parameters()
svh.load_parameters(file_name)
