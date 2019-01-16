import numpy as np
import matplotlib.pyplot as plt
# import itertools
# import warnings
import random
import os
import cPickle as Pickle
from stochastic_vector_host_numerics import \
    NumericsStochasticVectorHostDynamics

# from matplotlib import rcParams

# Stencil parameters

k = 6
p = 1
r = p
T0 = 0.0
T = 200

# SDE parameters from literature

sigma_v = 1.95  # Vector noise intensity
sigma_h = 1.95  # Host noise intensity
lambda_h = 114.286  # Whole host population
lambda_v = 21000.0  # Vector birth rate
beta_v = 0.00003900042152404787  # Host to vector transmission rate
beta_h = 0.00003269533157348633  # Vector to host transmission rate
mu_v = 2.1  # Vector mortality rate
mu_h = 0.0142857  # Host mortality rate

x_zero = np.array([2000.0, 1.0, 3500.0, 150.0])

sto_vector_host = NumericsStochasticVectorHostDynamics()
sto_vector_host.initialize_mesh(k, p, r, T0, T)
sto_vector_host.set_parameters_stochastic_vector_host_dynamics(mu_v, beta_v,
                                                               lambda_v,
                                                               mu_h, beta_h,
                                                               lambda_h,
                                                               sigma_v,
                                                               sigma_h,
                                                               x_zero)
x_det = sto_vector_host.deterministic_lsoda()
# x_det = sto_vector_host.deterministic_linear_steklov()
xst = sto_vector_host.em()


t = sto_vector_host.t
tk = sto_vector_host.dt * sto_vector_host.tau

r_zero = sto_vector_host.r_zero()
det_vector_cl = x_det[:, 0] + x_det[:, 1]
sto_vectot_cl = xst[:, 0] + xst[:, 1]
print "======================================================================="
print "\n"
print "\r R_D:= ", r_zero[0]
# print "\r R_S:= ", r_zero[1]
"""
    Auxiliary plot
"""
fig2 = plt.figure()
ax1 = plt.subplot2grid((2, 3), (0, 0))
ax2 = plt.subplot2grid((2, 3), (0, 1))
ax3 = plt.subplot2grid((2, 3), (0, 2))

ax4 = plt.subplot2grid((2, 3), (1, 0))
ax5 = plt.subplot2grid((2, 3), (1, 1))
ax6 = plt.subplot2grid((2, 3), (1, 2))
# Deterministic Plots
ax1.plot(tk, x_det[:, 0],
         color='red',
         marker='',
         alpha=1,
         lw=1,
         ls='-',
         ms=1,
         mfc='none',
         mec='red',
         label='Det'
         )

ax2.plot(tk, x_det[:, 1],
         color='red',
         marker='',
         alpha=1,
         lw=1,
         ls='-',
         ms=1,
         mfc='none',
         mec='red',
         label=r'Det'
         )
ax3.plot(tk, det_vector_cl,
         color='red',
         marker='',
         alpha=1,
         lw=1,
         ls='-',
         ms=1,
         mfc='none',
         mec='red',
         label=r'Det'
         )
ax4.plot(tk, x_det[:, 2],
         color='red',
         marker='',
         alpha=1,
         lw=1,
         ls='-',
         ms=1,
         mfc='none',
         mec='red',
         label=r'Det'
         )
ax5.plot(tk, x_det[:, 3],
         color='red',
         marker='',
         alpha=1,
         lw=1,
         ls='-',
         ms=1,
         mfc='none',
         mec='red',
         label=r'Det'
         )

ax6.plot(tk, x_det[:, 2] + x_det[:, 3],
         color='red',
         marker='',
         alpha=1,
         lw=1,
         ls='-',
         ms=1,
         mfc='none',
         mec='red',
         label=r'Det'
         )

ax4.set_xlabel(r'$t$')  # (days)')
#
ax1.set_ylabel(r'$S_V$')  # (r'Suceptibles vectors')
ax2.set_ylabel(r'$I_V$')  # (r'Infected vectors')
ax3.set_ylabel(r'$N_V$')  # (r'Infected Hosts')
ax4.set_ylabel(r'$S_H$')
ax5.set_ylabel(r'$I_H$')
ax6.set_ylabel(r'$N_H$')
ax1.legend(
    bbox_to_anchor=(0.15, 1, 1., .10),
    loc=3,
    ncol=4,
    numpoints=1,
    borderaxespad=0.4
    )

ax1.plot(tk, xst[:, 0],
         color='#696969',
         marker='.',
         alpha=0.4,
         lw=1,
         ls='-',
         ms=3,
         mfc='none',
         mec='#696969',
         label='sto'
         )

ax2.plot(tk, xst[:, 1],
         color='#696969',
         marker='.',
         alpha=0.4,
         lw=1,
         ls='-',
         ms=2,
         mfc='none',
         mec='#696969',
         label='sto'
         )
ax3.plot(tk, sto_vectot_cl,
         color='#696969',
         marker='.',
         alpha=0.4,
         lw=1,
         ls='-',
         ms=2,
         mfc='none',
         mec='#696969',
         label='sto'
         )

ax4.plot(tk, xst[:, 2],
         color='#696969',
         marker='.',
         alpha=0.4,
         lw=1,
         ls='-',
         ms=2,
         mfc='none',
         mec='#696969',
         label='sto'
         )
ax5.plot(tk, xst[:, 3],
         color='#696969',
         marker='.',
         alpha=0.4,
         lw=1,
         ls='-',
         ms=2,
         mfc='none',
         mec='#696969',
         label='sto'
         )
ax6.plot(tk, xst[:, 2] + xst[:, 3],
         color='#696969',
         marker='.',
         alpha=0.4,
         lw=1,
         ls='-',
         ms=2,
         mfc='none',
         mec='#696969',
         label='sto'
         )
plt.tight_layout()
plt.show()
