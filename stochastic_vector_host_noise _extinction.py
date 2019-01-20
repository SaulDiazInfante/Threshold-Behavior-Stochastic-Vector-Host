import numpy as np
import matplotlib.pyplot as plt
from stochastic_vector_host_numerics import \
    NumericsStochasticVectorHostDynamics

k = 6
p = 1
r = p
T0 = 0.0
T = 200

# Conditions for extinction by noise

sigma_v = 1.1                       # Vector noise intensity
sigma_h = 1.1                       # Host noise intensity
lambda_h = 114.286                  # Whole host population
lambda_v = 21000.0                  # Vector birth rate
beta_v = 0.00003900042152404787     # Host to vector transmission rate
beta_h = 0.00003269533157348633     # Vector to host transmission rate
mu_v = 2.1                          # Vector mortality rate
mu_h = 0.0142857                    # Host mortality rate
x_zero = np.array([2000.0, 1.0, 3500.0, 150.0])

svh = NumericsStochasticVectorHostDynamics()
svh.initialize_mesh(k, p, r, T0, T)
svh.set_parameters_stochastic_vector_host_dynamics(mu_v, beta_v, lambda_v,
                                                   mu_h, beta_h, lambda_h,
                                                   sigma_v, sigma_h, x_zero)
x_det = svh.deterministic_linear_steklov()
xst = svh.linear_steklov()

t = svh.t
tk = svh.dt * svh.tau

r_zero = svh.r_zero()
det_vector_cl = x_det[:, 0] + x_det[:, 1]
sto_vector_cl = xst[:, 0] + xst[:, 1]
print "======================================================================="
print "\n"
print "\r R_D:= ", r_zero[0]
print "\r R_S:= ", r_zero[1]
svh.extinction_conditions()

"""
    Auxiliary plot
"""
plt.style.use('ggplot')
fig2 = plt.figure()
ax_sv = plt.subplot2grid((4, 3), (0, 2))
ax_iv = plt.subplot2grid((4, 3), (0, 0), rowspan=2, colspan=2)
ax_nv = plt.subplot2grid((4, 3), (1, 2))

ax_sh = plt.subplot2grid((4, 3), (2, 2))
ax_ih = plt.subplot2grid((4, 3), (2, 0), rowspan=2, colspan=2)
ax_nh = plt.subplot2grid((4, 3), (3, 2))
det_color = '#ff0000'
sto_color = '#80ccff'

# Deterministic Plots


ax_sv.plot(tk, x_det[:, 0],
           color=det_color,
           marker='',
           alpha=1,
           lw=1,
           ls='-',
           ms=1,
           mfc='none',
           mec=det_color,
           label='Det'
           )

ax_iv.plot(tk, x_det[:, 1],
           color=det_color,
           marker='',
           alpha=1,
           lw=1,
           ls='-',
           ms=1,
           mfc='none',
           mec=det_color,
           label=r'Det'
           )
ax_nv.plot(tk, det_vector_cl,
           color=det_color,
           marker='',
           alpha=1,
           lw=1,
           ls='-',
           ms=1,
           mfc='none',
           mec=det_color,
           label=r'Det'
           )
ax_sh.plot(tk, x_det[:, 2],
           color=det_color,
           marker='',
           alpha=1,
           lw=1,
           ls='-',
           ms=1,
           mfc='none',
           mec=det_color,
           label=r'Det'
           )
ax_ih.plot(tk, x_det[:, 3],
           color=det_color,
           marker='',
           alpha=1,
           lw=1,
           ls='-',
           ms=1,
           mfc='none',
           mec=det_color,
           label=r'Det'
           )

ax_nh.plot(tk, x_det[:, 2] + x_det[:, 3],
           color=det_color,
           marker='',
           alpha=1,
           lw=1,
           ls='-',
           ms=1,
           mfc='none',
           mec=det_color,
           label=r'Det'
           )

ax_sh.set_xlabel(r'$t$')  # (days)')
#
ax_sv.set_ylabel(r'$S_V$')  # (r'Suceptibles vectors')
ax_iv.set_ylabel(r'$I_V$')  # (r'Infected vectors')
ax_nv.set_ylabel(r'$N_V$')  # (r'Conservative law')
ax_sh.set_ylabel(r'$S_H$')
ax_ih.set_ylabel(r'$I_H$')
ax_nh.set_ylabel(r'$N_H$')
#
#
# stochastic plot
#
ax_sv.plot(tk, xst[:, 0],
           color=sto_color,
           marker='.',
           alpha=0.04,
           lw=1,
           ls='-',
           ms=3,
           mfc='none',
           mec=sto_color,
           label='sto'
           )

ax_iv.plot(tk, xst[:, 1],
           color=sto_color,
           marker='.',
           alpha=0.04,
           lw=1,
           ls='-',
           ms=2,
           mfc='none',
           mec=sto_color,
           label='sto'
           )
ax_nv.plot(tk, sto_vector_cl,
           color=sto_color,
           marker='.',
           alpha=0.04,
           lw=1,
           ls='-',
           ms=2,
           mfc='none',
           mec=sto_color,
           label='sto'
           )

ax_sh.plot(tk, xst[:, 2],
           color=sto_color,
           marker='.',
           alpha=0.04,
           lw=1,
           ls='-',
           ms=2,
           mfc='none',
           mec=sto_color,
           label='sto'
           )
ax_ih.plot(tk, xst[:, 3],
           color=sto_color,
           marker='.',
           alpha=0.04,
           lw=1,
           ls='-',
           ms=2,
           mfc='none',
           mec=sto_color,
           label='sto'
           )
ax_nh.plot(tk, xst[:, 2] + xst[:, 3],
           color=sto_color,
           marker='.',
           alpha=0.04,
           lw=1,
           ls='-',
           ms=2,
           mfc='none',
           mec=sto_color,
           label='sto'
           )
ax_ih.legend(
        bbox_to_anchor=(0.15, 1, 1., .10),
        loc=0,
        ncol=2,
        numpoints=1,
        borderaxespad=0.04
    )
plt.tight_layout()
plt.show()
