"""
    This class solves numerically the model reported in
    Threshold behaviour of a stochastic vector host model
    Saul Diaz-Infante, Adrian Acu\~na Zegarra,
    with the Linear Steklov method.
    \begin{equation}
        \begin{aligned}
            d S_V &=
                \left [
                    \Lambda_V - \beta_V / N_H S_V I_H - \mu_V S_V
                \right ] dt
                - \sigma_V S_V I_H dB_t^V,
                \\
            d I_V &=
                \left [
                   \beta_V / N_H S_V I_H - \mu_V I_V
                \right ]
                dt
                + \sigma_V S_V I_H dB_t^V,
                \\
            d S_H & = \Lambda_H - \beta_H / N_V S_H I_V - \mu_H S_H
            d I_H &=
                \left [
                    \beta_H / N_V S_H I_V - \mu_H I_H
                \right ] dt
                + \sigma_H S_H I_V d B^H_t .
       \end{aligned}
    \end{equation}
"""
from stochastic_vector_host_model import StochasticVectorHostDynamics
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from matplotlib import rcParams
from scipy.integrate import ode
# from scipy.integrate import odeint

rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['DejaVu']
params = {
    'figure.titlesize': 10,
    'axes.titlesize': 10,
    'axes.labelsize': 10,
    'font.size': 10,
    'legend.fontsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'text.usetex': True
    }
rcParams.update(params)


class NumericsStochasticVectorHostDynamics(StochasticVectorHostDynamics):

    def __init__(self, t_f=2 ** 8, t_0=0.0, k=12, p=5, r=5):

        self.k = np.int(k)
        self.p = np.int(p)
        self.r = np.int(r)
        self.t_0 = np.int(t_0)

        """
        self.n = np.int(10.0 ** k)
        self.P = np.int(10.0 ** p)
        self.rr = np.int(10.0 ** r)
        """

        self.n = np.int(2.0 ** k)
        self.P = np.int(2.0 ** p)
        self.rr = np.int(2.0 ** r)
        self.t_f = t_f

        self.dt = self.t_f / np.float(self.n)
        self.index_n = np.arange(self.n + 1)

        # set of index to Ito integral
        self.tau = self.index_n[0:self.n + 1:self.P]
        self.t = np.linspace(0, self.t_f, self.n + 1)

        #
        self.d_op_t = np.float(self.rr) * self.dt
        self.ll = self.n / self.rr

        # diffusion part
        self.normal_dist_sampling_1 = np.random.randn(np.int(self.n))
        self.normal_dist_sampling_2 = np.random.randn(np.int(self.n))

        self.d_w_1 = np.sqrt(self.dt) * self.normal_dist_sampling_1
        self.d_w_2 = np.sqrt(self.dt) * self.normal_dist_sampling_2
        self.w_1 = np.cumsum(self.d_w_1)
        self.w_1 = np.concatenate(([0], self.w_1))
        self.w_2 = np.cumsum(self.d_w_2)
        self.w_2 = np.concatenate(([0], self.w_2))
        self.w_inc_1 = 0.0
        self.w_inc_2 = 0.0
        self.w_inc = np.array([[self.w_inc_1], [self.w_inc_1],
                               [self.w_inc_2]], dtype=np.float128)
        #
        #
        # Arrays for solutions
        #

        self.x_det = np.zeros([self.ll + 1, 4], dtype=np.float128)
        self.x_sto = np.zeros([self.ll + 1, 4], dtype=np.float128)
        super(NumericsStochasticVectorHostDynamics, self).__init__()

    def initialize_mesh(self, k, p, r, t_0, t_f):
        """
            Set stencil parameters
        """
        # Stencil of the mesh
        self.k = np.int(k)
        self.p = np.int(p)
        self.r = np.int(r)
        self.t_0 = np.int(t_0)

        self.n = np.int(10.0 ** k)
        self.P = np.int(10.0 ** p)
        self.rr = np.int(10.0 ** r)
        """
        self.n = np.int(2.0 ** k)
        self.P = np.int(2.0 ** p)
        self.rr = np.int(2.0 ** r)
        """
        self.t_f = t_f

        #

        self.dt = self.t_f / np.float(self.n)
        self.index_n = np.arange(self.n + 1)

        # set of index to Ito integral
        self.tau = self.index_n[0:self.n + 1:self.P]
        self.t = np.linspace(0, self.t_f, self.n + 1)
        #
        self.d_op_t = np.float(self.rr) * self.dt
        self.ll = self.n / self.rr
        # diffusion part
        self.normal_dist_sampling_1 = np.random.randn(np.int(self.n))
        self.normal_dist_sampling_2 = np.random.randn(np.int(self.n))

        self.d_w_1 = np.sqrt(self.dt) * self.normal_dist_sampling_1
        self.d_w_2 = np.sqrt(self.dt) * self.normal_dist_sampling_2

        self.w_1 = np.cumsum(self.d_w_1)
        self.w_1 = np.concatenate(([0], self.w_1))
        self.w_2 = np.cumsum(self.d_w_2)
        self.w_2 = np.concatenate(([0], self.w_2))

        # Arrays for solutions
        self.x_det = np.zeros([self.ll + 1, 4], dtype=np.float128)
        self.x_sto = np.zeros([self.ll + 1, 4], dtype=np.float128)

    def ai(self, x, xj, h):
        a = self.a(x)

        dw = self.w_inc
        drift = np.dot(self.b(xj), dw).reshape(4, )
        return h * a + xj + drift - x

    #

    def bem(self, flag=0):
        # %  Preallocate x_sto for efficiency.
        h = self.d_op_t
        ll = self.ll
        rj = self.rr

        if flag == 1:
            h = self.dt
            ll = self.n
            rj = 1
            self.x_sto = np.zeros([ll + 1, 3])
        #
        self.x_sto[0] = self.x_zero
        for j in np.arange(ll):
            xj = self.x_sto[j]
            self.w_inc_1 = np.sum(self.d_w_1[rj * j:rj * (j + 1)])
            self.w_inc_2 = np.sum(self.d_w_2[rj * j:rj * (j + 1)])
            self.w_inc = np.array([[self.w_inc_1], [self.w_inc_1],
                                   [self.w_inc_2], [self.w_inc_2]])
            increment = fsolve(self.ai, xj, args=(xj, h))
            self.x_sto[j + 1] = increment
        xbem = self.x_sto
        return xbem

    def em(self, flag=0):
        # %  Preallocate x_sto for efficiency.
        d_op_t = self.d_op_t
        ll = self.ll
        rr = self.rr

        if flag == 1:
            d_op_t = self.dt
            ll = self.n
            rr = 1
        self.x_sto = np.zeros([ll + 1, 4])
        self.x_sto[0] = self.x_zero
        for j in np.arange(ll):
            self.w_inc_1 = np.sum(self.d_w_1[rr * j:rr * (j + 1)])
            self.w_inc_2 = np.sum(self.d_w_2[rr * j:rr * (j + 1)])
            self.w_inc = np.array([[self.w_inc_1], [self.w_inc_1],
                                   [self.w_inc_2], [self.w_inc_2]])
            #
            xj = self.x_sto[j]
            aj = self.a(self.x_sto[j])
            diffusion = np.dot(self.b(self.x_sto[j]), self.w_inc).reshape(4, )
            increment = xj + d_op_t * aj + diffusion

            sign_em = np.sign(increment)
            sign_em = (sign_em < 0)
            sign_em_vector = sign_em[0: 2]
            sign_em_host = sign_em[2: 4]
            if sign_em_vector[1]:
                increment[0: 2] = xj[0: 2] + d_op_t * aj[0: 2] \
                                  - diffusion[0: 2]
            if sign_em_host[1]:
                increment[2: 4] = xj[2: 4] + d_op_t * aj[2: 4] \
                                  - diffusion[2: 4]
            self.x_sto[j + 1] = increment
        xem = self.x_sto
        return xem

    def milstein(self, flag=0):
        # %  Preallocate x_sto for efficiency.
        d_op_t = self.d_op_t
        ll = self.ll
        rr = self.rr

        if flag == 1:
            d_op_t = self.dt
            ll = self.n
            rr = 1
        self.x_sto = np.zeros([ll + 1, 4])
        self.x_sto[0] = self.x_zero
        for j in np.arange(ll):
            self.w_inc_1 = np.sum(self.d_w_1[rr * j:rr * (j + 1)])
            self.w_inc_2 = np.sum(self.d_w_2[rr * j:rr * (j + 1)])
            self.w_inc = np.array([[self.w_inc_1], [self.w_inc_1],
                                   [self.w_inc_2], [self.w_inc_2]])
            w_inc_milstein = self.w_inc ** 2 - d_op_t * np.ones([4, 1])
            diffusion = np.dot(self.b(self.x_sto[j]), self.w_inc).reshape(4, )
            milstein_correction = 0.5 * np.dot(self.b_prime(self.x_sto[j]),
                                               w_inc_milstein).reshape(4, )
            drift = self.x_sto[j] + d_op_t * self.a(self.x_sto[j])
            increment = drift + diffusion + milstein_correction
            # conservative law improvement
            sign_mls = np.sign(increment)
            sign_mls = (sign_mls < 0)
            sign_mls_vector = sign_mls[0: 2]
            sign_mls_host = sign_mls[2: 4]
            if any(sign_mls_vector):
                # increment[0: 2] = drift[0: 2] - diffusion[0: 2] \
                #                  + milstein_correction[0: 2]
                increment[0: 2] = [0.0, 0.0]
            if any(sign_mls_host):
                # increment[2: 4] = drift[2: 4] - diffusion[2: 4] \
                #                  + milstein_correction[2: 4]
                increment[2: 4] = [0.0, 0.0]
            self.x_sto[j + 1] = np.reshape(increment, 4)
        xml = self.x_sto
        return xml

    def tamed_em(self, flag=0):
        d_op_t = self.d_op_t
        ll = self.ll
        rr = self.rr
        if flag == 1:
            d_op_t = self.dt
            ll = self.n
            rr = 1

        self.x_sto[0] = self.x_zero
        for j in np.arange(ll):
            self.w_inc_1 = np.sum(self.d_w_1[rr * j:rr * (j + 1)])
            self.w_inc_2 = np.sum(self.d_w_2[rr * j:rr * (j + 1)])
            self.w_inc = np.array([[self.w_inc_1], [self.w_inc_1],
                                   [self.w_inc_2], [self.w_inc_1]])
            xj = self.x_sto[j]
            aj = self.a(xj)
            naj = 1 + d_op_t * np.linalg.norm(aj)
            diffusion = np.dot(self.b(xj), self.w_inc).reshape(4, )
            increment = xj + d_op_t / naj * self.a(xj) + diffusion
            self.x_sto[j + 1] = np.transpose(increment)
        x_sto = self.x_sto
        return x_sto

    def tamed_em_sabanis(self, flag=0):
        d_op_t = self.d_op_t
        ll = self.ll
        rr = self.rr
        alpha = 0.5
        # l_alpha = 0.25
        # r_alpha = 0.5 * 3 * l_alpha + 2
        if flag == 1:
            d_op_t = self.dt
            ll = self.n
            rr = 1
        self.x_sto = np.zeros([ll + 1, 4])
        self.x_sto[0] = self.x_zero
        for j in np.arange(ll):
            self.w_inc_1 = np.sum(self.d_w_1[rr * j:rr * (j + 1)])
            self.w_inc_2 = np.sum(self.d_w_2[rr * j:rr * (j + 1)])
            self.w_inc = np.array([[self.w_inc_1], [self.w_inc_1],
                                   [self.w_inc_2], [self.w_inc_1]])

            xj = self.x_sto[j]
            aj = self.a(xj)
            bj = self.b(xj)
            n_alpha_tamed_1 = 1 + ((j + 1) ** (-alpha)) * (
                    np.linalg.norm(aj) + np.linalg.norm(bj) ** 2)

            # nalphaTamed2 = 1 + ((j + 1)**(-alpha)) * (np.linalg.norm(xj)**r)

            diffusion = np.dot(self.b(xj), self.w_inc).reshape(4, )
            increment = xj + d_op_t / n_alpha_tamed_1 * aj \
                        + diffusion / n_alpha_tamed_1
            self.x_sto[j + 1] = np.transpose(increment)
        x_temsab = self.x_sto
        return x_temsab

    def deterministic_lsoda(self):
        def f(y, t):
            dy = self.a(y)
            return dy

        def f_ode(t, y, arg1=1):
            dy = self.a(y)
            return dy

        def jacobian_f(t, y, arg1=1):
            s_v = y[0]
            i_v = y[0]
            s_h = y[0]
            i_h = y[0]

            n_v = s_v + i_v
            n_h = s_h + i_h

            beta_v = self.beta_v
            beta_h = self.beta_h
            mu_v = self.mu_v
            mu_h = self.mu_h

            f_sv = [
                - beta_v * i_h / n_h - mu_v,
                0,
                beta_v * s_v * i_h / n_h ** 2,
                beta_v * s_v * i_h / n_h ** 2 - beta_v * s_v / n_h
                ]
            f_iv = [
                beta_v * i_h / n_h,
                - mu_v,
                - beta_v * s_v * i_h / n_h ** 2,
                - beta_v * s_v * i_h / n_h ** 2 + beta_v * s_v / n_h
                ]
            f_sh = [
                beta_h * s_h * i_v / n_v ** 2,
                beta_h * i_v * s_h / n_v ** 2 - beta_h * s_h / n_v,
                -(beta_h * i_v / n_v + mu_h),
                0
                ]
            f_ih = [
                - beta_h * s_h * i_v / n_v ** 2,
                - beta_h * s_h * i_v / n_v ** 2 + beta_h * s_h / n_v,
                beta_h * i_v / n_v,
                -mu_v
                ]
            jf = np.array([f_sv, f_iv, f_sh, f_ih]).reshape([4, 4])
            return jf

        y_0 = self.x_zero
        self.x_det[0] = y_0
        # solver = ode(f_ode, jacobian_f).set_integrator('zvode', method='BDF')
        solver = ode(f_ode).set_integrator('vode')
        solver.set_initial_value(y_0, 0.0)
        # t = self.tau
        # x_det = odeint(f, y_0, t)
        for j in np.arange(self.ll):
            x_det = solver.integrate(self.tau[j + 1])
            self.x_det[j + 1] = x_det
        return self.x_det

    def linear_steklov(self):
        h = self.d_op_t
        ll = self.ll
        rr = self.rr

        beta_v = self.beta_v
        beta_h = self.beta_h
        lambda_v = self.lambda_v
        # lambda_h = self.lambda_h
        mu_v = self.mu_v
        mu_h = self.mu_h

        eps = np.finfo(float).eps
        self.x_sto[0] = self.x_zero

        def phi(ai, bi):
            if np.isclose(ai, eps) or np.isclose(np.exp(ai * h), 1.0):
                phi_x = bi * h
            else:
                phi_x = (np.exp(ai * h) - 1.0) * ai ** (-1.0) * bi
            return phi_x

        for j in np.arange(ll):
            self.w_inc_1 = np.sum(self.d_w_1[rr * j:rr * (j + 1)])
            self.w_inc_2 = np.sum(self.d_w_2[rr * j:rr * (j + 1)])
            self.w_inc = np.array([[self.w_inc_1], [self.w_inc_1],
                                   [self.w_inc_2], [self.w_inc_2]])
            xj = self.x_sto[j, 0]
            yj = self.x_sto[j, 1]
            zj = self.x_sto[j, 2]
            wj = self.x_sto[j, 3]

            a1 = - (beta_v * wj + mu_v)
            b1 = lambda_v
            x = np.exp(h * a1) * xj + phi(a1, b1)

            a2 = - mu_v
            b2 = beta_v * xj * wj
            y = np.exp(a2 * h) * yj + phi(a2, b2)

            a3 = - beta_h * yj
            b3 = mu_h * wj
            z = np.exp(a3 * h) * zj + phi(a3, b3)

            a4 = - mu_h
            b4 = beta_h * zj * yj
            w = np.exp(a4 * h) * wj + phi(a4, b4)

            drift_increment = np.array([[x], [y], [z], [w]])
            winner_increment = np.dot(self.b(self.x_sto[j]), self.w_inc)
            st_k = drift_increment + winner_increment

            # conservative law improvement
            sign_st_k = np.sign(st_k)
            sign_st_k = (sign_st_k < 0)
            sign_st_k_vector = sign_st_k[0: 2]
            sign_st_k_host = sign_st_k[2: 4]
            if any(sign_st_k_vector):
                st_k[0: 2] = drift_increment[0: 2] - winner_increment[0: 2]
            if any(sign_st_k_host):
                st_k[2: 4] = drift_increment[2: 4] - winner_increment[2: 4]
            self.x_sto[j + 1] = np.reshape(st_k, 4)
        x_sto = self.x_sto
        return x_sto

    def deterministic_linear_steklov(self):
        h = self.d_op_t
        ll = self.ll
        # rr = self.rr

        beta_v = self.beta_v
        beta_h = self.beta_h
        lambda_v = self.lambda_v
        # lambda_h = self.lambda_h
        mu_v = self.mu_v
        mu_h = self.mu_h

        eps = np.finfo(np.float64).eps
        self.x_det[0] = self.x_zero

        def phi(ai, bi):
            if np.isclose(ai, eps) or np.isclose(np.exp(ai * h), 1.0):
                phi_x = bi * h
                # phi_x = (h * np.exp(ai * h) - h) * bi * (ai ** -1.0)
            else:
                phi_x = (np.exp(ai * h) - 1.0) * ai ** (-1.0) * bi
            return phi_x

        for j in np.arange(ll):
            xj = self.x_det[j, 0]
            yj = self.x_det[j, 1]
            zj = self.x_det[j, 2]
            wj = self.x_det[j, 3]

            a1 = - (beta_v * wj + mu_v)
            b1 = lambda_v
            x = np.exp(h * a1) * xj + phi(a1, b1)

            a2 = - mu_v
            b2 = beta_v * xj * wj
            y = np.exp(a2 * h) * yj + phi(a2, b2)

            a3 = - beta_h * yj
            b3 = mu_h * wj
            z = np.exp(a3 * h) * zj + phi(a3, b3)

            a4 = - mu_h
            b4 = beta_h * zj * yj
            w = np.exp(a4 * h) * wj + phi(a4, b4)

            st_k = np.array([x, y, z, w])
            self.x_det[j + 1, :] = st_k[:]

        x_det_st_km = self.x_det
        return x_det_st_km

    def save_data(self):
        """
            Method to save the numerical solutions and parameters of 
            Van der Pol ODE.
        """

        # t=self.t[0:-1:self.rr].reshape([self.t[0:-1:self.rr].shape[0],1])

        t = self.dt * self.tau
        u_det1 = self.x_det[:, 0]
        u_det2 = self.x_det[:, 1]
        u_det3 = self.x_det[:, 2]
        u_det4 = self.x_det[:, 3]

        u_sto1 = self.x_sto[:, 0]
        u_sto2 = self.x_sto[:, 1]
        u_sto3 = self.x_sto[:, 2]
        u_sto4 = self.x_sto[:, 3]

        tag_par = np.array([
            'k = ',
            'r = ',
            't_0 = ',
            'rr = ',
            't_f = ',
            'dt = ',
            'd_op_t = ',
            'll = ',
            'mu_v = ',
            'beta_v = ',
            'lambda =',
            'mu_h = ',
            'beta_h = ',
            'sigma_v = ',
            'sigma_h = ',
            'Sv0 = ',
            'Iv0 = ',
            'SH0 = ',
            'VH0 = '
            ])
        parameter_values = np.array([
            self.k,
            self.r,
            self.t_0,
            self.rr,
            self.t_f,
            self.dt,
            self.d_op_t,
            self.ll,
            self.mu_v,
            self.beta_v,
            self.lambda_v,
            self.mu_h,
            self.beta_h,
            self.sigma_v,
            self.sigma_h,
            self.x_zero[0, 0],
            self.x_zero[0, 1],
            self.x_zero[0, 2],
            self.x_zero[0, 3]
            ])
        str_prefix = str(self.d_op_t)
        name1 = 'parameters' + str_prefix + '.txt'
        name2 = 'det_solution' + str_prefix + '.txt'
        name3 = 'sto_solution' + str(self.dt) + '.txt'

        parameters = np.column_stack((tag_par, parameter_values))
        np.savetxt(name1, parameters, delimiter=" ", fmt="%s")
        data_det = np.transpose((t, u_det1, u_det2, u_det3, u_det4))
        data_sto = np.transpose((self.t, u_sto1, u_sto2, u_sto3, u_sto4))
        np.savetxt(name2, data_det, fmt='%1.8f', delimiter='\t')
        np.savetxt(name3, data_sto, fmt='%1.8f', delimiter='\t')
        return

    def plotting(self, file_name='figure_.pdf'):

        x_det = self.x_det
        x_sto = self.x_sto
        t_k = self.dt * self.tau

        det_vector_cl = x_det[:, 0] + x_det[:, 1]
        sto_vector_cl = x_sto[:, 0] + x_sto[:, 1]

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

        ax_sv.plot(t_k, x_det[:, 0],
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

        ax_iv.plot(t_k, x_det[:, 1],
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
        ax_nv.plot(t_k, det_vector_cl,
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
        ax_sh.plot(t_k, x_det[:, 2],
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
        ax_ih.plot(t_k, x_det[:, 3],
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

        ax_nh.plot(t_k, x_det[:, 2] + x_det[:, 3],
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
        ax_sv.plot(t_k, x_sto[:, 0],
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

        ax_iv.plot(t_k, x_sto[:, 1],
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
        ax_nv.plot(t_k, sto_vector_cl,
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

        ax_sh.plot(t_k, x_sto[:, 2],
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
        ax_ih.plot(t_k, x_sto[:, 3],
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
        ax_nh.plot(t_k, x_sto[:, 2] + x_sto[:, 3],
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
        plt.savefig(file_name)
        plt.show()
