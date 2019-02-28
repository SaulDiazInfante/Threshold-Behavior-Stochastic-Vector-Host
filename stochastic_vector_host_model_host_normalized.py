"""
    This class define the
    Threshold behaviour of a stochastic vector host model
    Saul Diaz-Infante, Adrian Acu\~na Zegarra.
    Here we also implements the functions to coding several
    numerical methods.
    \begin{equation}
        \begin{aligned}
            d S_V &=
                \left [
                    \Lambda_V - \beta_V / N_V S_V I_H - \mu_V S_V
                \right ] dt
                - \sigma_V S_V I_H dB_t ^ V,
                \\
            d I_V &=
                \left [
                   \beta_V / N_V S_V I_H - \mu_V I_V
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

import numpy as np
import datetime
import yaml


class StochasticVectorHostDynamics(object):
    """
        Set the parameters and terms of the Stochastic Model.
    """

    def __init__(self, mu_v=2.1, beta_v=.3, lambda_v=189000.0,
                 mu_h=0.0142857, beta_h=.15, lambda_h=1142.856,
                 sigma_v=1.0, sigma_h=1.0, alpha=2.0 / 5.0,
                 x_zero=np.array([1190, 10, 1000, 100])):
        #
        self.mu_v = mu_v
        self.beta_v = beta_v
        self.lambda_v = lambda_v
        self.lambda_h = lambda_h
        self.mu_h = mu_h
        self.beta_h = beta_h
        self.alpha = alpha

        #
        #
        self.r_zero_det = 0.0
        self.r_zero_sto = 0.0
        self.n_h_inf = lambda_h / mu_h
        self.sigma_v = sigma_v
        self.sigma_h = sigma_h
        self.x_zero = x_zero
        self.deterministic_r_zero = 0.0
        self.stochastic_r_zero = 0.0
        self.noise_extinction_condition = 0.0
        self.noise_intensity_test = 0.0
        self.vector_upper_bound = lambda_v / mu_v
        self.host_upper_bound = self.n_h_inf

    def r_zero(self):

        mu_v = self.mu_v
        mu_h = self.mu_h
        beta_v = self.beta_v
        beta_h = self.beta_h
        lambda_v = self.lambda_v
        lambda_h = self.lambda_h
        #
        n_v_inf = lambda_v / mu_v
        # n_h_inf = lambda_h / mu_h
        n_h_inf = self.x_zero[2] + self.x_zero[3]
        sigma_v = self.sigma_v
        sigma_h = self.sigma_h
        a_b = beta_h + mu_h * n_h_inf / (mu_v + beta_v * n_v_inf)
        b_a = a_b ** (-1)
        sigma_aster = (.25 * mu_v * mu_h) ** (-1) * \
                      ((sigma_v * (
                              a_b * n_v_inf / n_h_inf - mu_h / beta_v))
                       ** 2
                       + (sigma_h * (b_a - mu_v / beta_h)) ** 2
                       )

        deterministic_r_zero = (beta_v * beta_h * n_v_inf) / (
                mu_v * mu_h * n_h_inf)
        stochastic_r_zero = deterministic_r_zero - sigma_aster
        self.deterministic_r_zero = deterministic_r_zero
        self.stochastic_r_zero = stochastic_r_zero

        aux_1 = np.sqrt((beta_v * n_v_inf) ** 2 / (2 * mu_v))
        n_v_inf = np.sqrt((beta_h * n_h_inf) ** 2 / (2 * mu_h))
        aux_3 = sigma_v > aux_1 and sigma_h > n_v_inf

        self.noise_extinction_condition = np.max([aux_1, n_v_inf])
        self.noise_intensity_test = aux_3 > self.noise_extinction_condition
        self.vector_upper_bound = n_v_inf
        self.host_upper_bound = n_h_inf

        print "\n\t Extinction by Noise: "
        print '\t----------------------'
        print ('\t R0_D: %2.8f, \t R0_S: %2.8f '
               % (deterministic_r_zero, stochastic_r_zero))
        cond = aux_3
        if cond:
            print "\t(ebn): =)"
            print ('\t\t[sig_v, sig_h, bound_v, bound_h] '
                   '= [%5.8f, %5.8f, %5.8f, %5.8f]'
                   % (sigma_v, sigma_h, aux_1, n_v_inf))
        else:
            print "\t(ebn): =("
            print ('\t\t[sig_v, sig_h, bound_v, bound_h] '
                   '= [%5.4f, %5.4f, %5.4f, %5.4f]'
                   % (sigma_v, sigma_h, aux_1, n_v_inf))

        return np.array([deterministic_r_zero, stochastic_r_zero])

    def extinction_conditions(self):
        mu_v = self.mu_v
        mu_h = self.mu_h
        beta_v = self.beta_v
        beta_h = self.beta_h
        lambda_v = self.lambda_v
        lambda_h = self.lambda_h
        n_v_inf = lambda_v / mu_v
        n_h_inf = self.x_zero[2] + self.x_zero[3]
        c_v = 5.0
        sigma_v = self.sigma_v
        sigma_h = self.sigma_h
        #
        a_b = (beta_h + mu_h * n_h_inf) / (mu_v + beta_v * n_v_inf)
        b_a = a_b ** (-1)
        sigma_aster = 0.5 * (mu_v + mu_h) ** (-1) * \
                      ((sigma_v * (a_b * n_v_inf / n_h_inf - mu_h / beta_v))
                       ** 2
                       + (sigma_h * (b_a - mu_v / beta_h)) ** 2
                       )

        deterministic_r_zero = \
            (beta_v * beta_h * c_v) / (mu_v * mu_h)
        stochastic_r_zero = deterministic_r_zero - sigma_aster
        #
        #

        cond_e1_a = a_b * beta_v * n_v_inf > mu_h * n_h_inf
        cond_e1_b = b_a * beta_h > mu_v
        cond_e1 = cond_e1_a and cond_e1_b

        aux_1 = \
            np.sqrt(2.0 * beta_v ** 2 * n_h_inf /
                    (2 * a_b * beta_v * n_v_inf - mu_h * n_h_inf))
        aux_2 = np.sqrt(
            2.0 * beta_h ** 2 /
            (2 * b_a * beta_h - mu_v))

        aux_3 = (sigma_v <= aux_1) and (sigma_h <= aux_2)

        print "\n\n\n\t Extinction by Noise: "
        print '\t----------------------'
        print ('\t R0_D: %2.8f, \t R0_S: %2.8f '
               % (deterministic_r_zero, stochastic_r_zero))
        noise = (sigma_h ** 2 + sigma_v ** 2)

        cond = aux_3
        if cond:
            print "\t(ebn): =)"
            print ('\t\t[sig_v, sig_h, bound_v, bound_h] '
                   '= [%5.8f, %5.8f, %5.8f, %5.8f]'
                   % (sigma_v, sigma_h, aux_1, aux_2))
        else:
            print "\t(ebn): =("
            print ('\t\t[sig_v, sig_h, bound_v, bound_h] '
                   '= [%5.4f, %5.4f, %5.4f, %5.4f]'
                   % (sigma_v, sigma_h, aux_1, aux_2))

        print"\t----------------------"
        if cond_e1:
            print "\t (E-1): =)"
        else:
            print "\t (E-1): =("

    def set_parameters_stochastic_vector_host_dynamics(self, mu_v, beta_v,
                                                       lambda_v, mu_h, beta_h,
                                                       lambda_h, sigma_v,
                                                       sigma_h, x_zero):
        """
            Set parameters of SDE Model.
        """
        self.mu_v = mu_v
        self.beta_v = beta_v
        self.lambda_v = lambda_v
        self.lambda_h = lambda_h
        self.mu_h = mu_h
        self.beta_h = beta_h

        self.sigma_v = sigma_v
        self.sigma_h = sigma_h
        self.x_zero = x_zero

    def a(self, x_in):
        """
            The drift term of the SDE.
        """
        s_v = x_in[0]
        i_v = x_in[1]
        s_h = x_in[2]
        i_h = x_in[3]

        n_h_inf = s_h + i_h
        mu_v = self.mu_v
        mu_h = self.mu_h
        beta_v = self.beta_v
        lambda_v = self.lambda_v
        self.lambda_h = mu_h * n_h_inf
        beta_h = self.beta_h
        #
        n_h_inf = s_h + i_h
        infection_force_v = beta_v / n_h_inf * s_v * i_h
        infection_force_h = beta_h / n_h_inf * s_h * i_v

        x1 = lambda_v - infection_force_v - mu_v * s_v
        x2 = infection_force_v - mu_v * i_v
        x3 = mu_h * n_h_inf - infection_force_h - mu_h * s_h
        x4 = infection_force_h - mu_h * i_h

        r = np.array([[x1], [x2], [x3], [x4]])
        r = r.reshape(4, )
        return r

    def b(self, x_in):
        """
            The diffusion term.
        """
        # n_h_inf = self.x_zero[2] + self.x_zero[3]
        sigma_v = self.sigma_v
        sigma_h = self.sigma_h

        s_v = x_in[0]
        i_v = x_in[1]
        s_h = x_in[2]
        i_h = x_in[3]

        # n_v_inf = self.lambda_v / self.mu_v
        n_h_inf = self.x_zero[2] + self.x_zero[3]

        x1 = - sigma_v * s_v * i_h / n_h_inf
        x2 = sigma_v * s_v * i_h / n_h_inf
        x3 = - sigma_h * s_h * i_v / n_h_inf
        x4 = sigma_h * s_h * i_v / n_h_inf

        bb = np.zeros([4, 4], dtype=np.float128)
        bb[0, 0] = x1
        bb[1, 1] = x2
        bb[2, 2] = x3
        bb[3, 3] = x4
        return bb

    def b_prime(self, x_in):
        sigma_v = self.sigma_v
        sigma_h = self.sigma_h

        b = self.b(x_in)
        #
        s_v = x_in[0]
        i_v = x_in[1]
        s_h = x_in[2]
        i_h = x_in[3]
        #

        n_h_inf = s_h + i_h

        x1 = - sigma_v * i_h / n_h_inf
        x3 = - sigma_h * i_v / n_h_inf

        bp = np.zeros([4, 4], dtype=np.float128)
        bp[0, 0] = x1
        bp[2, 2] = x3
        bp = np.dot(b, bp)
        return bp

    def save_parameters(self, file_name_prefix='./output/parameters/'):

        # load parameters
        lambda_v = np.float64(self.lambda_v)
        lambda_h = np.float64(self.lambda_h)
        beta_v = np.float64(self.beta_v)
        beta_h = np.float64(self.beta_h)
        mu_v = np.float64(self.mu_v)
        mu_h = np.float64(self.mu_h)
        sigma_v = np.float64(self.sigma_v)
        sigma_h = np.float64(self.sigma_h)
        s_v0 = np.float64(self.x_zero[0])
        i_v0 = np.float64(self.x_zero[1])
        s_h0 = np.float64(self.x_zero[2])
        i_h0 = np.float64(self.x_zero[3])

        n_v_inf = np.float64(self.lambda_v / self.mu_v)
        n_h_inf = np.float64(self.x_zero[2] + self.x_zero[3])
        r_zero_s = np.float64(self.r_zero_sto)
        r_zero_d = np.float64(self.r_zero_det)
        #
        parameters = {
            'lambda_v': lambda_v,
            'lambda_h': lambda_h,
            'beta_v': beta_v,
            'beta_h': beta_h,
            'mu_v': mu_v,
            'mu_h': mu_h,
            'sigma_v': sigma_v,
            'sigma_h': sigma_h,
            's_v0': s_v0,
            'i_v0': i_v0,
            's_h0': s_h0,
            'i_h0': i_h0,
            'n_v_inf': n_v_inf,
            'n_h_inf': n_h_inf,
            'r_zero_s': r_zero_s,
            'r_zero_d': r_zero_d
            }
        #
        str_time = str(datetime.datetime.now())
        file_name = file_name_prefix + str_time + '.yml'
        with open(file_name, 'w') as outfile:
            yaml.dump(parameters, outfile, default_flow_style=False)

    def load_parameters(self, file_name):
        with open(file_name, 'r') as f:
            parameter_data = yaml.load(f)
        # Set initial conditions
        #
        self.lambda_v = np.float64(parameter_data.get('lambda_v'))
        self.lambda_h = np.float64(parameter_data.get('lambda_h'))
        self.beta_v = np.float64(parameter_data.get('beta_v'))
        self.beta_h = np.float64(parameter_data.get('beta_h'))
        self.mu_v = np.float64(parameter_data.get('mu_v'))
        self.mu_h = np.float64(parameter_data.get('mu_h'))
        #
        self.sigma_v = np.float64(parameter_data.get('sigma_v'))
        self.sigma_h = np.float64(parameter_data.get('sigma_h'))
        #
        self.x_zero[0] = np.float64(parameter_data.get('s_v0'))
        self.x_zero[1] = np.float64(parameter_data.get('i_v0'))
        self.x_zero[2] = np.float64(parameter_data.get('s_h0'))
        self.x_zero[3] = np.float64(parameter_data.get('i_v0'))
