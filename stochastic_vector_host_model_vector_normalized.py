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
        self.n_h = lambda_h / mu_h
        self.sigma_v = sigma_v
        self.sigma_h = sigma_h
        self.x_zero = x_zero
        self.deterministic_r_zero = 0.0
        self.stochastic_r_zero = 0.0
        self.noise_extinction_condition = 0.0
        self.noise_intensity_test = 0.0
        self.vector_upper_bound = lambda_v / mu_v
        self.host_upper_bound = self.n_h

    def r_zero(self):

        mu_v = self.mu_v
        mu_h = self.mu_h
        beta_v = self.beta_v
        beta_h = self.beta_h
        lambda_v = self.lambda_v
        lambda_h = self.lambda_h

        n_v = lambda_v / mu_v
        n_h = lambda_h / mu_h
        sigma_v = self.sigma_v
        sigma_h = self.sigma_h
        deterministic_r_zero = (beta_v * beta_h * n_v * n_h) / (mu_v * mu_h)
        stochastic_r_zero = deterministic_r_zero - 0.5 * (sigma_v ** 2
                                                          + sigma_h ** 2)
        self.deterministic_r_zero = deterministic_r_zero
        self.stochastic_r_zero = stochastic_r_zero

        aux_1 = np.sqrt((beta_v * n_v) ** 2 / (2 * mu_v))
        aux_2 = np.sqrt((beta_h * n_h) ** 2 / (2 * mu_h))
        aux_3 = sigma_v > aux_1 and sigma_h > aux_2

        self.noise_extinction_condition = np.max([aux_1, aux_2])
        self.noise_intensity_test = aux_3 > self.noise_extinction_condition
        self.vector_upper_bound = n_v
        self.host_upper_bound = n_h

        print "\n\t Extinction by Noise: "
        print '\t----------------------'
        print ('\t R0_D: %2.8f, \t R0_S: %2.8f '
               % (deterministic_r_zero, stochastic_r_zero))
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

        return np.array([deterministic_r_zero, stochastic_r_zero])

    def extinction_conditions(self):
        mu_v = self.mu_v
        mu_h = self.mu_h
        beta_v = self.beta_v
        beta_h = self.beta_h
        lambda_v = self.lambda_v
        lambda_h = self.lambda_h
        n_v_zero = self.x_zero[0] + self.x_zero[1]
        n_v_inf = lambda_v / mu_v
        n_h_inf = lambda_h / mu_h
        sigma_v = self.sigma_v
        sigma_h = self.sigma_h
        a = (beta_v * n_v_inf + mu_v) / (mu_v * mu_h)
        b = (beta_h * n_h_inf + mu_h) / (mu_v * mu_h)

        # Reproductive numbers
        deterministic_r_zero = (beta_v * beta_h * n_v_inf * n_h_inf) / \
                               (mu_v * mu_h)
        # sigma_aster = (sigma_v ** 2 + sigma_h ** 2) / (mu_v * mu_h)
        """
        sigma_aster = \
            ((1.0 - mu_h / (beta_v * n_v_inf) * sigma_v) ** 2
             +
             (1.0 - mu_h / (beta_v * n_v_inf) * sigma_h) ** 2) \
            / (2.0 * mu_v * mu_h)
        """
        sigma_aster = ((a / b) * sigma_v) ** 2 + ((b / a) * sigma_h) ** 2
        stochastic_r_zero = deterministic_r_zero - 0.5 * sigma_aster
        aux_1 = np.sqrt((beta_v * n_v_inf) ** 2 / (2 * mu_v))
        aux_2 = np.sqrt((beta_h * n_h_inf) ** 2 / (2 * mu_h))
        aux_3 = sigma_v < aux_1 and sigma_h < aux_2

        print "\n\t Extinction by Noise: "
        print '\t----------------------'
        print ('\t R0_D: %2.8f, \t R0_S: %2.8f '
               % (deterministic_r_zero, stochastic_r_zero))
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

        cond_e1_a = beta_v * n_v_inf + mu_v > 1.0
        cond_e1_b = beta_h * n_h_inf + mu_h > 1.0
        cond_e1 = cond_e1_a and cond_e1_b

        min_max = np.min([a, b]) / np.max([a, b])
        cond_e2_a = sigma_v < np.sqrt(min_max * beta_v * n_v_inf)
        cond_e2_b = sigma_h < np.sqrt(min_max * beta_h * n_h_inf)
        cond_e2 = cond_e2_a and cond_e2_b

        cond_e3_a = deterministic_r_zero > 1.0
        cond_e3_b = stochastic_r_zero < 1.0
        cond_e3 = cond_e3_a and cond_e3_b

        #
        print"\n"
        if cond:
            str_cond = '\n\tR0s extinction: ' + '=)'
        else:
            str_cond = '\n\tR0s extinction: ' + '=('
        print str_cond
        print"\t----------------------"
        if cond_e1:
            print "\t (E-1): =)"
            print ('\t\t [beta_v n_v + mu_v, beta_h n_h + mu_h] = ['
                   '%5.8f, %5.8f]'
                   % (beta_v * n_v_inf + mu_v, beta_h * n_h_inf + mu_h))
        else:
            print "\t (E-1): =("
            print ('\t\t [beta_v n_v + mu_v, beta_h n_h + mu_h] = ['
                   '%5.8f, %5.8f]'
                   % (beta_v * n_v_inf + mu_v, beta_h * n_h_inf + mu_h))

        if cond_e2:
            print "\t (E-2): =)"
        else:
            print "\t (E-2): =("
        print ('\t\t (sig_v, sig_h) = (%5.8f, %5.8f)' % (sigma_v, sigma_h))
        print ('\t\t (sig_v_bound, sig_h_bound) = (%5.8f, %5.8f)'
               % (cond_e2_a, cond_e2_b))

        if cond_e3:
            print "\t (E-3): =)"
        else:
            print "\t (E-3): =)"
        print ('\t\tR0D: %5.12f, \t R0S: %5.12f'
               % (deterministic_r_zero, stochastic_r_zero))

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

        n_h = s_h + i_h
        mu_v = self.mu_v
        mu_h = self.mu_h
        beta_v = self.beta_v
        lambda_v = self.lambda_v
        self.lambda_h = mu_h * n_h
        beta_h = self.beta_h

        n_v = s_v + i_v
        n_h = s_h + i_h
        infection_force_v = beta_v / n_v * s_v * i_h
        infection_force_h = beta_h / n_v * s_h * i_v

        x1 = lambda_v - infection_force_v - mu_v * s_v
        x2 = infection_force_v - mu_v * i_v
        x3 = mu_h * n_h - infection_force_h - mu_h * s_h
        x4 = infection_force_h - mu_h * i_h

        r = np.array([[x1], [x2], [x3], [x4]])
        r = r.reshape(4, )
        return r

    def b(self, x_in):
        """
            The diffusion term.
        """
        n_h = self.x_zero[2] + self.x_zero[3]
        sigma_v = self.sigma_v
        sigma_h = self.sigma_h

        s_v = x_in[0]
        i_v = x_in[1]
        s_h = x_in[2]
        i_h = x_in[3]
        n_v = s_v + i_v

        x1 = - sigma_v * s_v * i_h / n_v
        x2 = sigma_v * s_v * i_h / n_v
        x3 = - sigma_h * s_h * i_v / n_v
        x4 = sigma_h * s_h * i_v / n_v

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
        i_h = x_in[3]
        #

        n_v = s_v + i_v

        x1 = - sigma_v * i_h / n_v
        x3 = - sigma_h * i_v / n_v

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

        n_v = np.float64(self.lambda_v / self.mu_v)
        n_h = np.float64(self.x_zero[2] + self.x_zero[3])
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
            'n_v': n_v,
            'n_h': n_h,
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
