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
import numpy as np


class StochasticVectorHostDynamics(object):
    """
        Set the parameters and terms of the Stochastic Model.
    """

    def __init__(self, mu_v=2.1, beta_v=.3, lambda_v=189000.0,
                 mu_h=0.0142857, beta_h=.15, lambda_h=1142.856,
                 sigma_v=1.0, sigma_h=1.0,
                 x_zero=np.array([1190, 10, 1000, 100])):
        #
        self.mu_v = mu_v
        self.beta_v = beta_v
        self.lambda_v = lambda_v
        self.lambda_h = lambda_h
        self.mu_h = mu_h
        self.beta_h = beta_h

        #
        #

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
        # lambda_h = self.lambda_h
        
        n_v = lambda_v / mu_v
        n_h = self.x_zero[2] + self.x_zero[3]
        sigma_v = self.sigma_v
        sigma_h = self.sigma_h

        deterministic_r_zero = (beta_v * beta_h * n_v * n_h) / (mu_v * mu_h)
        stochastic_r_zero = deterministic_r_zero - 0.5 * (sigma_v ** 2
                                                          + sigma_h ** 2)
        self.deterministic_r_zero = deterministic_r_zero
        self.stochastic_r_zero = stochastic_r_zero

        aux_1 = np.sqrt((beta_v * n_v) ** 2 / (2 * mu_v))
        aux_2 = np.sqrt((beta_h * n_h) ** 2 / (2 * mu_h))
        aux_3 = np.min([sigma_h, sigma_v])
        
        self.noise_extinction_condition = np.max([aux_1, aux_2])
        self.noise_intensity_test = aux_3 > self.noise_extinction_condition
        self.vector_upper_bound = n_v
        self.host_upper_bound = n_h

        print '==============================================================='
        print ('\t R0_D: %2.8f, \t R0_S: %2.8f '
               % (deterministic_r_zero, stochastic_r_zero))
        print "\t Extinction by Noise: "
        cond = np.max([aux_1, aux_2]) < aux_3
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
        # lambda_h = self.lambda_h
        n_v = lambda_v / mu_v
        n_h = self.x_zero[2] + self.x_zero[3]
        sigma_v = self.sigma_v
        sigma_h = self.sigma_h

        # Reproductive numbers
        deterministic_r_zero = (beta_v * beta_h * n_v * n_h) / (mu_v * mu_h)

        stochastic_r_zero = deterministic_r_zero - 0.5 * (sigma_v ** 2
                                                          + sigma_h ** 2)

        r_zero_minus_one = (1.0 - deterministic_r_zero)
        n = -3.35
        den_x = r_zero_minus_one * 2.0 ** (-n) + beta_v * n_v + beta_h * n_h
        num_x = mu_v * mu_h
        x = num_x / den_x
        y = x * 2.0 ** (-n)
        cond_e1 = (mu_h <= x and mu_v <= y) or (mu_v <= y or mu_v <= x)
        sigma_v_bound = np.sqrt((y / x) * beta_v * n_v)
        sigma_h_bound = np.sqrt((y / x) * beta_h * n_h)
        cond_e2 = (sigma_v <= sigma_v_bound) and (sigma_h <= sigma_h_bound)
        cond_e3 = (stochastic_r_zero < 1.0)
        cond = (cond_e1 and cond_e2) and cond_e3

        print"\n ============================================================="
        if cond:
            str_cond = '\tR0s extinction: ' + '=)'
        else:
            str_cond = '\tR0s extinction: ' + '=('
        print str_cond

        if cond_e1:
            print "\t (E-1): =)"
            print ('\t\t [x, y, mu_v, mu_h]= [%5.4f, %5.4f, %5.4f, %5.4f]'
                   % (x, y, mu_v, mu_h))
        else:
            print "\t (E-1): =("
            print ('\t\t [x, y, mu_v, mu_h]= [%5.4f, %5.4f, %5.4f, %5.4f]'
                   % (x, y, mu_v, mu_h))
        if cond_e2:
            print "\t (E-2): =)"
            print ('\t\t (sig_v, sig_h) = (%5.4f, %5.4f)'
                   % (sigma_v, sigma_h))
            print ('\t\t (sig_v_bound, sig_h_bound) = (%5.4f, %5.4f)'
                   % (sigma_v_bound, sigma_h_bound))
        else:
            print "\t (E-2): =("
            print ('\t\t (sig_v, sig_h) = (%5.4f, %5.4f)'
                   % (sigma_v, sigma_h))
            print ('\t\t (sig_v_bound, sig_h_bound) = (%5.4f, %5.4f)'
                   % (sigma_v_bound, sigma_h_bound))
        if cond_e3:
            print "\t (E-3): =)"
            print ('\t\tR0D: %5.4f, \t R0S: %5.4f'
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

        mu_v = self.mu_v
        beta_v = self.beta_v
        lambda_v = self.lambda_v
        # lambda_h = self.lambda_h
        mu_h = self.mu_h
        beta_h = self.beta_h

        # n_v = self.lambda_v / self.mu_v
        n_h = s_h + i_h

        x1 = lambda_v - beta_v * s_v * i_h - mu_v * s_v
        x2 = beta_v * s_v * i_h - mu_v * i_v
        x3 = mu_h * n_h - beta_h * s_h * i_v - mu_h * s_h
        x4 = beta_h * s_h * i_v - mu_h * i_h

        r = np.array([[x1], [x2], [x3], [x4]])
        r = r.reshape(4, )
        return r

    def b(self, x_in):
        """
            The diffusion term.
        """
        n_v = self.lambda_v / self.mu_v
        n_h = self.x_zero[2] + self.x_zero[3]
        sigma_v = self.sigma_v
        sigma_h = self.sigma_h

        s_v = x_in[0]
        i_v = x_in[1]
        s_h = x_in[2]
        i_h = x_in[3]

        x1 = - sigma_v * s_v * i_h / n_v
        x2 = sigma_v * s_v * i_h / n_v
        x3 = - sigma_h * s_h * i_v / n_h
        x4 = sigma_h * s_h * i_v / n_h

        bb = np.zeros([4, 4], dtype=np.float128)
        bb[0, 0] = x1
        bb[1, 1] = x2
        bb[2, 2] = x3
        bb[3, 3] = x4
        return bb

    def b_prime(self, x_in):
        sigma_v = self.sigma_v
        sigma_h = self.sigma_h
        n_v = self.lambda_v / self.mu_v
        n_h = self.x_zero[2] + self.x_zero[3]
        b = self.b(x_in)
        #
        i_v = x_in[1]
        i_h = x_in[3]
        #
        x1 = - sigma_v * i_h / n_v
        x3 = - sigma_h * i_v / n_h
        bp = np.zeros([4, 4], dtype=np.float128)
        bp[0, 0] = x1
        bp[2, 2] = x3
        bp = np.dot(b, bp)

        return bp
