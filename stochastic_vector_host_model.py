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
        Set the parameters and terms of the Stochastic
        Duffin Van der Pol equation.
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
        beta_v = self.beta_v
        lambda_v = self.lambda_v
        lambda_h = self.lambda_h
        mu_h = self.mu_h
        beta_h = self.beta_h
        n_h = self.x_zero[2] + self.x_zero[3]
        sigma_v = self.sigma_v
        sigma_h = self.sigma_h
        n_v = lambda_v / mu_v
        deterministic_r_zero = (beta_v * beta_h * n_v * n_h) / (mu_v * mu_h)
        stochastic_r_zero = deterministic_r_zero - 0.5 * (sigma_v ** 2
                                                          + sigma_h ** 2)
        self.deterministic_r_zero = deterministic_r_zero
        self.stochastic_r_zero = stochastic_r_zero

        aux_1 = np.sqrt(beta_v ** 2 / (2 * mu_v))
        aux_2 = np.sqrt(beta_h ** 2 / (2 * mu_h))
        aux_3 = np.max([sigma_h, sigma_v])
        self.noise_extinction_condition = np.max([aux_1, aux_2])
        self.noise_intensity_test = aux_3 > self.noise_extinction_condition
        self.vector_upper_bound = n_v
        self.host_upper_bound = n_h
        self.r_zer0_det = deterministic_r_zero

        print"\n"
        print "\t noise conditions:\t", aux_1, aux_2
        print "\t noise intensities:\t", self.sigma_v, self.sigma_h
        print "\t vector_upper_bound:\t", self.vector_upper_bound
        print "\t host_upper_bound:\t", self.host_upper_bound
        return np.array([deterministic_r_zero, stochastic_r_zero])

    def set_parameters_stochastic_vector_host_dynamics(self, mu_v, beta_v,
                                                       lambda_v, mu_h, beta_h,
                                                       lambda_h, sigma_v,
                                                       sigma_h, x_zero):
        """
            Set parameters of SDE Duffin Van Der Pol.
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
        lambda_h = self.lambda_h
        mu_h = self.mu_h
        beta_h = self.beta_h

        n_v = self.lambda_v / self.mu_v
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

        i_v = x_in[1]
        i_h = x_in[3]

        x1 = - sigma_v * i_h / n_v
        x3 = - sigma_h * i_v / n_h
        bp = np.zeros([4, 4], dtype=np.float128)
        bp[0, 0] = x1
        bp[2, 2] = x3

        bp = np.dot(b, bp)
        return bp
