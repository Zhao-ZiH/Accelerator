import matplotlib.pyplot as plt
import numpy as np
from scipy import constants
from scipy import integrate
import warnings
'''
This code can be used for calculating the field of a Gaussian bunch,which can theoretically calculate the field- 
-generated by bunches of any size at any position, but, for extreme parameters, there may be some integration errors.
About the specific method, I calculate the value of the electric field using the scalar potential in the co-moving system,
and then transform it back into the laboratory system.
This code uses the SI units.
'''


class Gaussian_Bunch:
    mass_e = 0.511E6  # [MeV]
    epsilon_0 = 8.854e-12

    def __init__(self, Energy, sigma_x, sigma_y, sigma_z, Number_e):
        self.Energy = Energy
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.sigma_z = sigma_z
        self.Number_e = Number_e
        self.gam = Energy / Gaussian_Bunch.mass_e
        self.beta = (1 - 1 / (self.gam ** 2)) ** (0.5)

    def __E_y_fun(self, q, x, y, z, sigma_x, sigma_y, sigma_z, gam):
        up = np.exp(-x ** 2 / (q + 2 * sigma_x ** 2)) \
             * np.exp(-y ** 2 / (q + 2 * sigma_y ** 2)) \
             * np.exp(-(z * gam) ** 2 / (q + 2 * (sigma_z * gam) ** 2)) \
             * 2 * y * gam / (q + 2 * sigma_y ** 2)

        down = (q + 2 * sigma_x ** 2) ** 0.5 * \
               (q + 2 * sigma_y ** 2) ** 0.5 * \
               (q + 2 * (sigma_z * gam) ** 2) ** 0.5

        return up / down

    def __E_x_fun(self, q, x, y, z, sigma_x, sigma_y, sigma_z, gam):
        up = np.exp(-x ** 2 / (q + 2 * sigma_x ** 2)) \
             * np.exp(-y ** 2 / (q + 2 * sigma_y ** 2)) \
             * np.exp(-(z * gam) ** 2 / (q + 2 * (sigma_z * gam) ** 2)) \
             * 2 * x * gam / (q + 2 * sigma_x ** 2)

        down = (q + 2 * sigma_x ** 2) ** 0.5 * \
               (q + 2 * sigma_y ** 2) ** 0.5 * \
               (q + 2 * (sigma_z * gam) ** 2) ** 0.5

        return up / down

    def __E_z_fun(self, q, x, y, z, sigma_x, sigma_y, sigma_z, gam):
        up = np.exp(- x ** 2 / (q + 2 * sigma_x ** 2)
                    - y ** 2 / (q + 2 * sigma_y ** 2)
                    - (z * gam) ** 2 / (q + 2 * (sigma_z * gam) ** 2)) \
             * 2 * gam * z / (q + 2 * (gam * sigma_z) ** 2)

        down = ((q + 2 * sigma_x ** 2) * \
                (q + 2 * sigma_y ** 2) * \
                (q + 2 * (gam * sigma_z) ** 2)) ** 0.5

        return up / down

    def __Get_q_limit(self, x, y, z):
        x = np.abs(x)
        y = np.abs(y)
        z = np.abs(z)
        if x / self.sigma_x > 50 and y / self.sigma_y < 50 and z / self.sigma_z < 50:
            return 40 * x
        elif x / self.sigma_x < 50 and y / self.sigma_y > 50 and z / self.sigma_z < 50:
            return 40 * y
        elif x / self.sigma_x > 50 and y / self.sigma_y < 50:
            return (1e3 * x) ** 2 * (1e3 * z) ** 2
        elif x / self.sigma_x < 50 and y / self.sigma_y > 50:
            return (1e3 * y) ** 2 * (1e3 * z) ** 2
        else:
            return np.min([1e3 * self.sigma_x ** 2, 1e3 * self.sigma_y ** 2])

    def __Get_q_limit_for_z(self):
        return 60 * self.sigma_z

    def Get_E_z(self, x, y, z):
        q_limit = self.__Get_q_limit_for_z()

        inte_value_z = integrate.quad(self.__E_z_fun, 0, q_limit,
                                      args=(x, y, z, self.sigma_x, self.sigma_y, self.sigma_z, self.gam))

        if np.abs(inte_value_z[1]) > np.abs(0.01 * inte_value_z[0]):
            print('Excessive integration error !')
        else:
            compare_val = integrate.quad(self.__E_z_fun, 0, q_limit * 10,
                                         args=(x, y, z, self.sigma_x, self.sigma_y, self.sigma_z, self.gam))
            if inte_value_z[0] == compare_val[0]:
                self.__E_z = inte_value_z[0] * constants.e * self.Number_e / (np.pi) ** 0.5 * \
                             1 / (4 * np.pi * Gaussian_Bunch.epsilon_0)
            elif np.abs((inte_value_z[0] - compare_val[0]) / inte_value_z[0]) < 0.05:
                self.__E_z = inte_value_z[0] * constants.e * self.Number_e / (np.pi) ** 0.5 * \
                             1 / (4 * np.pi * Gaussian_Bunch.epsilon_0)
            else:
                print('Integration upper limit is too small !')

    @property
    def E_z(self):
        return self.__E_z

    def set_Ez_local(self, x, y, z):
        self.Get_E_z(x, y, z)

    def Get_E_z_derivative_z(self, x, y, z):
        E_z_temp = []
        Z_temp = []
        for z_val in np.linspace(z, z + 0.001 * self.sigma_z, 10):

            q_limit = self.__Get_q_limit_for_z()

            inte_value_z = integrate.quad(self.__E_z_fun, 0, q_limit,
                                          args=(x, y, z_val, self.sigma_x, self.sigma_y, self.sigma_z, self.gam))

            if np.abs(inte_value_z[1]) > np.abs(0.01 * inte_value_z[0]):
                print('Excessive integration error !')

            else:
                compare_val = integrate.quad(self.__E_z_fun, 0, q_limit * 10,
                                             args=(x, y, z, self.sigma_x, self.sigma_y, self.sigma_z, self.gam))
                if inte_value_z[0] == compare_val[0]:
                    E_z_temp.append(inte_value_z[0] * constants.e * self.Number_e / (np.pi) ** 0.5 * \
                                    1 / (4 * np.pi * Gaussian_Bunch.epsilon_0))
                    Z_temp.append(z_val)
                elif np.abs((inte_value_z[0] - compare_val[0]) / inte_value_z[0]) < 0.05:
                    E_z_temp.append(inte_value_z[0] * constants.e * self.Number_e / (np.pi) ** 0.5 * \
                                    1 / (4 * np.pi * Gaussian_Bunch.epsilon_0))
                    Z_temp.append(z_val)
                else:
                    print('Integration upper limit is too small !')

        diff_E_z = np.diff(E_z_temp)
        diff_z = np.diff(Z_temp)
        self.__E_z_derivative_z = np.mean([i / j for i, j in zip(diff_E_z, diff_z)])

    @property
    def E_z_derivative_z(self):
        return self.__E_z_derivative_z

    def set_Ez_derivative_z_local(self, x, y, z):
        self.Get_E_z_derivative_z(x, y, z)

    def Get_E_y(self, x, y, z):
        q_limit = self.__Get_q_limit(x, y, z)
        while True:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", integrate.IntegrationWarning)
                    inte_value_y = integrate.quad(self.__E_y_fun, 0, q_limit,
                                                  args=(x, y, z, self.sigma_x, self.sigma_y, self.sigma_z, self.gam))
                break
            except integrate.IntegrationWarning as e:
                q_limit = q_limit / 5
        while True:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", integrate.IntegrationWarning)
                    compare_val = integrate.quad(self.__E_y_fun, 0, q_limit * 10,
                                                 args=(x, y, z, self.sigma_x, self.sigma_y, self.sigma_z, self.gam))
                break
            except integrate.IntegrationWarning as e:
                q_limit = q_limit / 5
        if inte_value_y[0] == compare_val[0]:
            self.__E_y = inte_value_y[0] * constants.e * self.Number_e / (np.pi) ** 0.5 * \
                         1 / (4 * np.pi * Gaussian_Bunch.epsilon_0)
        elif inte_value_y[0] == 0:
            self.__E_y = inte_value_y[0] * constants.e * self.Number_e / (np.pi) ** 0.5 * \
                         1 / (4 * np.pi * Gaussian_Bunch.epsilon_0)
            print('Please check integration limit')
        else:
            while np.abs((inte_value_y[0] - compare_val[0]) / inte_value_y[0]) > 0.05:
                q_limit = q_limit * 5
                while True:
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("error", integrate.IntegrationWarning)
                            inte_value_y = integrate.quad(self.__E_y_fun, 0, q_limit,
                                                          args=(x, y, z,
                                                                self.sigma_x, self.sigma_y, self.sigma_z, self.gam))
                            compare_val = integrate.quad(self.__E_y_fun, 0, q_limit * 5,
                                                         args=(x, y, z,
                                                               self.sigma_x, self.sigma_y, self.sigma_z, self.gam))
                            break
                    except integrate.IntegrationWarning as e:
                        q_limit = q_limit / 2
            self.__E_y = inte_value_y[0] * constants.e * self.Number_e / (np.pi) ** 0.5 * \
                         1 / (4 * np.pi * Gaussian_Bunch.epsilon_0)

    @property
    def E_y(self):
        return self.__E_y

    def set_Ey_local(self, x, y, z):
        self.Get_E_y(x, y, z)

    def Get_E_y_derivative_y(self, x, y, z):
        E_y_temp = []
        Y_temp = []
        for y_val in np.linspace(y, y + 0.001 * self.sigma_y, 10):
            q_limit = self.__Get_q_limit(x, y_val, z)
            while True:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("error", integrate.IntegrationWarning)
                        inte_value_y = integrate.quad(self.__E_y_fun, 0, q_limit,
                                                      args=(x, y_val, z,
                                                            self.sigma_x, self.sigma_y, self.sigma_z, self.gam))
                    break
                except integrate.IntegrationWarning as e:
                    q_limit = q_limit / 5
            while True:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("error", integrate.IntegrationWarning)
                        compare_val = integrate.quad(self.__E_y_fun, 0, q_limit * 10,
                                                     args=(x, y_val, z,
                                                           self.sigma_x, self.sigma_y, self.sigma_z, self.gam))
                    break
                except integrate.IntegrationWarning as e:
                    q_limit = q_limit / 5
            if inte_value_y[0] == compare_val[0]:
                E_y_temp.append(inte_value_y[0] * constants.e * self.Number_e / (np.pi) ** 0.5 * \
                                1 / (4 * np.pi * Gaussian_Bunch.epsilon_0))
                Y_temp.append(y_val)
            elif inte_value_y[0] == 0:
                E_y_temp.append(inte_value_y[0] * constants.e * self.Number_e / (np.pi) ** 0.5 * \
                                1 / (4 * np.pi * Gaussian_Bunch.epsilon_0))
                Y_temp.append(y_val)
                print('Please check integration limit')
            else:
                while np.abs((inte_value_y[0] - compare_val[0]) / inte_value_y[0]) > 0.05:
                    q_limit = q_limit * 5
                    while True:
                        try:
                            with warnings.catch_warnings():
                                warnings.simplefilter("error", integrate.IntegrationWarning)
                                inte_value_y = integrate.quad(self.__E_y_fun, 0, q_limit,
                                                              args=(x, y_val, z,
                                                                    self.sigma_x, self.sigma_y, self.sigma_z, self.gam))
                                compare_val = integrate.quad(self.__E_y_fun, 0, q_limit * 5,
                                                             args=(x, y_val, z,
                                                                   self.sigma_x, self.sigma_y, self.sigma_z, self.gam))
                                break
                        except integrate.IntegrationWarning as e:
                            q_limit = q_limit / 2
                E_y_temp.append(inte_value_y[0] * constants.e * self.Number_e / (np.pi) ** 0.5 * \
                                1 / (4 * np.pi * Gaussian_Bunch.epsilon_0))
                Y_temp.append(y_val)

        diff_E_y = np.diff(E_y_temp)
        diff_y = np.diff(Y_temp)
        self.__E_y_derivative_y = np.mean([i / j for i, j in zip(diff_E_y, diff_y)])

    @property
    def E_y_derivative_y(self):
        return self.__E_y_derivative_y

    def set_Ey_derivative_y_local(self, x, y, z):
        self.Get_E_y_derivative_y(x, y, z)

    def Get_E_x(self, x, y, z):
        q_limit = self.__Get_q_limit(x, y, z)
        while True:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", integrate.IntegrationWarning)
                    inte_value_x = integrate.quad(self.__E_x_fun, 0, q_limit,
                                                  args=(x, y, z, self.sigma_x, self.sigma_y, self.sigma_z, self.gam))
                break
            except integrate.IntegrationWarning as e:
                q_limit = q_limit / 5
        while True:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", integrate.IntegrationWarning)
                    compare_val = integrate.quad(self.__E_x_fun, 0, q_limit * 10,
                                                 args=(x, y, z, self.sigma_x, self.sigma_y, self.sigma_z, self.gam))
                break
            except integrate.IntegrationWarning as e:
                q_limit = q_limit / 5
        if inte_value_x[0] == compare_val[0]:
            self.__E_x = inte_value_x[0] * constants.e * self.Number_e / (np.pi) ** 0.5 * \
                         1 / (4 * np.pi * Gaussian_Bunch.epsilon_0)
        elif inte_value_x[0] == 0:
            self.__E_x = inte_value_x[0] * constants.e * self.Number_e / (np.pi) ** 0.5 * \
                         1 / (4 * np.pi * Gaussian_Bunch.epsilon_0)
            print('Please check integration limit')
        else:
            while np.abs((inte_value_x[0] - compare_val[0]) / inte_value_x[0]) > 0.05:
                q_limit = q_limit * 5
                while True:
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("error", integrate.IntegrationWarning)
                            inte_value_x = integrate.quad(self.__E_x_fun, 0, q_limit,
                                                          args=(x, y, z,
                                                                self.sigma_x, self.sigma_y, self.sigma_z, self.gam))
                            compare_val = integrate.quad(self.__E_x_fun, 0, q_limit * 5,
                                                         args=(x, y, z,
                                                               self.sigma_x, self.sigma_y, self.sigma_z, self.gam))
                            break
                    except integrate.IntegrationWarning as e:
                        q_limit = q_limit / 2
            self.__E_x = inte_value_x[0] * constants.e * self.Number_e / (np.pi) ** 0.5 * \
                         1 / (4 * np.pi * Gaussian_Bunch.epsilon_0)

    @property
    def E_x(self):
        return self.__E_x

    def set_Ex_local(self, x, y, z):
        self.Get_E_x(x, y, z)

    def Get_E_x_derivative_x(self, x, y, z):
        E_x_temp = []
        X_temp = []
        for x_val in np.linspace(x, x + 0.001 * self.sigma_x, 10):
            q_limit = self.__Get_q_limit(x_val, y, z)
            while True:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("error", integrate.IntegrationWarning)
                        inte_value_x = integrate.quad(self.__E_x_fun, 0, q_limit,
                                                      args=(x_val, y, z,
                                                            self.sigma_x, self.sigma_y, self.sigma_z, self.gam))
                    break
                except integrate.IntegrationWarning as e:
                    q_limit = q_limit / 5
            while True:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("error", integrate.IntegrationWarning)
                        compare_val = integrate.quad(self.__E_x_fun, 0, q_limit * 10,
                                                     args=(x_val, y, z,
                                                           self.sigma_x, self.sigma_y, self.sigma_z, self.gam))
                    break
                except integrate.IntegrationWarning as e:
                    q_limit = q_limit / 5
            if inte_value_x[0] == compare_val[0]:
                E_x_temp.append(inte_value_x[0] * constants.e * self.Number_e / (np.pi) ** 0.5 * \
                                1 / (4 * np.pi * Gaussian_Bunch.epsilon_0))
                X_temp.append(y_val)
            elif inte_value_x[0] == 0:
                E_x_temp.append(inte_value_x[0] * constants.e * self.Number_e / (np.pi) ** 0.5 * \
                                1 / (4 * np.pi * Gaussian_Bunch.epsilon_0))
                X_temp.append(y_val)
                print('Please check integration limit')
            else:
                while np.abs((inte_value_x[0] - compare_val[0]) / inte_value_x[0]) > 0.05:
                    q_limit = q_limit * 5
                    while True:
                        try:
                            with warnings.catch_warnings():
                                warnings.simplefilter("error", integrate.IntegrationWarning)
                                inte_value_x = integrate.quad(self.__E_x_fun, 0, q_limit,
                                                              args=(x_val, y, z,
                                                                    self.sigma_x, self.sigma_y, self.sigma_z, self.gam))
                                compare_val = integrate.quad(self.__E_x_fun, 0, q_limit * 5,
                                                             args=(x_val, y, z,
                                                                   self.sigma_x, self.sigma_y, self.sigma_z, self.gam))
                                break
                        except integrate.IntegrationWarning as e:
                            q_limit = q_limit / 2
                E_x_temp.append(inte_value_x[0] * constants.e * self.Number_e / (np.pi) ** 0.5 * \
                                1 / (4 * np.pi * Gaussian_Bunch.epsilon_0))
                X_temp.append(x_val)
        diff_E_x = np.diff(E_x_temp)
        diff_x = np.diff(X_temp)
        self.__E_x_derivative_x = np.mean([i / j for i, j in zip(diff_E_x, diff_x)])

    @property
    def E_x_derivative_x(self):
        return self.__E_x_derivative_x

    def set_Ex_derivative_x_local(self, x, y, z):
        self.Get_E_x_derivative_x(x, y, z)

    def Get_B_x(self, x, y, z):

        self.set_Ey_local(x, y, z)

        self.__B_x = self.beta / constants.c * self.__E_y

    @property
    def B_x(self):
        return self.__B_x

    def set_Bx_local(self, x, y, z):
        self.Get_B_x(x, y, z)

    def Get_B_x_derivative_y(self, x, y, z):

        self.set_Ey_derivative_y_local(x, y, z)

        self.__B_x_derivative_y = self.beta / constants.c * self.__E_y_derivative_y

    @property
    def B_x_derivative_y(self):
        return self.__B_x_derivative_y

    def set_Bx_derivative_y_local(self, x, y, z):
        self.Get_B_x_derivative_y(x, y, z)

    def Get_B_y(self, x, y, z):

        self.set_Ex_local(x, y, z)

        self.__B_y = -self.beta / constants.c * self.__E_x

    @property
    def B_y(self):
        return self.__B_y

    def set_By_local(self, x, y, z):
        self.Get_B_y(x, y, z)

    def Get_B_y_derivative_x(self, x, y, z):

        self.set_Ex_derivative_x_local(x, y, z)

        self.__B_y_derivative_x = -self.beta / constants.c * self.__E_x_derivative_x

    @property
    def B_y_derivative_x(self):
        return self.__B_y_derivative_x

    def set_By_derivative_x_local(self, x, y, z):
        self.Get_B_y_derivative_x(x, y, z)
