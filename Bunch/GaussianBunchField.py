import matplotlib.pyplot as plt
import numpy as np
from scipy import constants
from scipy import integrate


class Gaussian_Bunch:
    mass_e=0.511E6 #[MeV]
    epsilon_0=8.854e-12
    def __init__(self,Energy,sigma_x,sigma_y,sigma_z,Number_e):
        self.Energy=Energy
        self.sigma_x=sigma_x
        self.sigma_y=sigma_y
        self.sigma_z=sigma_z
        self.Number_e=Number_e
        self.gam=Energy/Gaussian_Bunch.mass_e
        self.beta = (1 - 1 / (self.gam ** 2)) ** (0.5)
    def __E_y_fun(self,q, x, y, z, sigma_x, sigma_y, sigma_z, gam):
        up = np.exp(-x ** 2 / (q + 2 * sigma_x ** 2)) \
             * np.exp(-y ** 2 / (q + 2 * sigma_y ** 2)) \
             * np.exp(-(z * gam) ** 2 / (q + 2 * (sigma_z * gam) ** 2)) \
             * 2 * y * gam / (q + 2 * sigma_y ** 2)

        down = (q + 2 * sigma_x ** 2) ** 0.5 * \
               (q + 2 * sigma_y ** 2) ** 0.5 * \
               (q + 2 * (sigma_z * gam) ** 2) ** 0.5

        return up / down

    def __E_x_fun(self,q, x, y, z, sigma_x, sigma_y, sigma_z, gam):
        up = np.exp(-x ** 2 / (q + 2 * sigma_x ** 2)) \
             * np.exp(-y ** 2 / (q + 2 * sigma_y ** 2)) \
             * np.exp(-(z * gam) ** 2 / (q + 2 * (sigma_z * gam) ** 2)) \
             * 2 * x * gam / (q + 2 * sigma_x ** 2)

        down = (q + 2 * sigma_x ** 2) ** 0.5 * \
               (q + 2 * sigma_y ** 2) ** 0.5 * \
               (q + 2 * (sigma_z * gam) ** 2) ** 0.5

        return up / down

    def __Get_q_limit(self,x,y,z):
        if  x/self.sigma_x>100  or y/self.sigma_y>100 or z/self.sigma_z>100:
            return 1

        else:
            return np.max([20*self.sigma_x,20*self.sigma_y])
    def Get_E_y(self,x,y,z):
        q_limit=self.__Get_q_limit(x,y,z)

        inte_value_y=integrate.quad(self.__E_y_fun,0,q_limit,
                               args=(x,y,z,self.sigma_x,self.sigma_y,self.sigma_z,self.gam))

        if inte_value_y[1]>0.01*inte_value_y[0]:
            print('Excessive integration error !')

        else:
            self.__E_y=inte_value_y[0]*constants.e*self.Number_e/(np.pi)**0.5*\
                       1 / (4 * np.pi * Gaussian_Bunch.epsilon_0)
    @property
    def E_y(self):
        return self.__E_y

    def set_Ey_local(self, x, y,z):
        self.Get_E_y(x, y,z)

    def Get_E_y_derivative_y(self,x,y,z):
        E_y_temp=[]
        Y_temp=[]
        for y_val in np.linspace(y,y+0.001*self.sigma_y,10):

            q_limit = self.__Get_q_limit(x,y,z)

            inte_value_y = integrate.quad(self.__E_y_fun, 0, q_limit,
                                          args=(x, y_val, z, self.sigma_x, self.sigma_y, self.sigma_z, self.gam))

            if inte_value_y[1] > 0.01 * inte_value_y[0]:
                print('Excessive integration error !')

            else:
                E_y_temp.append(inte_value_y[0]*constants.e*self.Number_e/(np.pi)**0.5*\
                           1 / (4 * np.pi * Gaussian_Bunch.epsilon_0))
                Y_temp.append(y_val)

        diff_E_y=np.diff(E_y_temp)
        diff_y=np.diff(Y_temp)
        self.__E_y_derivative_y=np.mean([i/j for i,j in zip (diff_E_y,diff_y)])

    @property
    def E_y_derivative_y(self):
        return self.__E_y_derivative_y

    def set_Ey_derivative_y_local(self, x, y,z):
        self.Get_E_y_derivative_y(x, y,z)

    def Get_E_x(self,x,y,z):
        q_limit=self.__Get_q_limit(x,y,z)

        inte_value_x=integrate.quad(self.__E_x_fun,0,q_limit,
                               args=(x,y,z,self.sigma_x,self.sigma_y,self.sigma_z,self.gam))

        if inte_value_x[1]>0.01*inte_value_x[0]:
            print('Excessive integration error !')

        else:
            self.__E_x=inte_value_x[0]*constants.e*self.Number_e/(np.pi)**0.5*\
                       1 / (4 * np.pi * Gaussian_Bunch.epsilon_0)
    @property
    def E_x(self):
        return self.__E_x

    def set_Ex_local(self, x, y,z):
        self.Get_E_x(x, y,z)

    def Get_E_x_derivative_x(self,x,y,z):
        E_x_temp=[]
        X_temp=[]
        for x_val in np.linspace(x,x+0.001*self.sigma_x,10):

            q_limit = self.__Get_q_limit(x,y,z)

            inte_value_x = integrate.quad(self.__E_x_fun, 0, q_limit,
                                          args=(x_val, y, z, self.sigma_x, self.sigma_y, self.sigma_z, self.gam))

            if inte_value_x[1] > 0.01 * inte_value_x[0]:
                print('Excessive integration error !')

            else:
                E_x_temp.append(inte_value_x[0]*constants.e*self.Number_e/(np.pi)**0.5*\
                           1 / (4 * np.pi * Gaussian_Bunch.epsilon_0))
                X_temp.append(x_val)

        diff_E_x=np.diff(E_x_temp)
        diff_x=np.diff(X_temp)
        self.__E_x_derivative_x=np.mean([i/j for i,j in zip (diff_E_x,diff_x)])

    @property
    def E_x_derivative_x(self):
        return self.__E_x_derivative_x

    def set_Ex_derivative_x_local(self, x, y,z):
        self.Get_E_x_derivative_x(x, y,z)

    def Get_B_x(self,x,y,z):

        self.set_Ey_local(x,y,z)

        self.__B_x=self.beta/constants.c*self.__E_y
    @property
    def B_x(self):
        return self.__B_x

    def set_Bx_local(self, x, y, z):
        self.Get_B_x(x, y, z)

    def Get_B_x_derivative_y(self,x,y,z):

        self.set_Ey_derivative_y_local(x,y,z)

        self.__B_x_derivative_y = self.beta/constants.c*self.__E_y_derivative_y
    @property
    def B_x_derivative_y(self):
        return self.__B_x_derivative_y

    def set_Bx_derivative_y_local(self, x, y,z):
        self.Get_B_x_derivative_y(x, y,z)


    def Get_B_y(self,x,y,z):

        self.set_Ex_local(x,y,z)

        self.__B_y=-self.beta/constants.c*self.__E_x
    @property
    def B_y(self):
        return self.__B_y

    def set_By_local(self, x, y,z):
        self.Get_B_y(x, y,z)
    def Get_B_y_derivative_x(self,x,y,z):

        self.set_Ex_derivative_x_local(x,y,z)

        self.__B_y_derivative_x= -self.beta/constants.c*self.__E_x_derivative_x
    @property
    def B_y_derivative_x(self):
        return self.__B_y_derivative_x

    def set_By_derivative_x_local(self, x, y,z):
        self.Get_B_y_derivative_x(x, y, z)