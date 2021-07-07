import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

PI = np.pi
log10 = np.log(10)
CONVERT = (1/2.247)*10**7



class Calculator_of_Jfactor:
    def __init__(self , generate_log10_rho_square , log10_r_min , log10_r_max , bins):
        self.generate_log10_rho_square = generate_log10_rho_square
        self.log10_r_min = log10_r_min
        self.log10_r_max = log10_r_max

        self.log10_r = np.linspace(log10_r_min , log10_r_max , bins)
        self.r = 10 ** self.log10_r
        self.rho_square = 10 ** generate_log10_rho_square(self.log10_r) / (self.r ** 3) / (4 * PI * log10)
        self.result_with_r = None
        self.result_with_logr = None


    def calc_Jfactor_with_linspace_of_r(self , D = 10 ** 4 , print_flag = True):
        Rmax = np.max(self.r)
        J_tilde = self.rho_square * self.W(self.r , 0 , self.r , D) * self.step(Rmax-self.r)
        integral_cum = integrate.cumtrapz(J_tilde , self.r)#手法１
        integral_sim = integrate.simps(J_tilde , self.r)#手法２
        if print_flag:
            print("--------------------------------[Integral with linspace of r]------------------------------------")
            print("Result of cum-method = ", 4 * PI * integral_cum[-1])
            print("Result of sim-method = ", 4 * PI * integral_sim)
            print()
        if not print_flag:
            return 4 * PI * integral_cum[-1]


    def calc_Jfactor_with_linspace_of_logr(self , D = 10 ** 4 , print_flag = True):
        Rmax = np.max(self.r)
        J_tilde = self.rho_square * (self.W(self.r , 0 , self.r , D) * self.step(Rmax-self.r))
        integral_cum = integrate.cumtrapz(J_tilde * self.r * log10 , self.log10_r)#手法１
        integral_sim = integrate.simps(J_tilde * self.r * log10, self.log10_r)#手法２
        if print_flag:
            print("--------------------------------[Integral with linspace of log r]--------------------------------")
            print("Result of cum-method = ", 4 * PI * integral_cum[-1])
            print("Result of sim-method = ", 4 * PI * integral_sim)
            print()
        if not print_flag:
            return 4 * PI * integral_cum[-1]

    def calc_Jfactor_with_standard_method(self , D = 10 ** 4 , print_flag = True):
        f = lambda x : self.generate_log10_rho_square(x)
        g = lambda x : (10**f(np.log10(x))) / (x**3) / (4 * PI * log10)
        h = lambda x : g(x) * self.W(x , 0 , x , D)
        result = 4 * PI * (integrate.quad(h , 10 ** self.log10_r_min, 10 ** self.log10_r_max)[0])
        if print_flag:
            print("--------------------------------[Integral with standard method]--------------------------------")
            print("standard result = " , result)
            print()
        if not print_flag:
            return result

    def calc_relative_ratio(self , D = 10*4):
        result=self.calc_Jfactor_with_standard_method(D = D , print_flag = False)
        result_with_r = self.calc_Jfactor_with_linspace_of_r(D = D , print_flag = False)
        result_with_logr = self.calc_Jfactor_with_linspace_of_logr(D = D , print_flag = False)
        print("--------------------------------[ratio]--------------------------------")
        print("Relative error with log r = " , (result_with_logr-result)/result)
        print("Relative error with r = " , (result_with_r-result)/result)

    def W(self, r, s, t, D):
        a1 = np.sqrt(D ** 2 - t ** 2) - np.sqrt(r ** 2 - t ** 2)
        a2 = np.sqrt(D ** 2 - s ** 2) - np.sqrt(r ** 2 - s ** 2)
        return (r / D) * np.log(a1 / a2)

    def step(self, x):
        y = np.array(x)
        y[y>=0] = 1
        y[y<0] = 0

        if type(x) == float:
            return y[0]

        return y
