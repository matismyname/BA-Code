import math
import numpy as np

'Mit dieser Klasse berechnen wir die lambdas bzw. die deltas im Paper von Lui et al..'
class Setup:
    def __init__(self,
        k,
        A,
        Sigma,
        m_x
    ):
        self.k = k
        self.A = A
        self.Sigma = Sigma
        self.m_x = m_x

    def berechne_lambda_h(self):
        return np.trace((np.matmul(self.A, self.Sigma))**self.k)
    
    def berechne_lambda_d(self):
        w = np.matmul(self.A, self.Sigma)**(self.k - 1)
        w = np.matmul(w, self.A)
        w = np.matmul(self.m_x, w)
        w = np.matmul(w, np.transpose(self.m_x))
        return w

    def kte_kum(self):
        h = self.berechne_lambda_h()
        d = self.berechne_lambda_d()
        kappa = 2**(self.k - 1) * math.factorial(self.k - 1) * (h + self.k * d)
        return kappa