import numpy as np
from cvxopt import matrix
from cvxopt.blas import dot
from cvxopt.solvers import qp, options
from cvxopt import matrix, sparse

class Region:
    
    def __init__(self, min_ax, maj_ax, cx, cy, nat):
        
        self.a = min_ax
        self.b = maj_ax
        self.cx = cx
        self.cy = cy
        self.beta = nat
        self.barrier = 1
        
    def computeRegionBarrier(self, x):
        
        self.A = np.zeros((2, 2))
        self.B = np.zeros((2, 1))
        
        self.gamma = 10
        self.P = np.array([[1/(self.a)**2, 0], [0, 1/(self.b)**2]]) 
        self.C = np.array([[self.cx], [self.cy]])
        
        self.temp_goal = np.matmul(np.transpose(x[0:2]-self.C), self.P)    
        self.barrier = self.beta*(1 - np.matmul(self.temp_goal,(x[0:2]-self.C)))
            
        self.A =  np.matmul(self.beta*2*np.transpose(x[0:2]-self.C), self.P)
        
        if self.beta == 1:
            self.B = self.gamma*np.sign(self.barrier)*np.absolute(self.barrier)**(0.4)
        else:
            self.B = self.gamma*self.barrier**3
        
        return self.A, self.B, self.barrier
