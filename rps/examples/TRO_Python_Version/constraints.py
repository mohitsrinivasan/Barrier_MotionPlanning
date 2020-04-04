import numpy as np
from cvxopt import matrix
from cvxopt.blas import dot
from cvxopt.solvers import qp, options
from cvxopt import matrix, sparse
from region import Region

class Constraints:
    
    def __init__(self, ro):
        self.reach_obj = ro
        self.f = matrix(np.array([[0], [0]]), tc='d')
        self.H = matrix(np.array([[1, 0], [0, 1]]), tc='d')
        self.hg = -1
    
    def compute(self, reach_obj, x):

        self.A_g = np.zeros((2, 2))
        self.B_g = np.zeros((2, 1))
        
        self.A_o = np.zeros((2, 2))
        self.B_o = np.zeros((2, 1))
            
        ## Constraints for QP 1
        if reach_obj == 1:
            
            rg = Region(0.2, 0.3, 0.8, 0.5, 1)
            ro = Region(0.25, 0.5, 0, 0, -1)
            self.A_g, self.B_g, self.hg = rg.computeRegionBarrier(x)
            self.A_o, self.B_o, self.ho = ro.computeRegionBarrier(x)       

            
            self.A_qp = np.vstack((self.A_g, self.A_o))
            self.B_qp = np.vstack((self.B_g, self.B_o))

            ## Constraints for QP 2
        elif reach_obj == 2:
            
            
            rg = Region(0.2, 0.3, 0.8, -0.6, 1)
            ro = Region(0.25, 0.5, 0, 0, -1)
            self.A_g, self.B_g, self.hg = rg.computeRegionBarrier(x)
            self.A_o, self.B_o, self.ho = ro.computeRegionBarrier(x)       

            
            self.A_qp = np.vstack((self.A_g, self.A_o))
            self.B_qp = np.vstack((self.B_g, self.B_o))           
            
            ## Constraints for QP 3
        else:
            
            rg = Region(0.2, 0.2, -1, 0.5, 1)
            ro = Region(0.25, 0.5, 0, 0, -1)
            self.A_g, self.B_g, self.hg = rg.computeRegionBarrier(x)
            self.A_o, self.B_o, self.ho = ro.computeRegionBarrier(x)       

            
            self.A_qp = np.vstack((self.A_g, self.A_o))
            self.B_qp = np.vstack((self.B_g, self.B_o))
            
            
        return self.A_qp, self.B_qp, self.H, self.f, self.hg