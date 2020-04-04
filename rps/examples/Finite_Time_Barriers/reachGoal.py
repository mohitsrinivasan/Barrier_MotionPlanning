import numpy as np
from cvxopt import matrix
from cvxopt.blas import dot
from cvxopt.solvers import qp, options
from cvxopt import matrix, sparse


## Function to solve a simple go-to-region problem. The level-set is
## characterized by the finite time barrier function h. The constraint is

def reachGoal(x):
    
    gamma = 10;
    f = matrix(np.array([[0], [0]]), tc='d')
    H = matrix(np.array([[1, 0], [0, 1]]), tc='d')
    P_goal = np.array([[1/(0.2)**2, 0], [0, 1/(0.2)**2]]) 
    P_obs = np.array([[1/(0.2)**2, 0], [0, 1/(0.5)**2]])
    C_goal = np.array([[1], [0]])
    C_obs = np.array([[0], [0]])
    
    temp_goal = np.matmul(np.transpose(x[0:2]-C_goal), P_goal)    
    hg = 1 - np.matmul(temp_goal,(x[0:2]-C_goal))                               # Goal region finite time barrier function
    
    temp_obs = np.matmul(np.transpose(x[0:2]-C_obs), P_obs)
    ho = np.matmul(temp_obs, (x[0:2]-C_obs)) - 1                                         # Obstacle zeroing barrier function
        
    A_goal = np.matmul(2*np.transpose(x[0:2]-C_goal), P_goal)
    B_goal = gamma*np.sign(hg)*np.absolute(hg)**(0.4)
    
    A_obs = np.matmul(-2*np.transpose(x[0:2]-C_obs), P_obs)
    B_obs = gamma*ho**3
    
    A = np.vstack((A_goal, A_obs))
    B = np.vstack((B_goal, B_obs))
    
    u = qp(H, f, matrix(A), matrix(B))
    
    return np.array(u['x']), hg
    