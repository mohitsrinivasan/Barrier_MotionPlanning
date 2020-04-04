import rps.robotarium as robotarium
from rps.utilities import *
from rps.utilities.barrier_certificates import *
from rps.utilities.controllers import *
from rps.utilities.transformations import *
from matplotlib import patches
import numpy as np
import time
from constraints import Constraints
from cvxopt import matrix
from cvxopt.blas import dot
from cvxopt.solvers import qp

N = 1
initial_conditions = np.transpose(np.array([[-1, -0.8, 0]]))

r = robotarium.Robotarium(number_of_robots=N, show_figure=True, initial_conditions=initial_conditions, sim_in_real_time=False)
si_to_uni_dyn = create_si_to_uni_dynamics_with_backwards_motion()
_, uni_to_si_states = create_si_to_uni_mapping()
si_barrier_cert = create_single_integrator_barrier_certificate_with_boundary()

## Initialize variables
h_t1 = -1
h_t2 = -1
h_b = -1
obj_prefix = 3
count_prefix = 1

## Visualize goals, obstacles and the base
r.axes.add_patch(patches.Ellipse((0.8, 0.5), 0.4, 0.6, 0, fill=False))          # Target 1
r.axes.add_patch(patches.Ellipse((0.8, -0.6), 0.4, 0.6, 0, fill=False))         # Target 2
r.axes.add_patch(patches.Ellipse((0, 0), 0.5, 1, 0, fill=False))                # Obstacle
r.axes.add_patch(patches.Circle((-1, 0.5), 0.2, fill=False, zorder=10))         # Base

## First solve the prefix objective
while (count_prefix <= obj_prefix):
    
    if count_prefix == 1:
        
        c = Constraints(1)
        
        while (h_t1 <= 0):
            
            x = r.get_poses()
            x_si = uni_to_si_states(x)
            A, B, H, f, h_t1 = c.compute(count_prefix, x)
            u = qp(H, f, matrix(A), matrix(B))
            u = si_barrier_cert(np.array(u['x']), x[0:2])
            dxu = si_to_uni_dyn(u, x)
            r.set_velocities(np.arange(N), dxu)
            r.step()
            
    elif count_prefix == 2:
            
        c = Constraints(2)
        
        while (h_t2 <= 0):
            
            x = r.get_poses()
            x_si = uni_to_si_states(x)
            A, B, H, f, h_t2 = c.compute(count_prefix, x)
            u = qp(H, f, matrix(A), matrix(B))
            u = si_barrier_cert(np.array(u['x']), x[0:2])
            dxu = si_to_uni_dyn(u, x)
            r.set_velocities(np.arange(N), dxu)
            r.step()
            
    else:
        
        c = Constraints(3)
        
        while (h_b <= 0):
            
            x = r.get_poses()
            x_si = uni_to_si_states(x)
            A, B, H, f, h_b = c.compute(count_prefix, x)
            u = qp(H, f, matrix(A), matrix(B))
            u = si_barrier_cert(np.array(u['x']), x[0:2])
            dxu = si_to_uni_dyn(u, x)
            r.set_velocities(np.arange(N), dxu)
            r.step()    
    
    count_prefix = count_prefix + 1
    

r.call_at_scripts_end()