import rps.robotarium as robotarium
from rps.utilities import *
from rps.utilities.barrier_certificates import *
from rps.utilities.controllers import *
from rps.utilities.transformations import *
from reachGoal import reachGoal
from matplotlib import patches
import numpy as np
import time

N = 1
initial_conditions = np.transpose(np.array([[-1, 0.2, 0]]))

r = robotarium.Robotarium(number_of_robots=N, show_figure=True, initial_conditions=initial_conditions, sim_in_real_time=False)
si_to_uni_dyn = create_si_to_uni_dynamics_with_backwards_motion()
_, uni_to_si_states = create_si_to_uni_mapping()
si_barrier_cert = create_single_integrator_barrier_certificate_with_boundary()

hg = -1

## Visualize goals and obstacles
r.axes.add_patch(patches.Circle((1, 0), 0.2, fill=False, zorder=10))            # Goal region
r.axes.add_patch(patches.Ellipse((0, 0), 0.4, 1.0, 0, fill=False))              # Obstacle

while (hg <= 0):

    x = r.get_poses()
    
    x_si = uni_to_si_states(x)
    
    u, hg = reachGoal(x)
    
    # Create safe control inputs (i.e., no collisions)
    u = si_barrier_cert(u, x[0:2])

    # Transform single integrator velocity commands to unicycle
    dxu = si_to_uni_dyn(u, x)
    
    # Set the velocities by mapping the single-integrator inputs to unciycle inputs
    r.set_velocities(np.arange(N), dxu)
    
    # Iterate the simulation
    r.step()

r.call_at_scripts_end()