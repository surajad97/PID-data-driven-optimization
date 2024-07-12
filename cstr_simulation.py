"""## CSTR simulation

Use CSTR model to simulate the operation under some aleatoric conditions.

First, let's define the initial conditions and create a dictionary where to store the information of the simulation
"""

from cstr_model import simulate_CSTR
from plotting import plot_simulation
import imports

data_res = {}
# Initial conditions for the states
x0             = np.zeros(2)
x0[0]          = 0.87725294608097
x0[1]          = 324.475443431599
data_res['x0'] = x0

"""let's now define the time interval of the process and create some storing arrays for plotting"""

# Time interval (min)
n             = 101 # number of intervals
tp            = 25 # process time (min)
t             = np.linspace(0,tp,n)
data_res['t'] = t
data_res['n'] = n

# Store results for plotting
Ca = np.zeros(len(t));      Ca[0]  = x0[0]
T  = np.zeros(len(t));      T[0]   = x0[1]
Tc = np.zeros(len(t)-1);

data_res['Ca_dat'] = copy.deepcopy(Ca)
data_res['T_dat']  = copy.deepcopy(T)
data_res['Tc_dat'] = copy.deepcopy(Tc)

"""we will assume some noise level of the measurements"""

# noise level
noise             = 0.1
data_res['noise'] = noise

"""and define lower and upper bounds on the input"""

# control upper and lower bounds
data_res['Tc_ub']  = 305
data_res['Tc_lb']  = 295
Tc_ub              = data_res['Tc_ub']
Tc_lb              = data_res['Tc_lb']

"""let's define the desired setpoints"""

# desired setpoints
n_1                = int(n/2)
n_2                = n - n_1
Ca_des             = [0.8 for i in range(n_1)] + [0.9 for i in range(n_2)]
T_des              = [330 for i in range(n_1)] + [320 for i in range(n_2)]
data_res['Ca_des'] = Ca_des
data_res['T_des']  = T_des

"""The below plot shows the reactor simulated for some aleatory input. We assume
process disturbances, and therefore simulate 10 realizations and plot the
median along with the interval for all realizations.
"""

# Step cooling temperature to 295
u_example          = np.zeros((1,n-1))
u_example[0,:30]   = 302.0
u_example[0,30:60] = 295.0
u_example[0,60:]   = 299.0

# Simulation
Ca_dat, T_dat, Tc_dat = simulate_CSTR(u_example, data_res, 10)

# Plot the results
plot_simulation(Ca_dat, T_dat, Tc_dat, data_res)