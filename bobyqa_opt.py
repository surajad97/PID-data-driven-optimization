import pybobyqa
from utils import J_ControlCSTR
import numpy as np
import plotting

def opt_PyBOBYQA(f, x_dim, bounds, iter_tot):
    '''
    More info:
    https://numericalalgorithmsgroup.github.io/pybobyqa/build/html/userguide.html#a-simple-example
    '''

    # iterations to find good starting point
    n_rs = 3

    # evaluate first point
    f_best, x_best = Random_search(f, x_dim, bounds, n_rs)
    iter_          = iter_tot - n_rs

    # restructure bounds
    a = bounds[:,0]; b = bounds[:,1]
    pybobyqa_bounds    = (a,b)
    other_outputs      = {}

    soln = pybobyqa.solve(f, x_best, seek_global_minimum=True, 
                          objfun_has_noise=True,
                          user_params = {'restarts.use_restarts':True,
                                         'logging.save_diagnostic_info': True,
                                         'logging.save_xk': True}, 
                          maxfun=iter_, 
                          bounds=pybobyqa_bounds, 
                          rhobeg=0.1)
    
    other_outputs['soln']  = soln
    other_outputs['x_all'] = np.array(soln.diagnostic_info['xk'].tolist())

    return soln.x, f(soln.x), other_outputs

#it is not immidiately clear what the bounds on $K_P,K_I,K_D$ should be, 
#and these are generally set as a comnbination of intuition, prior knowledge, and trial and error.


iter_tot =  50

# bounds
boundsK = np.array([[0.,10./0.2]]*3 + [[0.,10./15]]*3 + [[Tc_lb-20,Tc_lb+20]])

# plot training data
data_res['Ca_train']    = []; data_res['T_train']     = []
data_res['Tc_train']    = []; data_res['err_train']   = []
data_res['u_mag_train'] = []; data_res['u_cha_train'] = []
data_res['Ks']          = []

start_time = time.time()
Kbobyqa, f_opt, other_outputs = opt_PyBOBYQA(J_ControlCSTR, 7, boundsK, iter_tot)
end_time   = time.time()

print('this optimization took ',end_time - start_time,' (s)')

plot_convergence(np.array(data_res['Ks'])[5:], None, J_ControlCSTR)

plot_training(data_res,iter_tot)

reps = 10

Ca_eval = np.zeros((data_res['Ca_dat'].shape[0], reps))
T_eval = np.zeros((data_res['T_dat'].shape[0], reps))
Tc_eval = np.zeros((data_res['Tc_dat'].shape[0], reps))

for r_i in range(reps):
  Ca_eval[:,r_i], T_eval[:,r_i], Tc_eval[:,r_i] = J_ControlCSTR(Kbobyqa,
                                                                collect_training_data=False,
                                                                traj=True)
# Plot the results
plot_simulation(Ca_eval, T_eval, Tc_eval, data_res)
