### EntMooT - Ensemble Tree Model Optimization

from entmoot.optimizer.optimizer import Optimizer
from utils import J_ControlCSTR
import numpy as np
import plotting

def opt_ENTMOOT(f, x_dim, bounds, iter_tot):
    '''
    params: parameters that define the rbf model
    X:      matrix of previous datapoints
    '''

    opt = Optimizer(bounds,
                    base_estimator="ENTING",
                    n_initial_points=int(iter_tot*.2),
                    initial_point_generator="random",
                    acq_func="LCB",
                    acq_optimizer="sampling",
                    random_state=100,
                    model_queue_size=None,
                    base_estimator_kwargs={
                        "lgbm_params": {"min_child_samples": 1}
                    },
                    verbose=False,
                    )

    # run optimizer for 20 iterations
    res = opt.run(f, n_iter=iter_tot)

    return res.x, res.fun, res

iter_tot =  50

# bounds
boundsK = np.array([[0.,10./0.2]]*3 + [[0.,10./15]]*3 + [[Tc_lb-20,Tc_lb+20]])
# plot training data
data_res['Ca_train']    = []; data_res['T_train']     = []
data_res['Tc_train']    = []; data_res['err_train']   = []
data_res['u_mag_train'] = []; data_res['u_cha_train'] = []
data_res['Ks']          = []

start_time = time.time()
KentMoot, f_opt, other_outputs = opt_ENTMOOT(J_ControlCSTR, 7, boundsK, iter_tot)
end_time   = time.time()

print('this optimization took ',end_time - start_time,' (s)')

#evals = np.array(data_res['Ks']).shape[0]
plot_convergence(np.array(data_res['Ks']), None, J_ControlCSTR)

plot_training(data_res,iter_tot)

reps = 10

Ca_eval = np.zeros((data_res['Ca_dat'].shape[0], reps))
T_eval = np.zeros((data_res['T_dat'].shape[0], reps))
Tc_eval = np.zeros((data_res['Tc_dat'].shape[0], reps))

for r_i in range(reps):
  Ca_eval[:,r_i], T_eval[:,r_i], Tc_eval[:,r_i] = J_ControlCSTR(KentMoot,
                                                                collect_training_data=False,
                                                                traj=True)
# Plot the results
plot_simulation(Ca_eval, T_eval, Tc_eval, data_res)
