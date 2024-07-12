# PID tuning via data-driven optimization

### Data-driven optimization

This project **data-driven optimization** algorithms to the class of methods that use only function evaluations to optimize an unknown function. This is also referred to as [derivative-free](https://arxiv.org/abs/1904.11585), simulation-based, zeroth-order, and gradient-free optimization by other communities. 

Consider the optimization problem of the following form 

$$ \min_{ {\bf x} \in X} \quad  f({\bf x}) $$

The vector ${\bf x} = [x_1,..., x_n]^T$ is the optimization variable of the problem, the function $f : \mathbb{R}^n \to \mathbb{R} $ is the objective function.

Data-driven optimization algorithms assume:


*   Derivative information of $f({\bf x})$ is unavailable
*   It is only posible to sample $f$ for values of ${\bf x}$
*   $f({\bf x})$ is called a black-box function, given that we can only see the input (${\bf x}_i$) and output $(f({\bf x}_i))$, but we do not know the explict closed-form of $f$

For *expensive* (in terms of time, cost, or other metric) black-box functions, *model-based* data-driven optimization algorithms seem to offer particularly good performance. 

The **general idea** of model-based (also called *surrogate based*) algorithms is to sample the objective function and create a *surrogate* function $\hat{f}_{\mathcal{S}}$ which can be optimized easily. After optimizing $\hat{f}_{\mathcal{S}}$, the "true" objective function $f$ is sampled at the optimal location found by the surrogate. With this new datapoint, the surrogate function $\hat{f}_{\mathcal{S}}$ is refined with this new datapoint, and then optimized again. This is done iteratively until a covergence criterion is achieved.

In this specific notebook tutorial we have included 3 different state-of-the-art data-driven optimization packages, each using a different surrogate function 

*   (Py)BOBYQA
> The name BOBYQA is an acronym for **B**ound **O**ptimization **BY** **Q**uadratic **A**pproximation. BOBYQA is a type of trust-region method, and the choice of surrogate is a quadractic approximation to $f$. More details can be found in [Py-BOBYQA](https://dl.acm.org/doi/10.1145/3338517) and [BOBYQA](https://optimization-online.org/2010/05/2616/).

*   GPyOpt
> GPyOpt is a Python open-source library for Bayesian Optimization developed by the Machine Learning group of the University of Sheffield. It is based on GPy, a Python framework for Gaussian process modelling. More information can be found on their [webpage](https://sheffieldml.github.io/GPyOpt/firstexamples/index.html).
 
*   EntMoot
> ENTMOOT (**EN**semble **T**ree **MO**del **O**ptimization Tool) is a framework to handle tree-based surrogate models in Bayesian optimization applications. Gradient-boosted tree models from [LightGBM](https://lightgbm.readthedocs.io/en/v3.3.2/) are combined with a distance-based uncertainty measure in a deterministic global optimization framework to optimize black-box functions. More details on the method can be found on the [paper](https://arxiv.org/abs/2003.04774) or the [GitHub repository](https://github.com/cog-imperial/entmoot)

A comparative study can be found in: [Data-driven optimization for process systems engineering applications](https://www.sciencedirect.com/science/article/pii/S0009250921007004)

### PID controller

A proportional–integral–derivative controller ([PID controller](https://en.wikipedia.org/wiki/PID_controller)) is a control loop mechanism employing feedback that is used in industrial control systems. A PID controller calculates an error value $e(k)$ at time-step $k$ as the difference between a desired setpoint (SP) and a measured process variable (PV) and applies a correction based on proportional, integral, and derivative terms (denoted P, I, and D), hence the name. The control action is calculated as:

$$u(k)=K_P~e(k)+K_I~\sum^{i=k}_{i=0}e(i)+K_D~\frac{e(k)-e(k-1)}{\Delta t}$$

where $K_P,K_I,K_D$ are parameters to be tuned.

Traditionally, methods exist to tune such parameters, however, treating the problem as a (expensive) black-box optimization problem is an efficient solution method. 

We can formulate the discrete-time control of a chemical process by a PID controller as:

$$
\begin{aligned}
\min_{K_P,K_I,K_D} \quad & \sum_{k=0}^{k=T_f} (e(k))^2\\
\text{s.t.} \quad & x(k+1) = f(x(k),u(k)), \quad k=0,...,T_f-1 \\
& u(k)=K_P~e(k)+K_I~\sum^{i=k}_{i=0}e(i)+K_D~\frac{e(k)-e(k-1)}{\Delta t}, \quad k=0,...,T_f-1\\

## CSTR model

The system used in this tutorial notebook is a Continuous Stirred Tank Reactor (CSTR) described by the following equations

$$\frac{d\text{Ca}}{dt}  = (\text{Ca}_f - \text{Ca})q/V - r_A$$

$$\frac{dT}{dt}   = \frac{q (T_f - T)}{V} + \frac{\Delta H}{(\rho ~C_p)}r_A + \frac{U_A}{(\rho~ V~  Cp)}(Tc-T)$$

with $r_A = k_0 \exp^{(-E/(RT))}\text{Ca}$.

The reaction taking place in the reactor is 

$$ A \rightarrow B $$

The reactor has a cooling jacket with temperature $T_c$ that acts as the input of the system. The states are the temperature of the reactor $T$ and the concentration $C_a$.

Details of the nomenclature for this system can be found in the code below. 


& x(0)=x_0 \quad \text{given}
\end{aligned}
$$

where $e(k)=x_{SP}-x(k)$. Notice that the above optimization problem has only 3 degrees of freedom, $K_P,K_I,K_D$. Notice also that $u(k)$ is a function of $x(k)$, $u(x(k))$.

## PID tuning of CSTR controller ➿

Now, let's use some data-driven optimization algorithms to tune the gains for the proportional-integral-derivative (PID) controllers. Here, we address the tuning of PID controllers as a black-box optimization problem.

The optimization is as follows

**PID tuning Algorithm**

*Initialization*

Collect $d$ initial datapoints $\mathcal{D}=\{(\hat{f}^{(j)}=\sum_{k=0}^{k=T_f} (e(k))^2,~K_P^{(j)},K_I^{(j)},K_D^{(j)}) \}_{j=0}^{j=d}$ by simulating $x(k+1) = f(x(\cdot),u(\cdot))$ for different values of $K_P,K_I,K_D$

*Main loop*

1. *Repeat*
2. $~~~~~~$ Build the surrogate model $\hat{f}_\mathcal{S}(K_P,K_I,K_D)$.
3. $~~~~~~$ Optimize the surrogate $K_P^*,K_I^*,K_D^* = \arg \min_{K_P,K_I,K_D} \hat{f}_\mathcal{S}(K_P,K_I,K_D)$

3. $~~~~~~$ Simulate new values  $ x(k+1) = f(x(k),u(K_P^*,K_I^*,K_D^*;x(k))), ~ k=0,...,T_f-1 $
4. $~~~~~~$ Compute $\hat{f}^{(j+1)}=\sum_{k=0}^{k=T_f} (e(k))^2$.
5. $~~~~~~$ Update: $ \mathcal{D} \leftarrow \mathcal{D}+\{(\hat{f}^{(j+1)},~K_P^*,K_I^*,K_D^*) \}$
6. until stopping criterion is met.

Remarks: 
* The initial collection of $d$ points is generally done by some space filling (e.g. [Latin Hypercube](https://en.wikipedia.org/wiki/Latin_hypercube_sampling), [Sobol Sequence](https://en.wikipedia.org/wiki/Sobol_sequence)) procedure.
* Step two is generally done by some sort of least squares minimization $\min_{\hat{f}_\mathcal{S}}\sum_{j=0}^d(\hat{f}^{(j)}-\hat{f}_\mathcal{S}(K_P^{(j)},K_I^{(j)},K_D^{(j)}))^2$ 
* In Step 4 it is common not to optimize $\hat{f}_\mathcal{S}$ directly, but some adquisition function, for example, the Upper Confidence Bound in Bayesian optimization, where the mean, and some notion of the uncertainty are combined into an objective function that also explores the space. 

The example of the CSTR here is slightly more interesting, in that it has 2 state variables as set points, but the overall procedure followed is the same.

