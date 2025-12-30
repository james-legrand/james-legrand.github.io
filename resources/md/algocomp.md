# Algocomp

Algocomp is a Python package for simulating competition between pricing algorithms across market structures.

## Install

```sh
$ pip install algocomp
```

## Duopoly Simulation
All duopoly simulations take the following inputs regardless of functional form.

**Mandatory Arguments**
 - **costs:** List of marginal costs for each firm 
 - **qualities** List of product qualities for each firm ($a_i$)
 - **algorithm** Algorithm to use ('pso' or 'q-learning')
 - **seed** Random seed
 - **plot** Boolean flag to plot results
 - **shock** Boolean flag for HILP shock
 
 **Optional Arguments - PSO**
 - **num_particles** Number of particles per firm (Default: 5)
 - **num_iterations** Number of iterations (Default: 1000)
 - **w0** Initial inertia weight (Default: 0.025)
 - **self_confidence** Own particle confidence (Default: 1.75)
 - **social_confidence** Own firm confidence (Default: 1.75)
 - **v_min** Minimum particle velocity (Default: -0.3)
 - **v_max** Maximum particle velocity (Default: 0.3)
 - **memory_size** Size of particle memory in iterations (Default: 5)
 - **hilp_iteration** Iteration when the HILP shock occurs (Default: 500)
 - **shock_size** Price increase (Default: 0.5)

 **Optional Arguments - Q Learning**
 - **max_steps** Number of iterations (Default: 1000000)
 - **grid_lower_bound** Lower bound on price space (Default: Nash - 0.5)
 - **grid_upper_bound** Upper bound on price space (Default: Nash + 0.5)
 - **num_possible_actions** Number of discrete prices (Default: 15)
 - **learning_rate** (Default: 0.15)
 - **discount_factor** (Default: 0.95)
 - **beta** Exploration rate (Default: 0.00001)

### logit_duopoly

Duopoly implementation of logit shares $s_i = \frac{e^{\frac{a_i-p_i}{\mu}}}{e^{\frac{a_0}{\mu}}+\sum_{j=1}^{2}e^{\frac{a_j-p_j}{\mu}}}$

```sh
logit_duopoly(costs, qualities, outside_quality, mu, algorithm, seed, plot, shock, ...)
```
 - **outside_quality** Quality of the outside good ($a_0$)
 - **mu** Horizontal differentiation parameter

 ### hotelling_duopoly

Duopoly implementation of hotelling demand from utility $U_{ij} = a_i - p_i - \theta | x_j - k_i |$

```sh
hotelling_duopoly(costs, qualities, theta, algorithm, seed, plot, shock, ...)
```
 - **theta** Horizontal differentiation parameter

### linear_duopoly

Duopoly implementation of a linear inverse demand function $p_i = a_i - bq_i - dq_j$

```sh
linear_duopoly(costs, qualities, b, d, algorithm, seed, plot, shock, ...)
```
 - **b** Own quantity effect
 - **d** Cross quantity effect


## Multi-firm Simulation
Implementation of the PSO algorithm for logit demand with $N$ firms $s_i = \frac{e^{\frac{a_i-p_i}{\mu}}}{e^{\frac{a_0}{\mu}}+\sum_{j=1}^{N}e^{\frac{a_j-p_j}{\mu}}}$

```sh
logit_pso(costs, qualities, outside_quality, mu, num_firms, shock, seed, plot, ...)
```
The arguments are as in `logit_duopoly()`, with the addition of **num_firms**



 ## Examples

 #### logit_duopoly
```sh
logit_duopoly([1,1], [2,2], 0, 0.25, 'q', 456, True, True, grid_lower_bound = 1.45, grid_upper_bound = 1.95)
```
```s
Output:
Firm 1 Q-learning: 1.7357, Nash: 1.4729
Firm 2 Q-learning: 1.7714, Nash: 1.4729
```
![Duopoly Q learning](../images/algocomp_q.png)

 
 
#### hotelling_duopoly
```sh
hotelling_duopoly([1,1], [2,2], 1, 'pso', 1, False, False)
```
```sh
Output:
Firm 1 PSO: 2.0000, Nash: 2.0000
Firm 2 PSO: 2.0000, Nash: 2.0000
```


#### linear_duopoly
```sh
linear_duopoly([1,2], [1.5,3], 1, 0.85, 'pso', 1, False, True)
```
```sh
Output:
Firm 1 PSO: 0.9355, Nash: 0.9355
Firm 2 PSO: 2.2601, Nash: 2.2601
```

#### logit_pso
```sh
logit_pso([1,1,1,1], [2,2,2,2], 0, 0.01, 4, True, 1, True)
```
```sh
Output:
Firm 1 PSO: 1.2133, Nash: 1.0133
Firm 2 PSO: 1.2133, Nash: 1.0133
Firm 3 PSO: 1.2133, Nash: 1.0133
Firm 4 PSO: 1.2133, Nash: 1.0133
```
![Multi-firm PSO](../images/algocomp_pso.png)


