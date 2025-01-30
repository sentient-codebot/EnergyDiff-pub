# Guidelines for DPM-Solver Sampler

## Sample

Compute the sample at time `t_end` by DPM-Solver, given the initial `x` at time `t_start`.

We support the following algorithms for both noise prediction model and data prediction model:
- 'singlestep':
	Singlestep DPM-Solver (i.e. "DPM-Solver-fast" in the paper), which combines different orders of singlestep DPM-Solver. 
	We combine all the singlestep solvers with order <= `order` to use up all the function evaluations (steps).
	The total number of function evaluations (NFE) == `steps`.
	Given a fixed NFE == `steps`, the sampling procedure is:
	- If `order` == 1:
		- Denote K = steps. We use K steps of DPM-Solver-1 (i.e. DDIM).
	- If `order` == 2:
		- Denote K = (steps // 2) + (steps % 2). We take K intermediate time steps for sampling.
		- If steps % 2 == 0, we use K steps of singlestep DPM-Solver-2.
		- If steps % 2 == 1, we use (K - 1) steps of singlestep DPM-Solver-2 and 1 step of DPM-Solver-1.
	- If `order` == 3:
		- Denote K = (steps // 3 + 1). We take K intermediate time steps for sampling.
		- If steps % 3 == 0, we use (K - 2) steps of singlestep DPM-Solver-3, and 1 step of singlestep DPM-Solver-2 and 1 step of DPM-Solver-1.
		- If steps % 3 == 1, we use (K - 1) steps of singlestep DPM-Solver-3 and 1 step of DPM-Solver-1.
		- If steps % 3 == 2, we use (K - 1) steps of singlestep DPM-Solver-3 and 1 step of singlestep DPM-Solver-2.
- 'multistep':
	Multistep DPM-Solver with the order of `order`. The total number of function evaluations (NFE) == `steps`.
	We initialize the first `order` values by lower order multistep solvers.
	Given a fixed NFE == `steps`, the sampling procedure is (Denote K = steps):
	- If `order` == 1:
		- We use K steps of DPM-Solver-1 (i.e. DDIM).
	- If `order` == 2:
		- We firstly use 1 step of DPM-Solver-1, then use (K - 1) step of multistep DPM-Solver-2.
	- If `order` == 3:
		- We firstly use 1 step of DPM-Solver-1, then 1 step of multistep DPM-Solver-2, then (K - 2) step of multistep DPM-Solver-3.
- 'singlestep_fixed':
	Fixed order singlestep DPM-Solver (i.e. DPM-Solver-1 or singlestep DPM-Solver-2 or singlestep DPM-Solver-3).
	We use singlestep DPM-Solver-`order` for `order`=1 or 2 or 3, with total [`steps` // `order`] * `order` NFE.
- 'adaptive':
	Adaptive step size DPM-Solver (i.e. "DPM-Solver-12" and "DPM-Solver-23" in the paper).
	We ignore `steps` and use adaptive step size DPM-Solver with a higher order of `order`.
	You can adjust the absolute tolerance `atol` and the relative tolerance `rtol` to balance the computatation costs
	(NFE) and the sample quality.
	- If `order` == 2, we use DPM-Solver-12 which combines DPM-Solver-1 and singlestep DPM-Solver-2.
	- If `order` == 3, we use DPM-Solver-23 which combines singlestep DPM-Solver-2 and singlestep DPM-Solver-3.

---

Some advices for choosing the algorithm:

- For **unconditional sampling** or **guided sampling with small guidance scale** by DPMs:
	Use singlestep DPM-Solver ("DPM-Solver-fast" in the paper) with `order = 3`.
	e.g.
```python
dpm_solver = DPM_Solver(model_fn, noise_schedule, predict_x0=False)
x_sample = dpm_solver.sample(x, steps=steps, t_start=t_start, t_end=t_end, order=3,
				skip_type='time_uniform', method='singlestep')
```
- For **guided sampling with large guidance scale** by DPMs:
	Use multistep DPM-Solver with `predict_x0 = True` and `order = 2`.
	e.g.
```python
dpm_solver = DPM_Solver(model_fn, noise_schedule, predict_x0=True)
x_sample = dpm_solver.sample(x, steps=steps, t_start=t_start, t_end=t_end, order=2,
				skip_type='time_uniform', method='multistep')
```

We support three types of `skip_type`:
- 'logSNR': uniform logSNR for the time steps. **Recommended for low-resolutional images**
- 'time_uniform': uniform time for the time steps. **Recommended for high-resolutional images**.
- 'time_quadratic': quadratic time for the time steps.

---

```python
r"""
Args:

    x: A pytorch tensor. The initial value at time `t_start`
        e.g. if `t_start` == T, then `x` is a sample from the standard normal distribution.
    steps: A `int`. The total number of function evaluations (NFE).
    t_start: A `float`. The starting time of the sampling.
        If `T` is None, we use self.noise_schedule.T (default is 1.0).
    t_end: A `float`. The ending time of the sampling.
        If `t_end` is None, we use 1. / self.noise_schedule.total_N.
        e.g. if total_N == 1000, we have `t_end` == 1e-3.
        For discrete-time DPMs:
            - We recommend `t_end` == 1. / self.noise_schedule.total_N.
        For continuous-time DPMs:
            - We recommend `t_end` == 1e-3 when `steps` <= 15; and `t_end` == 1e-4 when `steps` > 15.
    order: A `int`. The order of DPM-Solver.
    skip_type: A `str`. The type for the spacing of the time steps. 'time_uniform' or 'logSNR' or 'time_quadratic'.
    method: A `str`. The method for sampling. 'singlestep' or 'multistep' or 'singlestep_fixed' or 'adaptive'.
    denoise_to_zero: A `bool`. Whether to denoise to time 0 at the final step.
        Default is `False`. If `denoise_to_zero` is `True`, the total NFE is (`steps` + 1).

        This trick is firstly proposed by DDPM (https://arxiv.org/abs/2006.11239) and
        score_sde (https://arxiv.org/abs/2011.13456). Such trick can improve the FID
        for diffusion models sampling by diffusion SDEs for low-resolutional images
        (such as CIFAR-10). However, we observed that such trick does not matter for
        high-resolutional images. As it needs an additional NFE, we do not recommend
        it for high-resolutional images.
    lower_order_final: A `bool`. Whether to use lower order solvers at the final steps.
        Only valid for `method=multistep` and `steps < 15`. We empirically find that
        this trick is a key to stabilizing the sampling by DPM-Solver with very few steps
        (especially for steps <= 10). So we recommend to set it to be `True`.
    solver_type: A `str`. The taylor expansion type for the solver. `dpm_solver` or `taylor`. We recommend `dpm_solver`.
    atol: A `float`. The absolute tolerance of the adaptive step size solver. Valid when `method` == 'adaptive'.
    rtol: A `float`. The relative tolerance of the adaptive step size solver. Valid when `method` == 'adaptive'.
Returns:
    x_end: A pytorch tensor. The approximated solution at time `t_end`.
"""
```

