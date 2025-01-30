import torch
import torch.distributions as dist
import matplotlib.pyplot as plt
import torch.nn as nn
import pandas as pd
import numpy as np

def batch_trace(A: torch.Tensor, keepdim=True) -> torch.Tensor:
    """return the trace of a batch of matrices
    
    Arguments:
        A {torch.Tensor} -- shape: (..., M, M)
    
    Returns:
        torch.Tensor -- shape: (..., 1, 1) if keepdim is True, otherwise (..., 1)
    """
    out = A.diagonal(dim1=-2, dim2=-1).sum(dim=-1, keepdim=True)
    if keepdim:
        return out.unsqueeze(-1)
    else:
        return out

class GaussianMixtureModel(torch.nn.Module):
    """GMM Model with a Pytorch Implementation
    
    Algorithm: Variational EM

    Arguments:
        num_variables {int} -- Number of variables in the dataset
        num_mixtures {int} -- Number of mixtures to use in the model
    """
    def __init__(self, 
        num_variables: int, 
        num_mixtures: int,
        hyperparameters: dict[str, torch.Tensor]|None = None
    ) -> None:
        super().__init__()
        self.num_variables = num_variables
        self.num_mixtures = num_mixtures
        if hyperparameters is None:
            self._set_default_hyperparameters()
        else:
            self.hyperparameters = hyperparameters

        self._set_prior_dist()
        self._init_params()
        
        self.register_forward_hook(self.fw_hook_isnan)
        
        self.epsilon = 1e-6
        
    def _set_default_hyperparameters(self,) -> None:
        "set the default hyperparameters if not specified"
        # init_pi = torch.rand(self.num_mixtures).reshape(-1,)
        # init_pi = init_pi / init_pi.sum()
        init_alpha = torch.rand(self.num_mixtures).reshape(-1,)
        init_alpha = init_alpha / init_alpha.sum()
        self.hyperparameters = {
            "alpha_0": init_alpha,                                  # shape: (num_mixtures,)
            # "pi": init_pi,                                          # shape: (num_mixtures,)
            "W_0": torch.diag(torch.ones(self.num_variables)),      # shape: (num_variables, num_variables)
            "nu_0": torch.tensor(self.num_variables + 1).reshape(1,),   # shape: (1,)
            "m_0": torch.zeros(self.num_variables),                 # shape: (num_variables,)
            "beta_0": torch.tensor(1.0,).reshape(1,)                # shape: (1,)
        }
            
    def _set_prior_dist(self,):
        "set the hyperparameters of the prior distributions"
        self.alpha_0 = self.hyperparameters["alpha_0"].reshape(1, -1, 1, 1) # just an initialization, this will be optimized.
        # self.pi = self.hyperparameters["pi"].reshape(1, -1, 1, 1) # just an initialization, this will be optimized. 
        self.W_0 = self.hyperparameters["W_0"].reshape(1, 1, self.num_variables, self.num_variables)
        self.nu_0 = self.hyperparameters["nu_0"].reshape(1, 1, 1, 1).float()
        self.m_0 = self.hyperparameters["m_0"].reshape(1, 1, self.num_variables, 1)
        self.beta_0 = self.hyperparameters["beta_0"].reshape(1, 1, 1, 1)
        
    def _init_params(self,) -> None:
        """initialize the iteration parameters (including intermediate parameters)
        
        primary parameters:
            - r_nk: responsibility of each data point to each mixture                       shape: (num_samples, num_mixtures, 1, 1)
            - alpha_k: concentration of each mixture                                        shape: (1, num_mixtures, 1, 1)
            - m_k: mean of the mean of each mixture                                         shape: (1, num_mixtures, num_variables, 1)
            - beta_k: precision of the mean of each mixture                                 shape: (1, num_mixtures, 1, 1)  
            - W_k: covariance matrix of the covariance of each mixture                      shape: (1, num_mixtures, num_variables, num_variables)
            - nu_k: degrees of freedom of the covariance of each mixture                    shape: (1, num_mixtures, 1, 1)
            
        intermediate parameters:
            - E_ln_Lambda_k: expectation of the log of the precision matrix of each mixture shape: (1, num_mixtures, 1, 1)
            - E_whitened_error_k: expectation of the whitened error                         shape: (num_samples, num_mixtures, 1, 1)
            - N_k: effective number of points in each mixture                               shape: (1, num_mixtures, 1, 1)
            - S_k: sum of squares of each mixture                                           shape: (1, num_mixtures, num_variables, num_variables)
            - x_k_bar: sample mean of each mixture                                          shape: (1, num_mixtures, num_variables, 1)
        """
        self.alpha_k_real = nn.Parameter(self.alpha_0.expand(-1,self.num_mixtures,-1,-1).clone())
            # real-valued alpha, shape: (1, num_mixtures, 1, 1)
        self.m_k_real = nn.Parameter(self.m_0.expand(-1,self.num_mixtures,-1,-1).clone())        # shape: (1, num_mixtures, num_variables, 1)
        self.beta_k_real = nn.Parameter(self.beta_0.expand(-1,self.num_mixtures,-1,-1).sqrt().clone())  
            # real-valued beta, shape: (1, num_mixtures, 1, 1)
        self.L_k_real = nn.Parameter(
            torch.linalg.cholesky(
                self.W_0.expand(-1,self.num_mixtures,-1,-1).clone()
        ))        # shape: (1, num_mixtures, num_variables, num_variables)
        self.nu_k_real = nn.Parameter((self.nu_0.expand(-1,self.num_mixtures,-1,-1)-self.num_variables).sqrt().clone())      
            # nu_k = floor(nu_k_real.square() + self.num_variables), shape: (1, num_mixtures, 1, 1)
        
    # TODO consider using torch.nn.utils.parametrize.register_parametrization(module, tensor_name, parametrization, *, unsafe=False)
    @property
    def alpha_k(self,) -> torch.Tensor:
        return torch.relu(self.alpha_k_real)    # positive real numbers
    
    # this is only accessed with no_grad() context
    @alpha_k.setter
    def alpha_k(self, value: torch.Tensor):
        self.alpha_k_real.data = value
    
    @property
    def m_k(self) -> torch.Tensor:
        return self.m_k_real                    # real numbers
    
    # this is only accessed with no_grad() context
    @m_k.setter
    def m_k(self, value: torch.Tensor):
        self.m_k_real.data = value
    
    @property
    def beta_k(self,) -> torch.Tensor:
        return self.beta_k_real.square()        # positive real numbers
    
    # this is only accessed with no_grad() context
    @beta_k.setter
    def beta_k(self, value: torch.Tensor):
        self.beta_k_real.data = value.sqrt()
        
    @property
    def W_k(self,) -> torch.Tensor:
        return self.L_k_real @ self.L_k_real.transpose(-1,-2)   # positive definite matrices, Cholesky decomposition, L_k is lower triangular
    
    # this is only accessed with no_grad() context
    @W_k.setter
    def W_k(self, value: torch.Tensor):
        self.L_k_real.data = torch.linalg.cholesky(value + self.epsilon*torch.eye(self.num_variables, device=value.device)).expand_as(value).clone()
    
    @property
    def nu_k(self,) -> torch.Tensor:
        return torch.floor(self.nu_k_real.square() + self.num_variables) # positive integers >= num_variables
    
    # this is only accessed with no_grad() context
    @nu_k.setter
    def nu_k(self, value: torch.Tensor):
        self.nu_k_real.data = (value-self.num_variables).sqrt()
    
    def _init_iteration(self, random=True) -> None:
        """reset the iteration parameters into the initial state
        
        """
        self.alpha_k.data = self.alpha_0.expand(-1,self.num_mixtures,-1,-1).clone()         # shape: (1, num_mixtures, 1, 1)
        # self.r_nk = self.pi.clone()                                 # shape: (1, num_mixtures, 1, 1)    -> (num_samples, num_mixtures, 1, 1)
        self.m_k.data = self.m_0.expand(-1,self.num_mixtures,-1,-1).clone()                 # shape: (1, num_mixtures, num_variables, 1)
        self.beta_k.data = self.beta_0.expand(-1,self.num_mixtures,-1,-1).clone()           # shape: (1, num_mixtures, 1, 1)
        self.W_k.data = self.W_0.expand(-1,self.num_mixtures,-1,-1).clone()                 # shape: (1, num_mixtures, num_variables, num_variables)
        self.nu_k.data = self.nu_0.expand(-1,self.num_mixtures,-1,-1).clone()               # shape: (1, num_mixtures, 1, 1)
        
        if random:
            self.alpha_k.data += torch.rand_like(self.alpha_k.data)
            self.m_k.data += torch.rand_like(self.m_k.data)
            self.beta_k.data += torch.rand_like(self.beta_k.data)
            self.W_k.data += torch.rand_like(self.W_k.data) @ torch.rand_like(self.W_k.data).transpose(-1,-2)
            self.nu_k.data += torch.rand_like(self.nu_k.data)
        
        self.E_step_count = 0
        self.M_step_count = 0
        
    def _Dir_ln_C_func(self, alpha: torch.Tensor) -> torch.Tensor:
        "return shape: (1, 1, 1, 1)"
        out = torch.lgamma(self.epsilon+alpha.sum(dim=1, keepdim=True)) - torch.lgamma(alpha+self.epsilon).sum(dim=1, keepdim=True) # shape: (1, 1, 1, 1)
        return out
    
    def _Wishart_ln_B_func(self, W, nu):
        "return shape: (1, k or 1, 1, 1)"
        i_vec = torch.arange(start=1, end=self.num_variables+1, device=W.device).reshape(1,1,-1,1) # shape: (1, 1, num_variables, 1)
        out = (-nu/2) * torch.logdet(W+1e-6).reshape(1,-1,1,1) \
            - nu*self.num_variables/2*torch.log(torch.tensor(2., device=W.device)) \
            - self.num_variables*(self.num_variables-1)/4*torch.log(torch.tensor(torch.pi, device=W.device)) \
            - torch.lgamma((nu+1-i_vec)/2).sum(dim=2, keepdim=True) # shape: (1, 1, 1, 1)
        return out
    
    def _calc_E_ln_pi_k(self,) -> torch.Tensor:
        "return shape: (1, num_mixtures, 1, 1)"
        out = torch.digamma(self.alpha_k+self.epsilon) - torch.digamma(self.alpha_k.sum(dim=1, keepdim=True)+self.epsilon) # shape: (1, num_mixtures, 1, 1)
        return out
        
    def _calc_E_ln_Lambda_k(self,) -> torch.Tensor:
        "return shape: (1, num_mixtures, 1, 1)"
        out = torch.digamma((self.nu_k + 1 - torch.arange(start=1, end=self.num_variables+1).reshape(1,1,-1,1).expand(-1,self.num_mixtures,-1,-1)) / 2) 
            # shape: (1, num_mixtures, num_variables, 1)
        out = out.sum(dim=2, keepdim=True) + self.num_variables * torch.log(torch.tensor(2., device=out.device)) + torch.logdet(self.W_k+1e-6).reshape(1, self.num_mixtures, 1, 1)
            # shape: (1, num_mixtures, 1, 1)
        return out
    
    def _Entropy_Wishart(self, W: torch.Tensor, nu: torch.Tensor) -> torch.Tensor:
        "return shape: (1, num_mixtures or 1, 1, 1)"
        out = - self._Wishart_ln_B_func(W, nu) \
            - (nu - self.num_variables - 1)/2 * self._calc_E_ln_Lambda_k() \
            + nu*self.num_variables/2 # shape: (1, num_mixtures or 1, 1, 1)
        return out
        
    def _calc_E_whitened_error_k(self, data_frame: torch.Tensor) -> torch.Tensor:
        "return shape: (num_samples, num_mixtures, 1, 1)"
        out_part_0 = self.num_variables/(self.beta_k + 1e-6) # shape: (1, num_mixtures, 1, 1)
        # data_frame: (num_samples, 1, num_variables, 1)
        # m_k: (1, num_mixtures, num_variables, 1)
        # W_k: (1, num_mixtures, num_variables, num_variables)
        # nu_k:(1, num_mixtures, 1, 1)
        x_minus_m = data_frame - self.m_k   # shape: (num_samples, num_mixtures, num_variables, 1)
        out_part_1 = x_minus_m.transpose(-1, -2) @ self.W_k @ x_minus_m # shape: (num_samples, num_mixtures, 1, 1)
        out = out_part_0 + self.nu_k * out_part_1 # shape: (num_samples, num_mixtures, 1, 1)
        return out
    
    def _calc_N_k(self,r_nk) -> torch.Tensor:
        "return shape: (1, num_mixtures, 1, 1)"
        # r_nk: (num_samples, num_mixtures, 1, 1)
        out = r_nk.sum(dim=0, keepdim=True) # shape: (1, num_mixtures, 1, 1)
        return out
    
    def _calc_x_k_bar(self, data_frame: torch.Tensor, r_nk: torch.Tensor) -> torch.Tensor:
        "return shape: (1, num_mixtures, num_variables, 1)"
        # data_frame: (num_samples, 1, num_variables, 1)
        # r_nk: (num_samples, num_mixtures, 1, 1)
        out = r_nk * data_frame # shape: (num_samples, num_mixtures, num_variables, 1)
        out = out.sum(dim=0, keepdim=True) / (self._calc_N_k(r_nk) + 1e-6) # shape: (1, num_mixtures, num_variables, 1)
        return out
    
    def _calc_S_K(self, data_frame: torch.Tensor, x_k_bar: torch.Tensor, r_nk: torch.Tensor) -> torch.Tensor:
        "return shape: (1, num_mixtures, num_variables, num_variables)"
        # r_nk: (num_samples, num_mixtures, 1, 1)
        # data_frame: (num_samples, 1, num_variables, 1)
        # x_k_bar: (1, num_mixtures, num_variables, 1)
        x_n_minus_x_k_bar = (data_frame - x_k_bar)  # shape: (num_samples, num_mixtures, num_variables, 1)
        out = r_nk * (x_n_minus_x_k_bar @ x_n_minus_x_k_bar.transpose(-1, -2)) # shape: (num_samples, num_mixtures, num_variables, num_variables)
        out = out.sum(dim=0,keepdim=True) / (self._calc_N_k(r_nk) + 1e-6)   # shape: (1, num_mixtures, num_variables, num_variables)
        return out
    
    def _calc_ln_rho_nk(self, data_frame: torch.Tensor) -> torch.Tensor:
        "return shape: (num_samples, num_mixtures, 1, 1)"
        ln_rho_nk = self._calc_E_ln_pi_k() + 1/2*self._calc_E_ln_Lambda_k() - self.num_variables/2*torch.log(2*torch.tensor(torch.pi, device=data_frame.device)) \
            - 1/2*self._calc_E_whitened_error_k(data_frame) # shape: (num_samples, num_mixtures, 1, 1)
        return ln_rho_nk
    
    def _calc_r_nk(self, data_frame: torch.Tensor) -> torch.Tensor:
        "return shape: (num_samples, num_mixtures, 1, 1)"
        ln_rho_nk = self._calc_ln_rho_nk(data_frame) # shape: (num_samples, num_mixtures, 1, 1)
        r_nk = torch.softmax(ln_rho_nk, dim=1) # shape: (num_samples, num_mixtures, 1, 1)
        return r_nk
    
    @torch.no_grad()
    def fit(self, data_frame: np.ndarray | torch.Tensor, num_iterations: int=100, max_num_inits: int=10, m_step_interval: int=1, verbose: bool=False, reset=True) -> None:
        """fit the model to the data

        Arguments:
            data_frame -- array-like, shape: (num_samples, num_variables)
        """
        data_frame = torch.tensor(data_frame)
        assert data_frame.dim() == 2 or data_frame.dim() == 4, "data_frame must be 2-dimensional or 4-dimensional"
        if data_frame.dim() == 2:
            data_frame = data_frame.reshape(data_frame.shape[0], 1, data_frame.shape[1], 1) # shape: (num_samples, 1, num_variables, 1)
        data_frame = data_frame.to(torch.float32)
        data_frame = self.clean_data(data_frame)
        self.num_init = 0
        if reset:
            self._init_iteration()
            print(f"****** {self.num_init}-th Fitting Start ******")
        else:
            print(f"****** Continue from E-step: {self.E_step_count}, M-step: {self.M_step_count} ******")
        while self.E_step_count < num_iterations:
            try:
                self.VB_E_step(data_frame)
                if self.E_step_count % m_step_interval == 0:
                    self.VB_M_step(data_frame)
                if verbose:
                    print(f"=== E-step: {self.E_step_count}, M-step: {self.M_step_count} ===")
                    print(f"N_k: {self._calc_N_k(self._calc_r_nk(data_frame))}")
                    print(f"pi: {self.pi}")
                    print(f"m_k: {self.m_k}")
                    print(f"\n")
            except:
                if self.num_init < max_num_inits:
                    print("NaN/Inf encountered. Re-initializing the model.")
                    self._init_iteration()
                    self.num_init += 1
                    print(f"****** {self.num_init}-th Fitting Start ******")
                else:
                    print(f"Fitting terminated. NaN/Inf encountered {self.num_init} times.")
        
    def VB_E_step(self, data_frame: torch.Tensor) -> None:
        """Variational Bayes Expectation Step
        
        q(z) <- r_nk
        q(mu) <- m_k, beta_k
        q(Lambda) <- W_k, nu_k
        
        E-step: (j+1) <- (j)
        ---------------------
        r_nk^(j+1)  <- pi
                    <- E_ln_Lambda_k^(j) <- nu_k^(j), W_k^(j)
                    <- E_whitened_error <- nu_k^(j), W_k^(j), beta_k^(j), m_k^(j)
        alpha_k^(j+1) <- alpha_0 + N_k^(j+1) <- r_nk^(j+1)
        ---------------------
        beta_k^(j+1)<- beta_0
                    <- N_k^(j+1) <- r_nk^(j+1)
        m_k^(j+1)   <- beta_k^(j+1), N_k^(j+1), {x_n}
        ---------------------
        nu_k^(j+1)  <- nu_0, N_k^(j+1)
        W_k^(j+1)   <- W_0, N_k^(j+1)
                    <- S_k^(j+1) <- N_k^(j+1), {r_nk}^(j+1), {x_n}
        
        """
        ln_rho_nk = self._calc_ln_rho_nk(data_frame) # shape: (num_samples, num_mixtures, 1, 1)
        r_nk = torch.softmax(ln_rho_nk, dim=1) # shape: (num_samples, num_mixtures, 1, 1)
        N_k = self._calc_N_k(r_nk) # shape: (1, num_mixtures, 1, 1)
        x_k_bar = self._calc_x_k_bar(data_frame, r_nk) # shape: (1, num_mixtures, num_variables, 1)
        S_k = self._calc_S_K(data_frame, x_k_bar, r_nk) # shape: (1, num_mixtures, num_variables, num_variables)
        
        self.alpha_k = self.alpha_0 + N_k # shape: (1, num_mixtures, 1, 1)
        
        self.beta_k = self.beta_0 + N_k # shape: (1, num_mixtures, 1, 1)
        self.m_k = 1/(self.beta_k + 1e-6) * ((self.beta_0*self.m_0) + N_k*x_k_bar) # shape: (1, num_mixtures, num_variables, 1)
        
        foo = x_k_bar - self.m_0 # shape: (1, num_mixtures, num_variables, 1)
        # full rank + rank 1 + rank 1 = full rank. also positive definite + positive semidefinite + positive semidefinite = positive definite
        self.inv_W_k = self.W_0.inverse() \
            + N_k * S_k \
            + (self.beta_0 * N_k)/(self.beta_0 + N_k) * (foo @ foo.transpose(-1, -2)) # shape: (1, num_mixtures, num_variables, num_variables)
        # assert not torch.det(self.inv_W_k).isinf().any()
        self.W_k = self.inv_W_k.inverse() \
            + self.epsilon*torch.eye(self.num_variables, device=self.inv_W_k.device).expand_as(self.W_k) # shape: (1, num_mixtures, num_variables, num_variables)
        self.nu_k = self.nu_0 + N_k # shape: (1, num_mixtures, 1, 1)
        
        self.E_step_count += 1
        
    @property
    def pi(self,) -> torch.Tensor:
        "mean of the mixture weights"
        try:
            pi = self.alpha_k / self.alpha_k.sum(dim=1, keepdim=True) # shape: (1, num_mixtures, 1, 1)
            return pi
        except:
            raise Exception("pi not available. likely because the computation hasn't started yet.")
        
    def VB_M_step(self, data_frame: torch.Tensor) -> None:
        """Variational Bayes Maximization Step
        
        TODO: be more considerate about this step with a more rigorous derivation
        
        pi^(j+1) <- N_k^(j+1)

        """
        # self.pi = self._calc_N_k() / data_frame.shape[0]
        
        self.M_step_count += 1
        
    def forward(self, data_frame: torch.Tensor) -> torch.Tensor:
        """return the log-likelihood of the data
        
        Arguments:
            data_frame {torch.Tensor} -- shape: (num_samples, num_variables)
        """
        assert data_frame.dim() == 2 or data_frame.dim() == 4, "data_frame must be 2-dimensional or 4-dimensional"
        num_samples = data_frame.shape[0]
        if data_frame.dim() == 2:
            data_frame = data_frame.reshape(data_frame.shape[0], 1, data_frame.shape[1], 1) # shape: (num_samples, 1, num_variables, 1)
        r_nk = self._calc_r_nk(data_frame)
        self.r_nk = r_nk # !! just for printint information
        x_k_bar = self._calc_x_k_bar(data_frame, r_nk) # shape: (1, num_mixtures, num_variables, 1)
        S_k = self._calc_S_K(data_frame, x_k_bar, r_nk) # shape: (1, num_mixtures, num_variables, num_variables)
        N_k = self._calc_N_k(r_nk) # shape: (1, num_mixtures, 1, 1)
        E_ln_Lambda_k = self._calc_E_ln_Lambda_k() # shape: (1, num_mixtures, 1, 1)
        r_nk = torch.softmax(self._calc_ln_rho_nk(data_frame), dim=1) # shape: (num_samples, num_mixtures, 1, 1)
        E_ln_pi_k = self._calc_E_ln_pi_k() # shape: (1, num_mixtures, 1, 1)
        assert not torch.isnan(E_ln_pi_k).any()
        
        E_ln_p_x_given_z_mu_Lambda = 1/2 * (N_k*(E_ln_Lambda_k - self.num_variables/self.beta_k - self.nu_k*batch_trace(S_k@self.W_k) \
            - self.nu_k*(x_k_bar - self.m_k).transpose(-1, -2) @ self.W_k @ (x_k_bar - self.m_k) \
                - self.num_variables*torch.log(torch.tensor(2*torch.pi, device=data_frame.device)))).sum(dim=1, keepdim=True) # shape: (1, 1, 1, 1)
        E_ln_p_z_given_pi = (r_nk * E_ln_pi_k).sum(dim=0, keepdim=True).sum(dim=1, keepdim=True) # shape: (1, 1, 1, 1)
        E_ln_p_pi = self._Dir_ln_C_func(self.alpha_0) + ((self.alpha_0 - 1) * E_ln_pi_k).sum(dim=1, keepdim=True) # shape: (1, 1, 1, 1)
        E_ln_p_mu_Lambda_part_0 = 1/2 * (self.num_variables*torch.log(self.beta_0/(2*torch.pi)) \
            + E_ln_Lambda_k - self.num_variables*self.beta_0/self.beta_k \
            - self.beta_0*self.nu_k*(self.m_k - self.m_0).transpose(-1, -2) @ self.W_k @ (self.m_k - self.m_0)).sum(dim=1, keepdim=True) # shape: (1, 1, 1, 1)
        E_ln_p_mu_Lambda_part_1 = self.num_mixtures*self._Wishart_ln_B_func(self.W_0, self.nu_0).sum(dim=1, keepdim=True) # shape: (1, 1, 1, 1)
        E_ln_p_mu_Lambda_part_2 = (self.nu_0 - self.num_variables -1)/2 * E_ln_Lambda_k.sum(dim=1, keepdim=True) # shape: (1, 1, 1, 1)
        E_ln_p_mu_Lambda_part_3 = -1/2 * (self.nu_k * batch_trace(self.W_0.inverse() @ self.W_k)).sum(dim=1, keepdim=True) # shape: (1, 1, 1, 1)
        E_ln_p_mu_Lambda = E_ln_p_mu_Lambda_part_0 + E_ln_p_mu_Lambda_part_1 + E_ln_p_mu_Lambda_part_2 + E_ln_p_mu_Lambda_part_3 # shape: (1, 1, 1, 1)
        assert not torch.isnan(E_ln_p_mu_Lambda).any()
        
        E_ln_q_z = (r_nk * torch.log(r_nk + 1e-6)).sum(dim=0, keepdim=True).sum(dim=1, keepdim=True).sum(dim=0, keepdim=True) # shape: (1, 1, 1, 1)
        E_ln_q_pi = ((self.alpha_k - 1) * E_ln_pi_k).sum(dim=1, keepdim=True) + self._Dir_ln_C_func(self.alpha_k) # shape: (1, num_mixtures, 1, 1)
        E_ln_q_mu_Lambda = (1/2 * E_ln_Lambda_k + self.num_variables/2 * torch.log(self.beta_k/(2*torch.pi)) \
            - self.num_variables/2 - self._Entropy_Wishart(self.W_k, self.nu_k)).sum(dim=1, keepdim=True) # shape: (1, 1, 1, 1)
        assert not torch.isnan(E_ln_q_mu_Lambda).any()
        
        log_likelihood = E_ln_p_x_given_z_mu_Lambda + E_ln_p_z_given_pi + E_ln_p_pi + E_ln_p_mu_Lambda \
            - E_ln_q_z - E_ln_q_pi - E_ln_q_mu_Lambda # shape: (1, 1, 1, 1)
            
        log_likelihood = log_likelihood / num_samples # shape: (1, 1, 1, 1), normalize by the number of samples
        assert not torch.isnan(log_likelihood).any()
        assert not torch.isinf(log_likelihood).any()
            
        return log_likelihood
        
    def sample(self, num_samples: int) -> torch.Tensor:
        alpha_k = self.alpha_k[self.alpha_k > 1e-4]
        post_pi = dist.Dirichlet(self.alpha_k.squeeze())
        sampled_pi = post_pi.sample(torch.Size((num_samples,))) # shape: (num_samples, num_mixtures)
        # post_z = dist.Categorical(sampled_pi) 
        pi = sampled_pi
        post_z = dist.Categorical(pi.squeeze()) # use expected value or sampled value of pi?
        sampled_z = post_z.sample() # shape: (num_samples,)
        sampled_z = nn.functional.one_hot(sampled_z, num_classes=self.num_mixtures) # shape: (num_samples, num_mixtures)
        post_Lambda = dist.Wishart(self.nu_k.squeeze(), self.W_k.squeeze())
        sampled_Lambda = post_Lambda.sample(torch.Size((num_samples,))) # shape: (num_samples, num_mixtures, num_variables, num_variables)
        E_Lambda = sampled_Lambda
        Lambda = E_Lambda
        post_mu = dist.MultivariateNormal(self.m_k.squeeze(-1), (self.beta_k*Lambda).inverse()) # m_k shape: (1, num_mixtures, num_variables, 1)
        sampled_mu = post_mu.sample() # shape: (num_samples, num_mixtures, num_variables)
        post_x_k = dist.MultivariateNormal(sampled_mu, Lambda) # shape: (num_samples, num_mixtures, num_variables)
        sampled_x_k = post_x_k.sample() # shape: (num_samples, num_mixtures, num_variables)
        sampled_x = sampled_x_k * sampled_z.unsqueeze(-1) # shape: (num_samples, num_mixtures, num_variables)
        sampled_x = sampled_x.sum(dim=1) # shape: (num_samples, num_variables)

        return sampled_x
    
    def clean_data(self, data_frame: torch.Tensor) -> torch.Tensor:
        if data_frame.dim() == 2:
            data_frame = data_frame.reshape(data_frame.shape[0], 1, data_frame.shape[1], 1) # shape: (num_samples, 1, num_variables, 1)
        data_frame = data_frame[~torch.isnan(data_frame).any(dim=2).squeeze((-1,-2)), :, :, :]  # Remove 'NaNs'
        data_frame = data_frame[~(data_frame == torch.inf).any(dim=2).squeeze((-1,-2)), :, :, :]  # Remove 'np.inf'
        
        return data_frame
    
    def __repr__(self,) -> str:
        print_str = f"GaussianMixtureModel(num_variables={self.num_variables}, num_mixtures={self.num_mixtures})"
        print_str += "\n"
        print_str += f"N_k: {self._calc_N_k(self.r_nk)}" + "\n"
        print_str += f"pi: {self.pi}" + "\n"
        print_str += (f"m_k: {self.m_k}")
        
        return print_str
    
    def _isnan(self, x):
        return torch.isnan(x).any()
    
    def _isinf(self, x):
        return torch.isinf(x).any()
    
    def fw_hook_isnan(self, module, input, output):
        if torch.isnan(output).any():
            print("NaN detected in forward pass")
            raise Exception("NaN detected in forward pass")
    
def main():
    fake_data = torch.concat([
        torch.randn((50, 2)) + torch.tensor([10., -10.]),
        1.5*torch.randn((100, 2)) + torch.tensor([20., 20.]),
        0.5*torch.randn((50, 2)) + torch.tensor([-10., 10.])
    ])
    # fake_data = torch.randn((100, 2)) + 17.
    model = GaussianMixtureModel(2, 4)
    model.fit(fake_data, num_iterations=50, verbose=True)
    print(model(fake_data))
    samples = model.sample(500)
    # print(samples)
    print(model(samples))
    fig, axes = plt.subplots(1,2)
    axes[0].scatter(fake_data[:,0], fake_data[:,1])
    axes[0].set_xlim(-30, 30)
    axes[0].set_ylim(-30, 30)
    axes[1].scatter(samples[:,0], samples[:,1])
    axes[1].set_xlim(-30, 30)
    axes[1].set_ylim(-30, 30)
    plt.savefig("example_gmm.png", dpi=300)
    
if __name__ == "__main__":
    main()