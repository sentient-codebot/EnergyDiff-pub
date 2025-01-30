"""
Code by Mauricio Salazar
Multivariate Elliptical Copulas (Gaussian and t-student)

Synthetic profiles using the Copula approach
"""

import numpy as np
import warnings
import pandas as pd
from scipy.stats import multivariate_normal, norm, t, wasserstein_distance, ks_2samp, shapiro
from scipy import optimize, interpolate
import energydiff.models.elliptical_distributions_study as utils


class EllipticalCopula:
    """
    An elliptical copula object, represent a copula model using a multivariate probability distribution which
    could be Multivariate 'Gaussian - Normal' (MVN) or Multivariate 't-student' (MVT).

    The transformation of the data to the uniform hyper-cube is done using a Probability Integral Transform (PIT),
    using and Estimated Conditional Distribution function (ECDF) aided by an interpolation that could be 'linear'
    or 'cubic spline'.

    The method for inference the parameters of the MVN is the Kendall's tau rank correlation transformed to the
    pearson correlation, using the relation: rho = sin((np.pi * tau) / 2). This relation creates the covariance matrix.

    The calculation of the 'nu' (Degrees of Freedom - DoF) for the MVT copula is done fixing the covariance matrix
    using the kendall's tau relation, same as MVN, and performing a numerical optimization over the variable 'nu',
    minimizing the neg-log-likelihood of the 'Multivariate t-student copula'

    Parameters:
    -----------
        data_frame: (obj:: numpy.array)
            2-D Numpy array in which rows represent variables and columns the instances of the variables (samples).
            The dimensions of the array is [D x N]. Where D: dimensions,  N: Samples number
        type: (str)
            The type of copula that want to be fitted. Only two options available: 'gaussian' or 't'

    Attributes:
    -----------
        covariance: (obj:: numpy.array)
            2-D Covariance matrix with dimension [D x D]. Calculated using kendall's-tau.
        dim: (int)
            Value of the number of dimension of the copula ('D')

        ecdf: (obj::list::scipy.interp1d)
            The empirical continuous distribution function calculated by the probability integral transform (PIT).
            Each element of the list is a linear models of each variable.

        icdf: (obj::list::scipy.interp1d)
            The inverse of the empirical continuous distribution function calculated by the probability integral
            transform (PIT).
            Each element of the list is a linear models of each variable.

        uniform_samples: (obj:: numpy.array)
            Samples transformed to the uniform hypercube using the PIT.
            This is a 2-D matrix with dimensions [D X N]

        nu: (float)
            Optimal 'nu' (Degrees of Freedom - DoF) calculated using numerical optimization.


    Methods:
    -------
        explore(...):
            Assuming that the data is a time series and each row represent a time step, this method creates a
            cartesian plot of the time series and the Spearman correlation plot.

        fit(..., plot=True):
            Fit the MVN or MVT according to the choice in 'type' on the instantiation of the class.
            If it is MVN, the fit is only calculate the covariance matrix Sigma.
            If it is MVT, the fit calculates the sigma and the degrees of freedom 'nu'. The parameter 'nu' is calculated
            via numerical optimization. If 'plot=True' a linear search is done and a plot of the negative log likelihood
            is shown, highlighting the optimal 'nu'. Also, the negative log likelihood of a gaussian is shown.

        sample(..., n_samples=1000, conditioned=True, variables={'x1': 0.5})
            Sample the fitted elliptical distribution.
            If 'conditioned=True' the sigma, mean and nu (if it is a MVT) parameters are recalculated based on the
            values provided the 'variables' input (conditioned distribution parameters).

        _probability_integral_transform(...):
            Returns the samples with uniform distribution,the models of the ecdf, and icdf (see in attributes
            of the class)

    """

    #TODO: If the object is initialized as 'gaussian'it will cretae a conflict in the model selection summary.
    def __init__(self, data_frame, copula_type='t', interpolation='linear'):

        assert (interpolation.lower() in ['linear', 'spline']),  'Wrong interpolation method'
        assert (copula_type.lower() in ['gaussian', 't']), 'Wrong distribution selected'

        self.sample_size = None
        self.data_frame = None

        if isinstance(data_frame, pd.DataFrame):
            self._check_clean_data_frame(data_frame.values)
        elif isinstance(data_frame, np.ndarray):
            self._check_clean_data_frame(data_frame)
        else:
            raise ValueError('data should be in a numpy array or pandas data frame')

        self.copula_type = copula_type
        self.interpolation = interpolation
        self.covariance_kendall = None  # Pearson's correlation using the kendall's tau
        self.covariance_spearman = None  # Pearson's correlation using the spearman's rho
        self.tau_matrix = None  # Kendall's tau matrix
        self.spearman_matrix = None  # Spearman's rho matrix
        self.dim = data_frame.shape[0]
        self.iecdf = []
        self.ecdf = []
        self.uniform_samples = None
        self.nu = None
        self.allow_singular = False
        self.neg_log_likelihood_t = None
        self.neg_log_likelihood_gaussian = None
        self.aic_t = None
        self.bic_t = None
        self.aic_gaussian = None
        self.bic_gaussian = None
        self.copula_fitted = False
        # self.copula_type = None  # Best copula selected
        self._verbose = False  # Set true to get all diagnosis plots.

    def __repr__(self):
        if self.copula_fitted:
            return ('-------------------------------------------\n' +
                    'Elliptical copula object: \n' +
                    '-------------------------------------------\n'
                    f'\tCopula type: {self.copula_type} \n' +
                    f'\tDimensions: {self.dim} \n' +
                    f'\tInterpolation method: {self.interpolation} \n' +
                    f'\t\'t\' Neg-log-likelihood: {self.neg_log_likelihood_t:.2f} \n' +
                    f'\t\'Gaussian\' Neg-log-likelihood: {self.neg_log_likelihood_gaussian:.2f} \n' +
                    f'\tNu (Degrees of Freedom): {self.nu:.2f} \n' +
                    '-------------------------------------------\n'
                    f'\tAIC - \'t-student Copula\': {self.aic_t:.2f}\n' +
                    f'\tAIC - \'Gaussian Copula\': {self.aic_gaussian:.2f}\n' +
                    f'\tBIC - \'t-student Copula\': {self.bic_t:.2f}\n' +
                    f'\tBIC - \'Gaussian Copula\': {self.bic_gaussian:.2f}\n' +
                    f'\tBEST MODEL (BIC selection): {self.copula_type}\n' +
                    f'\tDelta BIC: 2 * (BIC.t/BIC.gaussian): {(2 * self.bic_t / self.bic_gaussian).round(2)}\n')

        else:
            return ('-------------------------------------------\n' +
                    'Elliptical copula object: \n' +
                    '-------------------------------------------\n'
                    f'\tCopula type: {self.copula_type} \n' +
                    f'\tDimensions: {self.dim} \n' +
                    f'\tCOPULA IS NOT FITTED!!!!!')

    def fit(self, nu_bounds=((2, 200),),
            x0=5,
            plot_linear_search=False,
            step_size=200,
            plot_uniform_transform=False,
            plot_uniform_variable=1):
        """
        Fit the copula to the data.
        """
        (self.covariance_kendall,
         self.tau_matrix,
         self.covariance_spearman,
         self.spearman_matrix) = utils.covariance_kendall_tau(data_samples=self.data_frame)

        (self.uniform_samples,
         self.ecdf,
         self.iecdf) = utils.probability_integral_transform(data=self.data_frame,
                                                            plot=plot_uniform_transform,
                                                            variable=plot_uniform_variable,
                                                            interpolation=self.interpolation)

        # try:
        #     assert utils.is_pos_def(self.covariance_kendall),  "Matrix is not Positive Definite"
        # except AssertionError:
        #     warnings.warn('Covariance from Kendall\'s tau is not Positive semidefinite.\nUsing sperman\'s instead')
        #     self.covariance_kendall = self.covariance_spearman
        #     assert utils.is_pos_def(self.covariance_kendall), "Spearman's Matrix is not Positive Definite"

        # Check if the covariance matrix is positive semi-definite. If not apply near correlation matrix approximation.
        # TODO: Eliminate the nested try-except blocks
        if utils.is_pos_def(self.covariance_kendall):
            pass
        elif utils.is_pos_def(self.covariance_spearman):
            warnings.warn('Covariance from Kendall\'s tau is not Positive semidefinite.\nUsing sperman\'s instead')
            self.covariance_kendall = self.covariance_spearman
        elif utils.is_pos_def(self.tau_matrix):
            warnings.warn('Covariance from Kendall\'s tau  and Spearman\'s are not Positive semidefinite.\n'
                          'Trying BRUTE FORCE approach with raw-kendall matrix. Modelling could be incorrect')
            self.covariance_kendall = self.tau_matrix

        elif utils.is_pos_def(self.tau_matrix):
            warnings.warn('Covariance from Kendall\'s tau  and Spearman\'s are not Positive semidefinite.\n'
                          'Trying BRUTE FORCE approach with raw-Spearman matrix. Modelling could be incorrect')
            self.covariance_kendall = self.spearman_matrix
        else:
            warnings.warn('Need a method to find near correlation matrix')
            return NotImplementedError



        # else:
        #     warnings.warn('Covariance from Kendall\'s tau  and Spearman\'s are not Positive semidefinite.\n'
        #                   'Trying near correlation approximation using projection in MATLAB.\n'
        #                   'Check: nearcorr() in: https://nl.mathworks.com/help/stats/nearcorr.html')
        #     try:
        #         import matlab.engine
        #         import matlab
        #     except ModuleNotFoundError:
        #         warnings.warn("Install Matlab engine for python:"
        #                       "\nhttps://stackoverflow.com/questions/51406331/how-to-run-matlab-code-from-within-python")
        #         return ModuleNotFoundError
        #
        #     print("Starting Matlab engine...")
        #     eng = matlab.engine.start_matlab()
        #     print("Start... OK!")
        #
        #     NCM = eng.nearcorr(matlab.double(self.covariance_kendall.tolist()), 'method', 'projection')
        #     NCM = np.asarray(NCM)
        #
        #     self.allow_singular = True  # Matrix could be PSD but singular.
        #
        #     if utils.is_pos_def(NCM):
        #         self.covariance_kendall = NCM
        #     else:
        #         warnings.warn("Near correlation failed for Covariance Kendall\'s tau. Trying on Spearman\'s covariance.")
        #         NCM = eng.nearcorr(matlab.double(self.covariance_spearman.tolist()), 'method', 'projection')
        #         NCM = np.asarray(NCM)
        #
        #         try:
        #             assert utils.is_pos_def(NCM),  "Matrix is not Positive Definite"
        #             self.covariance_kendall = NCM
        #         except AssertionError:
        #             warnings.warn('Near correlation matrix approximation failed...')
        #             return AssertionError


        if self.copula_type == 't':
            # Bound the degrees of freedom search for the t-distribution
            if x0 is None:
                x_0 = utils.initial_guess(self.data_frame)
                print(f'Initialization of nu.  x_0 = {x_0}')
            else:
                x_0 = x0
                print(f'Initialization of nu.  x_0 = {x_0}')

            result = optimize.minimize(utils.neg_log_likelihood_copula_t,
                                       x0=np.array(x_0),
                                       method='SLSQP',
                                       bounds=nu_bounds,
                                       args=(self.uniform_samples,
                                             self.covariance_kendall,
                                             self.dim),
                                       options={'disp': False})
            print('\n')
            print('-------------------------------------------')
            print('"t-student" Copula (Numerical Optimization)')
            print('-------------------------------------------')
            print(f'Best nu value: {result.x}')
            print(f'Neg log-likelihood: {result.fun:.2f}')

            self.nu = result.x[0]
            self.neg_log_likelihood_t = result.fun
            self.copula_fitted = True

            # Log-likelihood gaussian
            values = utils.gaussian_copula(uniform_values=self.uniform_samples,
                                           covariance=self.covariance_kendall,
                                           dim=self.dim)

            # values = values[~np.isnan(values)]  # Remove the nan
            # values = values[~(values == np.inf)]  # Remove the division by zero in the copula
            # values[values <= 0.0] = np.finfo(np.float64).eps  # Remove the warning for creating np.inf values
            values = -np.log(values)

            self.neg_log_likelihood_gaussian = np.nansum(values)
            print('\n')
            print('-------------------------------------------')
            print('Gaussian Copula')
            print('-------------------------------------------')
            print(f'Neg log-likelihood: {self.neg_log_likelihood_gaussian:.2f}')
            print('\n')

            if plot_linear_search:
                utils.neg_log_likelihood_copula_t_plot(data_samples=self.uniform_samples,
                                                       covariance=self.covariance_kendall,
                                                       dim=self.dim,
                                                       upper_bnd=nu_bounds[0][1], step_size=step_size)
        else:
            # Log-likelihood gaussian
            values = utils.gaussian_copula(uniform_values=self.uniform_samples,
                                           covariance=self.covariance_kendall,
                                           dim=self.dim)

            # values = values[~np.isnan(values)]  # Remove the nan
            # values = values[~(values == np.inf)]  # Remove the division by zero in the copula
            # values[values <= 0.0] = np.finfo(np.float64).eps  # Remove the warning for creating np.inf values
            values = -np.log(values)

            self.neg_log_likelihood_gaussian = np.nansum(values)
            self.copula_fitted = True  # Gaussian copula only need the covariance matrix

        # Calculate AIC and BIC for 't-student' and 'Gaussian'.
        if self.copula_type == 't':
            param_gauss = self.dim * (self.dim - 1) / 2
            param_t = self.dim * (self.dim - 1) / 2 + 1

            self.aic_t = -2 * -self.neg_log_likelihood_t + 2 * param_t
            self.aic_gaussian = -2 * -self.neg_log_likelihood_gaussian + 2 * param_gauss

            self.bic_t = -2 * -self.neg_log_likelihood_t + np.log(self.sample_size) * param_t
            self.bic_gaussian = -2 * -self.neg_log_likelihood_gaussian + np.log(self.sample_size) * param_gauss

            models = ['t', 'gaussian']

            print('-------------------------------------------')
            print('Model comparison')
            print('-------------------------------------------')
            print('AIC: -2 * ln(likelihood) + 2 * parameters')
            print('BIC: -2 * ln(likelihood) + ln(#_samples) * parameters')
            print('-------------------------------------------')
            print(f'# of Samples: {self.sample_size:.0f}')
            print(f'Dimensions: {self.dim:.0f}, Lower triangular matrix elements in sigma : (d * (d - 1)) / 2')
            print(f'Parameters Gaussian: {param_gauss:.0f}')
            print(f'Parameters t-student: {param_t:.0f}')
            print(f'AIC - \'t-student Copula\': {self.aic_t:.2f}')
            print(f'AIC - \'Gaussian Copula\': {self.aic_gaussian:.2f}')
            print(f'BIC - \'t-student Copula\': {self.bic_t:.2f}')
            print(f'BIC - \'Gaussian Copula\': {self.bic_gaussian:.2f}')

            # Select the best model for sampling
            # if self.nu > nu_bounds[0][1] - 3:  # 3 is a rule of thumb threshold
            if self.nu > 100:  # A Rule of thumb threshold
                warnings.warn('The optimal \'nu\' is close to optimization boundary. \'gaussian\' set as best MODEL')
                self.copula_type = 'gaussian'
            else:
                self.copula_type = models[np.argmin([self.bic_t, self.bic_gaussian])]

            print(f'BEST MODEL (BIC selection): {self.copula_type}')
            print(f'Delta BIC: 2 * (BIC.t/BIC.gaussian): {(2 * self.bic_t/self.bic_gaussian).round(2)}')
            print('\n')
            print('-------------------------------------------')
            print('Delta BIC guidelines [1]:')
            print('-------------------------------------------')
            print('0 to 2: \tNot to worth bare mention')
            print('2 to 6: \tPositive')
            print('6 to 10:\tStrong')
            print('>10:    \tVery Strong')
            print('\n')
            print('[1] \'Kass, R. E.; Raftery, A.E. (1995). "Bayes Factors". Journal of the American Statistical '
                  'Association, Vol. 90. No. 430. page 777')
            print('\n\n\n\n')

    def sample(self, n_samples=5000, conditional=False, variables={'x3': 3.3}, negative_clamp=False, drop_inf=True, silent=True):
        '''
        Returns a np.array (rows are variables (time-steps), columns are instances)
        :param n_samples:
        :param conditional:
        :param variables:
        :param negative_clamp:
        :param silent:
        :return:
        '''

        assert self.copula_fitted, 'A copula should fitted first'

        if not silent:
            print('\n')
            print('-------------------------------------------')
            print('Multivariate "' + self.copula_type + '" Copula.....')

        if conditional:
            assert (self._check_variables_values(variables)), 'One of the variables is out of bounds of the model!'
            (mean_sample,
             cov_sample,
             nu_sample,
             idx_sample) = self.conditional_parameters(variables)
            variables_num = np.linspace(0, self.dim - 1, self.dim, dtype=np.int16)
            variables_num = variables_num[idx_sample]

        else:
            mean_sample = np.zeros((self.dim, 1))
            cov_sample = self.covariance_kendall
            nu_sample = self.nu
            variables_num = range(self.dim)

        if self._verbose:  # To troubleshoot
            print(f'mean used to sample: {mean_sample}')
            print(f'nu used to sample: {nu_sample}')
            print(f'covariance used to sample: \n{cov_sample}')

        '''
        SAMPLING
        '''
        if not silent:
            print('\n')
            print('-------------------------------------------')
            print('Sampling start:')
            print('-------------------------------------------')
            print('Number of samples requested: ' + str(n_samples))

        iteration = 0
        max_iteration = 100
        total_samples = 0
        samples_complete = np.zeros((cov_sample.shape[0], 1))

        while (total_samples < n_samples) and (iteration <= max_iteration):
            if not silent: print(f'iteration: {iteration}')

            if self.copula_type == 'gaussian':
                if mean_sample.shape[0] > 1:  # At least is bivariate
                    samples_matrix = multivariate_normal(mean_sample.squeeze(),
                                                         cov_sample,
                                                         allow_singular=self.allow_singular).rvs(n_samples).T
                else:  # The conditional is univariate
                    samples_matrix = norm(mean_sample.squeeze(), cov_sample.squeeze()).rvs(n_samples).reshape(1, -1)

            else:  # 't-student' copula
                if mean_sample.shape[0] > 1:  # At least is bivariate
                    samples_matrix = utils.samples_multivariate_t(mean=mean_sample,
                                                                  covariance=cov_sample,
                                                                  nu=nu_sample,
                                                                  n_samples=n_samples)
                else:  # The conditional is univariate
                    samples_matrix = t(df=nu_sample, loc=mean_sample, scale=cov_sample).rvs(n_samples).reshape(1, -1)

            # This is just for debug
            if conditional and self._verbose:
                ax_ = utils.plot_samples(samples_matrix)
                ax_.grid(True)
                ax_.set_title('Samples from Generator')

            # Transformation to the Actual Domain
            samples_copula = []
            samples_inv_matrix = []
            for ii, variable in enumerate(variables_num):
                if self.copula_type == 'gaussian':
                    samples_inv = norm.cdf(samples_matrix[ii, :])
                else:  # 't-student' copula
                    samples_inv = t(df=nu_sample).cdf(samples_matrix[ii, :])
                # self.samples_inv = self.fixSamplesArray(self.samples_inv, variable, 'icdf')
                samples_inv_matrix.append(samples_inv)

                if self.interpolation == 'linear':
                    samples_copula.append(self.iecdf[variable](samples_inv))
                else:
                    samples_copula.append(interpolate.splev(samples_inv, self.iecdf[variable]))

            samples_copula = np.array(samples_copula)  # Reminder: 2-D array dimensions [D X n_samples]: D: Variables
            samples_inv_matrix = np.array(samples_inv_matrix)

            # This is just for debug
            if conditional and self._verbose:
                ax_ = utils.plot_samples(samples_inv_matrix)
                ax_.grid(True, linestyle='--')
                ax_.set_title('Samples from Generator - Normalized')

            # Clean the samples:
            # samples_copula[:, np.isnan(samples_copula).any(axis=0)] = 0  # Remove 'NaNs'
            samples_copula = samples_copula[:, ~np.isnan(samples_copula).any(axis=0)]  # Remove 'NaNs'
            samples_copula = samples_copula[:, ~(samples_copula == np.inf).any(axis=0)]  # Remove 'np.inf'

            if drop_inf: # TODO: Check if this change doesn't cause problems (DROP THE IF)
                samples_copula = samples_copula[:, ~(samples_copula == -np.inf).any(axis=0)]  # Remove 'np.inf'  # TODO: Check if this change doesn't cause problems

            if not silent: print(f'Number of samples generated after droping NaNs and inf: {samples_copula.shape[1]}')

            if negative_clamp:
                # Drop all the samples that gives values below 0
                idx_neg = samples_copula < 0
                samples_copula = samples_copula[:, ~idx_neg.any(axis=0)]
                if not silent: print(f'Number of samples generated after removing negative values: '
                                     f'{samples_copula.shape[1]}')

            samples_complete = np.hstack((samples_complete, samples_copula))
            iteration = iteration + 1
            total_samples = samples_complete.shape[1] - 1  # -1 because the first row of 0's doesnt count
            if not silent: print(f'Total samples so far: {total_samples}\n\n')

        if iteration > max_iteration:
            raise Warning('Something is wrong with the model... you can not sample from it.')

        samples_complete = samples_complete[:, 1: n_samples + 1]  # Drop first column that it is only zeros

        return samples_complete

    def conditional_parameters(self, variables):
        r"""
        Calculate the conditional parameters: covariance (\sigma), mean (\mu) and degrees of freedom (\nu),
        for the elliptical distributions. The notation is the following:

            Covariance block matrix:
            -----------------------
                \sigma = [[\sigma_{aa}    ,  \sigma_{ab}],
                          [\sigma_{ab}^{T},  \sigma_{bb}]]

                \sigma{ba} == \sigma{ab}^{T}

            Conditional mean:
            -----------------
                \mu{a|b} = \mu_{a} + \sigma_{ab}^{T} * \sigma_{bb}^{-1} * (x_{b} - \mu_{b})

            Conditional covariance:
            -----------------------
                \sigma_{a|b} = k_cond * \sigma_{aa} - \sigma_{ab}^{T} * \sigma_{bb}^{-1} * \sigma_{ba}

                k_cond = 1   for  'gaussian'
                k_cond = (\nu + (x_{b} - \mu_{b})^{T} * \sigma_{bb}^{-1} * (x_{b} - \mu_{b})) / (\nu + d_{b})

                where d_{b}: Dimension of the known variables (e.g. how many variables are conditioned)

            Conditional degrees of freedom (nu):
            ------------------------------------
                \nu_{a|b} = \nu + d_{b}


        Return:
        ------
            mu_cond: (obj:: numpy.array)
                    2-D numpy array with dimension [(D - P) x 1]. P: Dimension of known variables.
                    (e.g. variables={'x2': 3.5, 'x4': 6.9}, then P = 2)

            sigma_cond:
            (obj:: numpy.array)
                    2-D numpy array with dimension [(D - P) x (D - P)]

            nu_cond:
            (obj:: numpy.array)
                    2-D numpy array with dimension [1 x 1]

        """

        known_var_idx = []
        value_var = []
        for key in variables.keys():
            value_var.append(float(variables[key]))
            known_var_idx.append(int(key.replace('x', '')) - 1)
        known_var_idx = np.array(known_var_idx)
        value_var = np.array(value_var)

        assert ((self.dim - known_var_idx.max()) > 0), 'Cond. variables has higher or equal dimension than model'
        assert ((self.dim - len(known_var_idx)) > 0), 'Number of cond. variables are more than dimensions in the model'

        shift_idx = np.array([False] * self.dim)
        shift_idx[known_var_idx.tolist()] = True

        variables_num = np.linspace(0, self.dim - 1, self.dim, dtype=np.int16)
        variables_num = variables_num[shift_idx]

        for ii, value in enumerate(variables_num):
            # Transform the variable value to uniform hyper cube
            if self.interpolation == 'linear':
                value_var[ii] = self.ecdf[value](value_var[ii])
            else:
                value_var[ii] = interpolate.splev(value_var[ii], self.ecdf[value])

            if self.copula_type == 'gaussian':
                value_var[ii] = norm.ppf(value_var[ii])  # Transform to the normal space (\phi^{-1})
            else:  # 't' copula
                value_var[ii] = t(df=self.nu).ppf(value_var[ii])

        value_var = np.array(value_var).reshape(len(value_var), 1)

        # Calculate the conditional covariance, mean and degrees of freedom
        # Pre-locate memory:
        dim_new = self.dim - len(known_var_idx)
        sigma_cond = np.zeros((dim_new, dim_new))
        mu_cond = np.zeros((dim_new, 1))
        d_B = len(known_var_idx)  # Dimensions of the known variables d_{b}

        # --------------------------------------
        # SIGMA CONDITIONAL:  Sigma_(a|b)
        # --------------------------------------
        # Block A will be the one to marginalize. p(x_a | x_b).
        # Meaning: a -> unknowns   b -> known, provided, fixed values
        # Covariance matrix will be build as:
        # | A   B |
        # | B^T D |

        cov_matrix = np.array(self.covariance_kendall)

        sigma_D = cov_matrix[shift_idx, :][:, shift_idx]
        sigma_A = cov_matrix[~shift_idx, :][:, ~shift_idx]
        sigma_B = cov_matrix[~shift_idx, :][:, shift_idx]

        # --------------------------------------
        # MEAN CONDITIONAL:  Mu_(a|b)
        # --------------------------------------
        # Means organized to follow the same convention
        # | mu_a |
        # | mu_b |

        mean_vector = np.array(np.zeros((self.dim, 1)))

        mu_A = mean_vector[~shift_idx]
        mu_B = mean_vector[shift_idx]

        if self.copula_type == 'gaussian':
            k_cond = 1
        else:
            k_cond = ((self.nu
                       + np.matmul(np.matmul((value_var - mu_B).T, np.linalg.inv(sigma_D)), (value_var - mu_B)))
                      / (self.nu + d_B))

        sigma_cond[:, :] = k_cond * (sigma_A - np.matmul(np.matmul(sigma_B, np.linalg.inv(sigma_D)), sigma_B.T))
        mu_cond[:] = mu_A + np.matmul(np.matmul(sigma_B, np.linalg.inv(sigma_D)), (value_var - mu_B))

        if self.copula_type == 't':
            # --------------------------------------
            # NU (Degrees of Freedom - DoF) CONDITIONAL:  Nu_(a|b)
            # --------------------------------------
            # DoF organized to follow the same convention
            # | nu_a |
            # | nu_b |

            nu_cond = self.nu + d_B

        else:
            nu_cond = None

        unknown_variables_index = ~shift_idx

        return mu_cond, sigma_cond, nu_cond, unknown_variables_index

    def _check_variables_values(self, variables):
        # Extract the index of the known variables
        known_var_idx = []
        value_var = []
        for key in variables.keys():
            value_var.append(float(variables[key]))
            known_var_idx.append(int(key.replace('x', '')) - 1)
        known_var_idx = np.array(known_var_idx, dtype=np.int16)
        shift_idx = np.array([False] * self.dim)
        shift_idx[known_var_idx.tolist()] = True

        value_var = np.array(value_var)

        if (self.dim - known_var_idx.max()) < 0:
            print('The highest index of known variables is higher than the dimension of the Model')

        if (self.dim - len(known_var_idx)) < 0:
            print('The number of conditional variables is lower than the dimension of the Model')

        for index, timeStep in enumerate(known_var_idx):
            # Check the lower bound of the interpolation model
            if self.interpolation == 'linear':
                lower_bound_model = self.ecdf[timeStep].x.min()
                upper_bound_model = self.ecdf[timeStep].x.max()
            else:
                lower_bound_model = self.ecdf[timeStep][0].min()
                upper_bound_model = self.ecdf[timeStep][0].max()

            if value_var[index] < lower_bound_model:
                print('Variable: {} is out of range.'.format(timeStep))
                print('Minimum value to use should be: {:.3f}'.format(self.ecdf[timeStep].x.min()))
                return False
            elif value_var[index] > upper_bound_model:
                print('Variable: {} is out of range.'.format(timeStep))
                print('Maximum value to use should be: {.3f}'.format(self.ecdf[timeStep].x.max()))
                return False

        return True

    def _check_clean_data_frame(self, data_frame):
        """
        Sanity checks about the dimensions and type of the object data_frame. Also, cleans for the the NaNs values
        in the 2-D Matrix
        """
        assert (len(data_frame.shape) == 2),  "The matrix should be 2-D"
        assert (isinstance(data_frame, np.ndarray)), "The matrix should be a numpy array"

        # Drop the instance that at least has one 'nan' value
        idx = np.isnan(data_frame).any(axis=0)

        print('*******************************************************************************************************')
        print('------------------------------------')
        print('SUMMARY OF DATA INPUT')
        print('------------------------------------')
        print(f'Number of variables: {data_frame.shape[0]}')
        print(f'Number of samples: {data_frame.shape[1]}')
        print(f'Samples deleted because of "NaNs": {np.sum(idx)}')
        print('\n')

        self.data_frame = data_frame[:, ~idx].copy()
        self.sample_size = self.data_frame.shape[1]

    def _plot_pit(self, variable):
        if len(variable) == 1:
            utils.probability_integral_transform(data=self.data_frame,
                                                 plot=True,
                                                 variable=variable,
                                                 interpolation=self.interpolation)
        else:
            for variable_number in variable:
                utils.probability_integral_transform(data=self.data_frame,
                                                     plot=True,
                                                     variable=variable_number,
                                                     interpolation=self.interpolation)


# if __name__ == '__main__':
#     #%% 2-Variables - t distribution
#     n_samples_ = 5000
#     covariance_ = np.array([[1, -0.6], [-0.6, 1]])
#     # mean_ = np.zeros((2, 1))
#     mean_ = np.array([[3], [4]])
#     nu_ = 8
#     conditioned_variable_ = {'x2': 5}
#
#     samples_t = utils.samples_multivariate_t(mean=mean_, covariance=covariance_, nu=nu_, n_samples=n_samples_)
#
#     ax1 = utils.plot_samples(samples_t)
#     ax1.set_title('Original Dataset')
#
#     copula_t = EllipticalCopula(samples_t,  copula_type='t')
#     copula_t.fit(plot_linear_search=False, x0=5)
#     samples_copula_t = copula_t.sample(n_samples=n_samples_)
#     samples_copula_t_conditioned = copula_t.sample(n_samples=n_samples_,
#                                                    conditional=True,
#                                                    variables=conditioned_variable_)
#     ax = utils.plot_samples(samples_copula_t)
#     ax.set_title('Sampled Copula')
#     ax.set_xlim(ax1.get_xlim())
#     ax.set_ylim(ax1.get_ylim())
#
#     (mean_cond,
#      cov_cond,
#      nu_cond,
#      _) = utils.conditional_parameters(dim=2,
#                                        mean_vector=mean_,
#                                        covariance_kendall=covariance_,
#                                        nu=nu_,
#                                        copula_type='t',
#                                        variables=conditioned_variable_)
#
#     ax1 = utils.plot_samples(t(df=nu_cond,
#                                loc=mean_cond.ravel(),
#                                scale=cov_cond.ravel()).rvs(n_samples_).reshape(1, -1))
#     ax1.set_title('Conditioned - Original Dataset')
#
#     ax = utils.plot_samples(samples_copula_t_conditioned)
#     ax.set_title('Conditioned - Sampled Copula')
#     ax.set_xlim(ax1.get_xlim())
#     ax.set_ylim(ax1.get_ylim())
#
#     #%% 3-Variables - t - distribution
#     import tensorflow.compat.v2 as tf
#     import tensorflow_probability as tfp
#     tf.enable_v2_behavior()
#
#     n_samples_ = 5000
#     tfd = tfp.distributions
#     df = 18.
#     loc = [1., 2, 3]
#     scale = [[0.6, 0., 0.],
#              [0.2, 0.5, 0.],
#              [0.1, -0.3, 0.4]]
#     dim_ = 3
#     conditioned_variable_ = {'x3': 3.5}
#
#     # scale = np.linalg.cholesky(scale)  # Be aware that the tensorflow function accept cholesky of sigma.
#
#     mvt = tfd.MultivariateStudentTLinearOperator(
#         df=df,
#         loc=loc,
#         scale=tf.linalg.LinearOperatorLowerTriangular(scale))
#     sample_t_dist = tfd.Sample(mvt, sample_shape=(n_samples_))
#     samples = sample_t_dist.sample()
#     samples = np.array(samples).T  # dimensions: [dim x samples] or [variables x instances]
#
#     ax1 = utils.plot_samples(samples)
#     ax1.set_title('Original Dataset')
#     ax1.set_xlabel('x')
#     ax1.set_ylabel('y')
#     ax1.set_zlabel('z')
#
#     sigma_hat = tf.matmul(scale, scale, adjoint_b=True)
#
#     print('\n\nSigma: ')
#     print(sigma_hat)
#     print('\n\nCovariance (which is (nu / (nu - 2)) * sigma: ')
#     print(f'Multiplier: {(df / (df - 2))}\n')
#     print(mvt.covariance())
#
#     copula_t = EllipticalCopula(samples, copula_type='t')
#     copula_t.fit(plot_linear_search=False, x0=5)
#     self = copula_t
#     samples_copula_t = copula_t.sample(n_samples=n_samples_)
#     samples_copula_t_conditioned = copula_t.sample(n_samples=n_samples_,
#                                                    conditional=True,
#                                                    variables=conditioned_variable_)
#
#     ax = utils.plot_samples(samples_copula_t)
#     ax.set_title('Sampled Dataset')
#     ax.set_xlabel('x')
#     ax.set_ylabel('y')
#     ax.set_zlabel('z')
#     ax.set_xlim(ax1.get_xlim())
#     ax.set_ylim(ax1.get_ylim())
#     ax.set_zlim(ax1.get_zlim())
#
#     (mean_cond,
#      cov_cond,
#      nu_cond,
#      _) = utils.conditional_parameters(dim=3,
#                                        mean_vector=np.array(loc).reshape(-1, 1),
#                                        covariance_kendall=np.array(sigma_hat),
#                                        nu=df,
#                                        copula_type='t',
#                                        variables=conditioned_variable_)
#
#     original_conditioned = utils.samples_multivariate_t(mean=mean_cond,
#                                                         covariance=cov_cond,
#                                                         nu=nu_cond,
#                                                         n_samples=n_samples_)
#     ax1 = utils.plot_samples(original_conditioned)
#     ax1.set_title('Conditioned - Original Dataset')
#
#     ax = utils.plot_samples(samples_copula_t_conditioned)
#     ax.set_title('Conditioned - Sampled Copula')
#     ax.set_xlim(ax1.get_xlim())
#     ax.set_ylim(ax1.get_ylim())
#
#     ks_2samp(original_conditioned[1,:], samples_copula_t_conditioned[1,:])
#     wasserstein_distance(original_conditioned[0,:], samples_copula_t_conditioned[0,:])
#
#
#     #%% 2-Variables - Gaussian distribution
#     n_samples_ = 5000
#     covariance_ = np.array([[1, -0.6], [-0.6, 1]])
#     # mean_ = np.zeros((2, 1))
#     mean_ = np.array([[3], [4]])
#     nu_ = 8
#     conditioned_variable_ = {'x2': 5.5}
#
#     samples_gauss = multivariate_normal(mean_.squeeze(), covariance_).rvs(n_samples_).T
#     # ks_2samp(samples_t[0, :], samples_gauss[0, :])
#
#     ax1 = utils.plot_samples(samples_gauss)
#     ax1.set_title('Original Dataset - Gaussian distribution')
#
#     copula_gauss = EllipticalCopula(samples_gauss, copula_type='gaussian')
#     copula_gauss.fit(plot_linear_search=False)
#     samples_copula_gaussian = copula_gauss.sample(n_samples=n_samples_)
#     samples_copula_gaussian_conditioned = copula_gauss.sample(n_samples=n_samples_,
#                                                               conditional=True,
#                                                               variables=conditioned_variable_)
#     ax = utils.plot_samples(samples_copula_gaussian)
#     ax.set_title('Sampled Copula - Gaussian distribution')
#     ax.set_xlim(ax1.get_xlim())
#     ax.set_ylim(ax1.get_ylim())
#
#     (mean_cond,
#      cov_cond,
#      nu_cond,
#      _) = utils.conditional_parameters(dim=2,
#                                        mean_vector=mean_,
#                                        covariance_kendall=covariance_,
#                                        nu=nu_,
#                                        copula_type='gaussian',
#                                        variables=conditioned_variable_)
#
#     original_conditioned = norm(loc=mean_cond.ravel(), scale=cov_cond.ravel()).rvs(n_samples_).reshape(1, -1)
#     ax1 = utils.plot_samples(original_conditioned)
#     ax1.set_title('Conditioned - Original Dataset')
#
#     ax = utils.plot_samples(samples_copula_gaussian_conditioned)
#     ax.set_title('Conditioned - Sampled Copula')
#     ax.set_xlim(ax1.get_xlim())
#     ax.set_ylim(ax1.get_ylim())
#
#     ks, p = ks_2samp(original_conditioned.ravel(), samples_copula_gaussian_conditioned.ravel())
#     print(f'Kolgomorov-Smirnoff (2-Sample),  p-value: {p}')
#     print(f'Kolgomorov-Smirnoff (2-Sample),  KS distance: {ks}')
#
#     W, p = shapiro(original_conditioned.ravel())
#     print(f'Original conditioned - Shapiro-Wilk,  p-value: {p}')
#     W, p = shapiro(samples_copula_gaussian_conditioned.ravel())
#     print(f'Sampled conditioned - Shapiro-Wilk,  p-value: {p}')
#
#
#     # ks, p = ks_2samp(norm(loc=mean_cond.ravel(), scale=cov_cond.ravel()).rvs(int(n_samples_/100)).reshape(1, -1).ravel(),
#     #                  norm(loc=mean_cond.ravel(), scale=cov_cond.ravel()).rvs(int(n_samples_/100)).reshape(1, -1).ravel())
#     # print(f'p-value: {p}')
#     # print(f'KS: {ks}')
#     #
#     # W, p = shapiro(norm(loc=mean_cond.ravel(), scale=cov_cond.ravel()).rvs(int(n_samples_/100)).reshape(1, -1).ravel())
#     # print(f'Shapiro-Wilk,  p-value: {p}')
#
#     # %% 3-Variables - Gaussian distribution
#     n_samples_ = 5000
#     covariance_ = np.array([[   1, -0.6,  0.7],
#                             [-0.6,    1, -0.4],
#                             [ 0.7,  -0.4,   1]])
#     # mean_ = np.zeros((2, 1))
#     mean_ = np.array([[1], [3], [4]])
#     nu_ = 8
#     df = None
#     # conditioned_variable_ = {'x2': 5.5}  # This will create artifacts because is in the lower side of interpolation
#     conditioned_variable_ = {'x3': 5.5}
#
#     samples_gauss = multivariate_normal(mean_.squeeze(), covariance_).rvs(n_samples_).T
#
#     ax1 = utils.plot_samples(samples_gauss)
#     ax1.set_title('Original Dataset - Gaussian distribution')
#     ax1.set_xlabel('x')
#     ax1.set_ylabel('y')
#     ax1.set_zlabel('z')
#
#     copula_gauss = EllipticalCopula(samples_gauss, copula_type='gaussian', interpolation='spline')
#     copula_gauss.fit(plot_linear_search=False)
#     # copula_gauss._plot_pit(variable=[0, 1, 2])
#     samples_copula_gaussian = copula_gauss.sample(n_samples=n_samples_, negative_clamp=False)
#     samples_copula_gaussian_conditioned = copula_gauss.sample(n_samples=n_samples_, negative_clamp=False,
#                                                               conditional=True, variables=conditioned_variable_)
#     ax = utils.plot_samples(samples_copula_gaussian)
#     ax.set_title('Sampled Copula - Gaussian distribution')
#     ax.set_xlabel('x')
#     ax.set_ylabel('y')
#     ax.set_zlabel('z')
#     ax.set_xlim(ax1.get_xlim())
#     ax.set_ylim(ax1.get_ylim())
#     ax.set_zlim(ax1.get_zlim())
#
#     (mean_cond,
#      cov_cond,
#      nu_cond,
#      _) = utils.conditional_parameters(dim=3,
#                                        mean_vector=mean_,
#                                        covariance_kendall=covariance_,
#                                        nu=df,
#                                        copula_type='gaussian',
#                                        variables=conditioned_variable_)
#
#     original_conditioned =multivariate_normal(mean_cond.squeeze(), cov_cond).rvs(n_samples_).T
#
#     ax1 = utils.plot_samples(original_conditioned)
#     ax1.set_title('Conditioned - Original Dataset')
#
#     ax = utils.plot_samples(samples_copula_gaussian_conditioned)
#     ax.set_title('Conditioned - Sampled Copula')
#     ax.set_xlim(ax1.get_xlim())
#     ax.set_ylim(ax1.get_ylim())
#
#     ks_2samp(original_conditioned[0, :], samples_copula_gaussian_conditioned[0, :])
#     ks_2samp(original_conditioned[1, :], samples_copula_gaussian_conditioned[1, :])
#     wasserstein_distance(original_conditioned[0, :], samples_copula_gaussian_conditioned[0, :])
#
#     # %% 3-Variables - Gaussian distribution - BANANA PLOT
#     n_samples_ = 10000
#     covariance_ = np.array([[1, -0.6, 0.7],
#                             [-0.6, 1, -0.4],
#                             [0.7, -0.4, 1]])
#     # mean_ = np.zeros((2, 1))
#     mean_ = np.array([[1], [3], [4]])
#     nu_ = 8
#     df = None
#     conditioned_variable_ = {'x3': np.log(4)}
#
#     samples_gauss = multivariate_normal(mean_.squeeze(), covariance_).rvs(n_samples_).T
#     samples_gauss = np.log(samples_gauss)
#
#     ax1 = utils.plot_samples(samples_gauss)
#     ax1.set_title('Original Dataset - Gaussian distribution')
#     ax1.set_xlabel('x')
#     ax1.set_ylabel('y')
#     ax1.set_zlabel('z')
#
#     copula_gauss = EllipticalCopula(samples_gauss, copula_type='gaussian', interpolation='spline')
#     copula_gauss.fit(plot_linear_search=False)
#     # copula_gauss._plot_pit(variable=[0, 1, 2])
#     samples_copula_gaussian = copula_gauss.sample(n_samples=n_samples_, negative_clamp=False)
#     samples_copula_gaussian_conditioned = copula_gauss.sample(n_samples=n_samples_, negative_clamp=False,
#                                                               conditional=True, variables=conditioned_variable_)
#     ax = utils.plot_samples(samples_copula_gaussian)
#     ax.set_title('Sampled Copula - Gaussian distribution')
#     ax.set_xlabel('x')
#     ax.set_ylabel('y')
#     ax.set_zlabel('z')
#     ax.set_xlim(ax1.get_xlim())
#     ax.set_ylim(ax1.get_ylim())
#     ax.set_zlim(ax1.get_zlim())
#
#     conditioned_variable_ = {'x3': 4}
#     (mean_cond,
#      cov_cond,
#      nu_cond,
#      _) = utils.conditional_parameters(dim=3,
#                                        mean_vector=mean_,
#                                        covariance_kendall=covariance_,
#                                        nu=df,
#                                        copula_type='gaussian',
#                                        variables=conditioned_variable_)
#
#     original_conditioned = multivariate_normal(mean_cond.squeeze(), cov_cond).rvs(n_samples_).T
#
#     original_conditioned[0, :] = np.log(original_conditioned[0, :])
#     original_conditioned[1, :] = np.log(original_conditioned[1, :])
#
#     cleaned_1 = original_conditioned[0, ~np.isnan(original_conditioned[0, :])]
#     cleaned_2 = original_conditioned[1, ~np.isnan(original_conditioned[1, :])]
#
#     ax1 = utils.plot_samples(original_conditioned)
#     ax1.set_title('Conditioned - Original Dataset')
#
#     ax = utils.plot_samples(samples_copula_gaussian_conditioned)
#     ax.set_title('Conditioned - Sampled Copula')
#     ax.set_xlim(ax1.get_xlim())
#     ax.set_ylim(ax1.get_ylim())
#
#     cleaned_1_copula = samples_copula_gaussian_conditioned[0,   ~np.isnan(samples_copula_gaussian_conditioned[0, :])]
#     cleaned_2_copula = samples_copula_gaussian_conditioned[1, ~np.isnan(samples_copula_gaussian_conditioned[1, :])]
#
#     ks_2samp(cleaned_1, cleaned_1_copula)
#     ks_2samp(cleaned_2, cleaned_2_copula)
#     wasserstein_distance(cleaned_1, cleaned_1_copula)

