"""
# Multi-Fidelity Co-Kriging
Assuming that each level of fidelity is not a subset of the previous.
Mauricio CASTANO AGUIRRE
ONERA- UPHF
2024
"""
#import warnings

import numpy as np
from scipy.linalg import solve_triangular
from scipy import optimize
from smt.sampling_methods import LHS
from smt.utils.kriging import (differences, componentwise_distance)
from smt.surrogate_models.krg_based import KrgBased

class MFCK(KrgBased):
    def _initialize(self):
        super()._initialize()
        declare = self.options.declare
        self.name = "MFCK"
        declare(
            "sigma_bounds",
            [1e-1, 100],
            types=(list, np.ndarray),
            desc="bounds for variance hyperparameters",
        )
        declare(
            "rho_bounds",
            [-5., 5.],
            types=(list, np.ndarray),
            desc="bounds for rho value, autoregressive model",
        )
        declare(
            "rho0",
            1.5,
            types=(float),
            desc="Power for the pow_exp kernel function (valid values in (0.0, 2.0]). \
                This option is set automatically when corr option is squar, abs, or matern.",
        )
        declare(
            "sigma0",
            1.0,
            types=(float),
            desc="Power for the pow_exp kernel function (valid values in (0.0, 2.0]). \
                This option is set automatically when corr option is squar, abs, or matern.",
        )
        self.params = {}
        self.K = None
        self.optimal_theta = None
        self.lvl = None
        self.X = []
        self.y = []

    def train(self):
        """
        Overrides MFK implementation
        Trains the Multi-Fidelity co-Kriging model
        Returns
        -------
        None.
        """
        xt = []
        yt = []
        i = 0
        while self.training_points.get(i, None) is not None:
            xt.append(self.training_points[i][0][0])
            yt.append(self.training_points[i][0][1])
            i = i + 1
        xt.append(self.training_points[None][0][0])
        yt.append(self.training_points[None][0][1])
        self.lvl = i+1
        self.X = xt
        self.y = np.vstack(yt)
        self._check_param()
        self.nx=1
        if self.lvl == 1:
            # For a single level, initialize theta_ini, lower_bounds, and upper_bounds with consistent shapes
            theta_ini = np.hstack((self.options["sigma0"], self.options["theta0"][0]))  # Kernel variance + theta0
            lower_bounds = np.hstack((self.options["sigma_bounds"][0], self.options["theta_bounds"][0]))
            upper_bounds = np.hstack((self.options["sigma_bounds"][1], self.options["theta_bounds"][1]))
            theta_ini = np.log10(theta_ini)
            lower_bounds = np.log10(lower_bounds)
            upper_bounds = np.log10(upper_bounds)
            x_opt=theta_ini
        else:
            for lvl in range(self.lvl):
                if lvl == 0:
                    # Initialize theta_ini for level 0
                    theta_ini = np.hstack((self.options["sigma0"], self.options["theta0"][0]))  # Variance + initial theta values
                    lower_bounds = np.hstack((self.options["sigma_bounds"][0],
                                              np.full(self.nx, self.options["theta_bounds"][0])))
                    upper_bounds = np.hstack((self.options["sigma_bounds"][1],
                                              np.full(self.nx, self.options["theta_bounds"][1])))
                    # Apply log10 to theta_ini and bounds
                    theta_ini[:len(self.options["theta0"])+1]=np.log10(theta_ini[:len(self.options["theta0"])+1])
                    lower_bounds[:len(self.options["theta0"])+1]=np.log10(lower_bounds[:len(self.options["theta0"])+1])
                    upper_bounds[:len(self.options["theta0"])+1]=np.log10(upper_bounds[:len(self.options["theta0"])+1])

                elif lvl > 0:
                    # For additional levels, append to theta_ini, lower_bounds, and upper_bounds
                    thetat = np.hstack((self.options["sigma0"], self.options["theta0"][0]))  # Variance + theta0
                    lower_boundst = np.hstack((self.options["sigma_bounds"][0],
                                               np.full(self.nx, self.options["theta_bounds"][0])))
                    upper_boundst = np.hstack((self.options["sigma_bounds"][1],
                                               np.full(self.nx, self.options["theta_bounds"][1])))
                    # Apply log10 to the newly added values
                    thetat = np.log10(thetat)
                    lower_boundst = np.log10(lower_boundst)
                    upper_boundst = np.log10(upper_boundst)
                    # Append to theta_ini, lower_bounds, and upper_bounds
                    theta_ini = np.hstack([theta_ini, thetat,self.options["rho0"]])
                    lower_bounds = np.hstack([lower_bounds, lower_boundst])
                    upper_bounds = np.hstack([upper_bounds, upper_boundst])
                    # Finally, append the rho bounds
                    lower_bounds = np.hstack([lower_bounds, self.options["rho_bounds"][0]])
                    upper_bounds = np.hstack([upper_bounds, self.options["rho_bounds"][1]])
        theta_ini = theta_ini[:].T
        x_opt= theta_ini
        if self.options["hyper_opt"] == "Cobyla":
            if self.options["n_start"] > 1:
                sampling = LHS(
                    xlimits = np.stack((lower_bounds,upper_bounds),axis=1),
                    criterion="maximin",
                    random_state=0,
                )
                theta_lhs_loops = sampling(self.options["n_start"])
                theta0 = np.vstack((theta_ini, theta_lhs_loops))
            constraints=[]
            for i in range(len(theta_ini)):
                constraints.append(lambda theta0, i=i: theta0[i] - lower_bounds[i])
                constraints.append(lambda theta0, i=i: upper_bounds[i] - theta0[i])
            for j in range(self.options["n_start"]):
                optimal_theta_res_loop = optimize.minimize(
                    self.neg_log_likelihooda,
                    theta0[j,:],
                    method="COBYLA",
                    constraints=[
                        {"fun": con, "type": "ineq"} for con in constraints
                    ],
                    options={
                        "rhobeg": 0.2,
                        "tol": 1e-6,
                        "maxiter": 500,
                    },
                )
                x_opt_iter = optimal_theta_res_loop.x
                if j==0:
                    x_opt=x_opt_iter
                    nll=optimal_theta_res_loop["fun"]
                else:
                    if optimal_theta_res_loop["fun"] < nll:
                        x_opt=x_opt_iter
                        nll=optimal_theta_res_loop["fun"]
        elif self.options["hyper_opt"]=="MMA":        
            try:
                import nlopt
            except ImportError:
                print("nlopt library is not installed or available on this system")
                
            opt = nlopt.opt(nlopt.LN_COBYLA, theta_ini.shape[0])
            opt.set_lower_bounds(lower_bounds)  # Lower bounds for each dimension
            opt.set_upper_bounds(upper_bounds)  # Upper bounds for each dimension
            opt.set_min_objective(self.neg_log_likelihoodb)
            opt.set_maxeval(2000)
            opt.set_xtol_rel(1e-6)
            x0 = np.copy(theta_ini) 
            x_opt = opt.optimize(x0)
            
        x_opt[0:2]=10**(x_opt[0:2])
        x_opt[2::3]=10**(x_opt[2:8:3])
        x_opt[3::3]=10**(x_opt[3:8:3])
        self.optimal_theta = x_opt
        self.K = self.compute_block_wise_K(self.optimal_theta)
    
    def eta(self,j, l, rho):
        """Compute eta_{j,l} based on the given rho values."""
        if j < l:
            return np.prod(rho[j:l])  # Product of rho[j+1] to rho[l]
        elif j == l:
            return 1
        else:
            return 0  # Should not occur, as j <= l' in the main covariance expression
    
    # Covariance between y_l(x) and y_l'(x')
    def covariance_yl_ylprime(self,x, x_prime, l, l_prime,param):
        """
        Calculation Cov(y_l(x), y_{l'}(x')) using the autoregressive formulation.
        
        Parmeters:
        - x: First input for the covariannce (vector)
        - x_prime: Second input for the covariannce (vector)
        - l: Index of the first output
        - l_prime: Index of the second output
        - param: Set of Hyper-parameters
        
        Returns:
        - Value of caovariance between y_l(x) and y_{l'}(x')
        """
        cov_value = 0.0
        
        param0 = param[0:2]
        sigmas_gamma = param[2::3]
        ls_gamma = param[3::3]
        rho_values = param[4::3]
        
        # Sum of j=0 until l_^prime
        for j in range(l_prime + 1):
            eta_j_l = self.eta(j, l, rho_values)
            eta_j_lprime = self.eta(j, l_prime, rho_values)
            
            if j==0:
                cov_gamma_j = self._compute_K(x, x_prime,param0)
            else:
                # Cov(γ_j(x), γ_j(x')) using the kernel
                cov_gamma_j = self._compute_K(x, x_prime,[sigmas_gamma[j-1],ls_gamma[j-1]])
            
            # Add to the value of the covariance
            cov_value += eta_j_l * eta_j_lprime * cov_gamma_j
        return cov_value
    
    def predict_all_levels(self,x):
        """
        Generalized prediction function for the multi-fidelity co-Kriging
        Parameters
        ----------
        x : np.ndarray
            Array with the inputs for make the prediction.
        Returns
        -------
        means : (list, np.array)
            Returns the conditional means per level.
        covariances: (list, np.array)
            Returns the conditional covariance matrixes per level.
        """
        means=[]
        covariances=[]
        if self.lvl==1:
            k_XX = self._compute_K(self.X[0],self.X[0],self.optimal_theta[0:2])
            k_xX = self._compute_K(x,self.X[0],self.optimal_theta[0:2])
            k_xx = self._compute_K(x,x,self.optimal_theta[0:2])
            means.append( np.dot(k_xX, np.matmul(np.linalg.inv(k_XX + 1e-9*np.eye(k_XX.shape[0])), self.y)))
            covariances.append(k_xx - np.matmul(k_xX,
                                                np.matmul(np.linalg.inv(k_XX+1e-9*np.eye(k_XX.shape[0])),
                                                          k_xX.transpose())))
        else:
            nugget = self.options["nugget"]
            L = np.linalg.cholesky(self.K+ nugget*np.eye(self.K.shape[0]))
            k_xX = []
            for ind in range(self.lvl):
                k_xx = self.covariance_yl_ylprime(x, x, ind, ind,self.optimal_theta)
                for j in range(self.lvl):  
                    if ind >= j:
                        k_xX.append(self.covariance_yl_ylprime(self.X[j], x, ind, j,self.optimal_theta))    
                    else:
                        k_xX.append(self.covariance_yl_ylprime(self.X[j], x, j, ind,self.optimal_theta)) 
                beta1 = solve_triangular(L, np.vstack(k_xX),lower=True)
                alpha1 = solve_triangular(L,self.y,lower=True)
                means.append( np.dot(beta1.T,alpha1) )
                covariances.append(  k_xx - np.dot(beta1.T,beta1) )
                k_xX.clear()
        return means,covariances

    def _predict(self,x):
        """
        Prediction function for the highest fidelity level
        Parameters
        ----------
        x : array
            Array with the inputs for make the prediction.
        Returns
        -------
        mean : np.array
            Returns the conditional means per level.
        covariance: np.ndarray
            Returns the conditional covariance matrixes per level.
        """
        means,covariances=self.predict_all_levels(x)
        return means[self.lvl-1],covariances[self.lvl-1]

    def neg_log_likelihood(self,param1,grad):
        
        if self.lvl == 1:
            self.K=self._compute_K(self.X[0],self.X[0],param1[0:2])
        else:
            self.K = self.compute_block_wise_K(param1)
        nugget = 1e-4#self.options["nugget"]#small number to ensure numerical stability.
        
        L = np.linalg.cholesky(self.K+ nugget*np.eye(self.K.shape[0]))
        beta = solve_triangular(L, self.y,lower=True)
        N=np.shape(self.y)[0]
        NMLL=1/2*(2*np.sum(np.log(np.diag(L)))+np.dot(beta.T,beta)+N*np.log(2*np.pi))
        nmll,=NMLL[0]
        return nmll

    def neg_log_likelihooda(self,param1):
        """
        Likelihood for Cobyla optimizer
        """
        param1 = np.array(param1, copy=True)
        param1[0:2]=10**(param1[0:2])
        param1[2::3]=10**(param1[2:8:3])
        param1[3::3]=10**(param1[3:8:3])
        return self.neg_log_likelihood(param1,1)
    
    def neg_log_likelihoodb(self,param1,grad):
        """
        Likelihood for nlopt optimizers
        """
        param1 = np.array(param1, copy=True)
        param1[0:2]=10**(param1[0:2])
        param1[2::3]=10**(param1[2:8:3])
        param1[3::3]=10**(param1[3:8:3])
        return self.neg_log_likelihood(param1,1)

    def compute_block_wise_K(self, param1):
        """
        Compute the co-kriging piece-wise matrix with correct handling of non-symmetric cross-correlations.
        Parameters
        ----------
        param1 : array
            Array with the hyperparameters for the co-kriging model.
        Returns
        -------
        K : np.ndarray
            The piece-wise matrix of the co-kriging model.
        """
        # Extract cross-level correlation coefficients
        rhos = param1[4::3]
        param0 = param1[0:2]  # Hyperparameters for the first level
        Kb={}
        n = self.y.shape[0]
        #Precompute K matrices to avoid redundant calculations
        precomputed_K = {}
        def get_K(lvl, lvl1, params):
            """Helper function to compute or retrieve precomputed K matrices for (lvl, lvl1)."""
            # Convert params to a tuple if it's a numpy array
            if isinstance(params, np.ndarray):
                params = tuple(params)
            key = (lvl, lvl1, params)
            if key not in precomputed_K:
                precomputed_K[key] = self._compute_K(self.X[lvl], self.X[lvl1], params)
            return precomputed_K[key]
        # Fill K_var matrix
        for lvel in range(self.lvl):
            if lvel == 0:
                Kb[str(lvel)+str(lvel)]= get_K(lvel , lvel, param0)
                for i in range (self.lvl-1):
                    a = str(lvel)
                    b=str(i+1)
                    Kb[a+b]= np.prod(rhos[:i+1])*get_K(lvel, i+1, param0)
            else:
                param = param1[3 * lvel - 1: 3 * lvel + 2]
                param_ant = param1[3 * (lvel - 1) - 1: 3 * (lvel - 1) + 2] if lvel > 1 else None
                k1 = get_K(lvel,lvel, param0[:2])
                k2 = get_K(lvel,lvel, param_ant[:-1]) if param_ant is not None else None
                k3 = get_K(lvel,lvel, param[:-1])
                if lvel == 1:

                    Kb[str(lvel)+str(lvel)]=param[-1]**2 * k1 + k3
                    for i in range (lvel,self.lvl-1):
                        a = str(lvel)
                        b=str(i+1)
                        Kb[a+b]=(np.prod(rhos[lvel:i+1])*(param[-1]**2 * get_K(lvel,i+1, param0[:2])+
                                                          get_K(lvel,i+1, param[:-1])))
                else:
                    Var_ant = param_ant[-1] ** 2 * k1 + (k2 if k2 is not None else 0)
                    Kb[str(lvel)+str(lvel)]=param[-1] ** 2 * Var_ant + k3
                    for i in range (lvel,self.lvl-1):
                        a = str(lvel)
                        b=str(i+1)
                        temp=(param_ant[-1] ** 2 * get_K(lvel,i+1, param0[:2])+
                        (get_K(lvel,i+1, param_ant[:-1]) if k2 is not None else 0))
                        Kb[a+b]=np.prod(rhos[lvel:i+1])*(param[-1]**2 * temp + get_K(lvel,i+1, param[:-1]))
        K = np.zeros((n, n))
        row1, col1 = 0, 0
        for i in range(self.lvl):
            col1 = row1
            for j in range(i, self.lvl):
                r, c = Kb[str(i)+str(j)].shape
                K[row1:row1+r, col1:col1+c] = Kb[str(i)+str(j)]
                if i != j:
                    K[col1:col1+c, row1:row1+r] = Kb[str(i)+str(j)].T
                col1 += c
            row1 += r
        return K

    def _compute_K(self, A: np.ndarray, B: np.ndarray, param):
        """
        Compute the covariance matrix K between A and B
            Modified for MFCK initial test (Same theta for each dimmension)
        """
        # Compute pairwise componentwise L1-distances between A and B
        dx = differences(A, B)
        d = componentwise_distance(dx,self.options["corr"],self.X[0].shape[1],power=self.options["pow_exp_power"])
        self.corr.theta=np.full(self.X[0].shape[1],param[1])
        r = self.corr(d)
        R = r.reshape(A.shape[0], B.shape[0])
        K = param[0] * R
        return K
