"""
# Multi-Fidelity Co-Kriging
Assuming that each level of fidelity is not a subset of the previous.
Mauricio CASTANO AGUIRRE
ONERA- UPHF
2024
"""
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
            [1e-2, 100],
            types=(list, np.ndarray),
            desc="bounds for variance hyperparameters",
        )
        declare(
            "rho_bounds",
            [-5., 5.],
            types=(list, np.ndarray),
            desc="bounds for rho value, autoregressive model",
        )
        self.params = {}
        self.K = None
        self.theta = None
        self.lvl = None
        self.X = []
        self.y = []

    def train(self):
        """
        Function for train the Hyper-parameters of the MFCK model
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
        self.options["theta0"]=[1.0]
        if self.lvl == 1:
            # For a single level, initialize theta_ini, lower_bounds, and upper_bounds with consistent shapes
            theta_ini = np.hstack((1.0, self.options["theta0"]))  # Kernel variance + theta0
            lower_bounds = np.hstack((self.options["sigma_bounds"][0], self.options["theta_bounds"][0]))
            upper_bounds = np.hstack((self.options["sigma_bounds"][1], self.options["theta_bounds"][1]))
            theta_ini = np.log10(theta_ini)
            lower_bounds = np.log10(lower_bounds)
            upper_bounds = np.log10(upper_bounds)
        else:
            for lvel in range(self.lvl):
                if lvel == 0:
                    # Initialize theta_ini for level 0
                    theta_ini = np.hstack((1.0, self.options["theta0"]))  # Variance + initial theta values
                    lower_bounds = np.hstack((self.options["sigma_bounds"][0],
                                              np.full(self.nx, self.options["theta_bounds"][0])))
                    upper_bounds = np.hstack((self.options["sigma_bounds"][1],
                                              np.full(self.nx, self.options["theta_bounds"][1])))
                    # Apply log10 to theta_ini and bounds
                    theta_ini[:len(self.options["theta0"])+1]=np.log10(theta_ini[:len(self.options["theta0"])+1])
                    lower_bounds[:len(self.options["theta0"])+1]=np.log10(lower_bounds[:len(self.options["theta0"])+1])
                    upper_bounds[:len(self.options["theta0"])+1]=np.log10(upper_bounds[:len(self.options["theta0"])+1])

                elif lvel > 0:
                    # For additional levels, append to theta_ini, lower_bounds, and upper_bounds
                    thetat = np.hstack((1.0, self.options["theta0"]))  # Variance + theta0
                    lower_boundst = np.hstack((self.options["sigma_bounds"][0],
                                               np.full(self.nx, self.options["theta_bounds"][0])))
                    upper_boundst = np.hstack((self.options["sigma_bounds"][1],
                                               np.full(self.nx, self.options["theta_bounds"][1])))
                    # Apply log10 to the newly added values
                    thetat = np.log10(thetat)
                    lower_boundst = np.log10(lower_boundst)
                    upper_boundst = np.log10(upper_boundst)
                    # Append to theta_ini, lower_bounds, and upper_bounds
                    theta_ini = np.hstack([theta_ini, thetat,1.0])
                    lower_bounds = np.hstack([lower_bounds, lower_boundst])
                    upper_bounds = np.hstack([upper_bounds, upper_boundst])
                    # Finally, append the rho bounds
                    lower_bounds = np.hstack([lower_bounds, self.options["rho_bounds"][0]])
                    upper_bounds = np.hstack([upper_bounds, self.options["rho_bounds"][1]])
        theta_ini = theta_ini[:].T
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
                        "rhobeg": 0.5,
                        "tol": 1e-6,
                        "maxiter": 100,
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
        elif self.options["hyper_opt"]=="TNC":
            if self.options["n_start"] > 1:
                sampling = LHS(
                    xlimits = np.stack((lower_bounds,upper_bounds),axis=1),
                    criterion="maximin",
                    random_state=0,
                )
                theta_lhs_loops = sampling(self.options["n_start"])
                theta0 = np.vstack((theta_ini, theta_lhs_loops))
            theta0[:,:(len(self.options["theta0"]))] = np.log10(theta0[:,:(len(self.options["theta0"]))])
            lower_bounds[:(len(lower_bounds))]=np.log10(lower_bounds[:(len(lower_bounds))])
            upper_bounds[:(len(upper_bounds))]=np.log10(upper_bounds[:(len(upper_bounds))])
            constraints=[]
            for i in range(len(theta_ini)):
                constraints.append(lambda theta0, i=i: theta0[i] - lower_bounds[i])
                constraints.append(lambda theta0, i=i: upper_bounds[i] - theta0[i])
            for j in range(self.options["n_start"]):
                optimal_theta_res_loop = optimize.minimize(
                    self.neg_log_likelihooda,
                    theta0[j,:],
                    method="TNC",
                    jac=self.log_likelihood_gradient
                )
                x_opt_iter = optimal_theta_res_loop.x
                x_opt_iter[:(len(x_opt_iter)-1)] = 10**(x_opt_iter[:(len(x_opt_iter)-1)])
                if j==0:
                    x_opt=x_opt_iter
                    nll=optimal_theta_res_loop["fun"]
                else:
                    if optimal_theta_res_loop["fun"] < nll:
                        x_opt=x_opt_iter
                        nll=optimal_theta_res_loop["fun"]
        else:
            if self.options["n_start"] > 1:
                sampling = LHS(
                    xlimits = np.stack((lower_bounds,upper_bounds),axis=1),
                    criterion="maximin",
                    random_state=0,
                )
                theta_lhs_loops = sampling(self.options["n_start"])
                theta0 = np.vstack((theta_ini, theta_lhs_loops))
            theta0[:,:(len(self.options["theta0"]))] = np.log10(theta0[:,:(len(self.options["theta0"]))])
            lower_bounds[:(len(lower_bounds))]=np.log10(lower_bounds[:(len(lower_bounds))])
            upper_bounds[:(len(upper_bounds))]=np.log10(upper_bounds[:(len(upper_bounds))])
            constraints=[]
            for i in range(len(theta_ini)):
                constraints.append(lambda theta0, i=i: theta0[i] - lower_bounds[i])
                constraints.append(lambda theta0, i=i: upper_bounds[i] - theta0[i])
            for j in range(self.options["n_start"]):
                optimal_theta_res_loop = optimize.minimize(
                    self.neg_log_likelihooda,
                    theta0[j,:],
                    method="BFGS",
                    constraints=[
                        {"fun": con, "type": "ineq"} for con in constraints
                    ],
                    options={
                        "rhobeg": 0.5,
                        "tol": 1e-6,
                        "maxiter": 100,
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

        x_opt[0:2]=10**(x_opt[0:2])
        x_opt[2:8:3]=10**(x_opt[2:8:3])
        x_opt[3:8:3]=10**(x_opt[3:8:3])
        self.theta = x_opt#10**(x_opt[:(len(x_opt))])

    def predict_multi_lvl(self,x):
        """
        Generalized prediction function for the multi-fidelity co-kriging
        Parameters
        ----------
        x : np.ndarray
            Array with the inputs for make the prediction.
        Returns
        -------
        means : np.array
            Returns the conditional means per level.
        covariances: np.ndarray
            Returns the conditional covariance matrixes per level.
        """
        param0 = self.theta[0:2]
        sigmas_gamma = self.theta[2::3]
        ls_gamma = self.theta[3::3]
        rhos=self.theta[4::3]
        Y= self.y
        means=[]
        covariances=[]
        if self.lvl==1:
            k_XX = self._compute_K(self.X[0],self.X[0],param0)
            k_xX = self._compute_K(x,self.X[0],param0)
            k_xx = self._compute_K(x,x,param0)
            means.append( np.dot(k_xX, np.matmul(np.linalg.inv(k_XX + 1e-9*np.eye(k_XX.shape[0])), Y)))
            covariances.append(k_xx - np.matmul(k_xX,
                                                np.matmul(np.linalg.inv(k_XX+1e-9*np.eye(k_XX.shape[0])),
                                                          k_xX.transpose())))
        else:
            Kernels_xX = []
            K_xX_gamma = []
            k_xx = self._compute_K(x,x,param0)
            for ind in range(self.lvl):
                Kernels_xX.append(self._compute_K(self.X[ind], x, param0))
                if ind > 0:
                    K_xX_gamma.append(self._compute_K(self.X[ind], x, [sigmas_gamma[ind-1],ls_gamma[ind-1]]))
            K_ast = []
            K_ast.append( np.vstack(( Kernels_xX[0] , rhos[0] * Kernels_xX[1]  ))   )

            for ind in range(self.lvl):
                 if ind > 0:
                     K_ast.append( np.vstack((rhos[ind-1]* K_ast[ind-1][0:self.X[ind-1].shape[0]] ,
                    rhos[ind-1] * K_ast[ind-1][self.X[ind-1].shape[0]:self.X[ind-1].shape[0]+self.X[ind].shape[0]]
                    + K_xX_gamma[ind-1])) )
            for lvel in range(self.lvl):
                jitter = self.options["nugget"]
                self.K = self.compute_K(self.theta)
                L = np.linalg.cholesky(self.K+ jitter*np.eye(self.K.shape[0]))
                beta1 = solve_triangular(L, K_ast[lvel],lower=True)
                alpha1 = solve_triangular(L,Y,lower=True)
                means.append( np.dot(beta1.T,alpha1) )
                covariances.append(  k_xx - np.dot(beta1.T,beta1) )
        return means,covariances

    def predict(self,x):
        """
        Prediction function with the exact math formulation for 1,2 and 3 levels
        (Temporal function, just for comparison with predict_multilvl)
        Parameters
        ----------
        x : array
            Array with the inputs for make the prediction.
        Returns
        -------
        means : np.array
            Returns the conditional means per level.
        covariances: np.ndarray
            Returns the conditional covariance matrixes per level.
        """
        param0 = self.theta[0:2]
        sigmas_gamma = self.theta[2::3]
        ls_gamma = self.theta[3::3]
        rhos=self.theta[4::3]
        Y= self.y
        if self.lvl==1:
            k_XX = self._compute_K(self.X[0],self.X[0],param0)
            k_xX = self._compute_K(x,self.X[0],param0)
            k_xx = self._compute_K(x,x,param0)
            mean1 = np.dot(k_xX, np.matmul(np.linalg.inv(k_XX + 1e-9*np.eye(k_XX.shape[0])), Y))
            covariance1 =  k_xx - np.matmul(k_xX,
                                            np.matmul(np.linalg.inv(k_XX+1e-9*np.eye(k_XX.shape[0])),
                                                      k_xX.transpose()))
            return mean1,covariance1
        elif self.lvl==2:
            param=self.theta[0:2]
            params_gamma=self.theta[2:4]
            rhoc=rhos[0]
            self.K = self.compute_K(self.theta)
            jitter = self.options["nugget"]  # small number to ensure numerical stability. tao of smt implementation?.
            L = np.linalg.cholesky(self.K+ jitter*np.eye(self.K.shape[0]))
            k1as = self._compute_K(x,self.X[1],param)
            k2as = self._compute_K(x,self.X[1],params_gamma)
            k3as = self._compute_K(x,self.X[0],param)

            kxxas = self._compute_K(x,x,param)
            kxxas1 = self._compute_K(x,x,params_gamma)
            k11_ast = rhoc*rhoc*k1as + k2as
            k10_ast = rhoc * k3as
            k_xX=np.concatenate((k10_ast.T, k11_ast.T)).T
            k_xx = rhoc*rhoc* kxxas + kxxas1
            beta0 = solve_triangular(L, k_xX.T,lower=True)
            alpha0 = solve_triangular(L,Y,lower=True)
            meanhf = np.dot(beta0.T,alpha0)
            covariancehf = k_xx-np.dot(beta0.T,beta0)

            k01_ast = rhoc*k1as
            k00_ast = k3as
            k_xX = np.concatenate((k00_ast.T, k01_ast.T)).T
            k_xx = kxxas
            beta1 = solve_triangular(L, k_xX.T,lower=True)
            alpha1 = solve_triangular(L,Y,lower=True)
            meanlf = np.dot(beta1.T,alpha1)
            covariancelf = k_xx - np.dot(beta1.T,beta1)

            return meanhf,covariancehf,meanlf,covariancelf
        elif self.lvl==3:
            X2=self.X[2]
            X1=self.X[1]
            X0=self.X[0]
            self.K = self.compute_K(self.theta)
            jitter = self.options["nugget"]  # small number to ensure numerical stability
            L = np.linalg.cholesky(self.K+ jitter*np.eye(self.K.shape[0]))
            k00ast=self._compute_K(x,X0,self.theta[0:2])
            k01ast=rhos[0]*self._compute_K(x,X1,self.theta[0:2])
            k02ast = (rhos[1]) * (rhos[0]) * self._compute_K(x,X2,self.theta[0:2])
            k_xX=np.concatenate((k00ast.T, k01ast.T, k02ast.T)).T
            beta0 = solve_triangular(L, k_xX.T,lower=True)
            alpha0 = solve_triangular(L,Y,lower=True)
            mean1 = np.dot(beta0.T,alpha0)
            covariance1 = self._compute_K(x,x,self.theta[0:2]) - np.dot(beta0.T,beta0)
            k01ast=rhos[0]*self._compute_K(x,X0,self.theta[0:2])
            k11ast=((rhos[0]**2)*self._compute_K(x,X1,self.theta[0:2])
            +self._compute_K(x,X1,[sigmas_gamma[0],ls_gamma[0]]))
            k12ast = ((rhos[1]) * (rhos[0]**2) * self._compute_K(x,X2,self.theta[0:2])+
            (rhos[1] * self._compute_K(x,X2,[sigmas_gamma[0],ls_gamma[0]])))
            k_xX = np.concatenate((k01ast.T, k11ast.T,k12ast.T)).T
            beta1 = solve_triangular(L, k_xX.T,lower=True)
            alpha1 = solve_triangular(L,Y,lower=True)
            mean2 = np.dot(beta1.T,alpha1)
            covariance2 = ((rhos[0]**2)*self._compute_K(x,x,self.theta[0:2])
            +self._compute_K(x,x,[sigmas_gamma[0],ls_gamma[0]]) - np.dot(beta1.T,beta1))
            k02ast = (rhos[1]) * (rhos[0]) * self._compute_K(x,X0,self.theta[0:2])
            k12ast = ((rhos[1]) * (rhos[0]**2) * self._compute_K(x,X1,self.theta[0:2])+
            (rhos[1] * self._compute_K(x,X1,[sigmas_gamma[0],ls_gamma[0]])))
            k22ast = (rhos[1]**2) * ((rhos[0]**2)*self._compute_K(x,X2,self.theta[0:2])
                                     +self._compute_K(x,X2,[sigmas_gamma[0],ls_gamma[0]]))
            +self._compute_K(x,X2,[sigmas_gamma[1],ls_gamma[1]])
            k_xX = np.concatenate((k02ast.T, k12ast.T,k22ast.T)).T
            beta2 = solve_triangular(L, k_xX.T,lower=True)
            alpha2 = solve_triangular(L,Y,lower=True)
            mean3 = np.dot(beta2.T,alpha2)
            covariance3 = ((rhos[1]**2) * ((rhos[0]**2)*self._compute_K(x,x,self.theta[0:2])
                                          +self._compute_K(x,x,[sigmas_gamma[0],ls_gamma[0]]))
            +self._compute_K(x,x,[sigmas_gamma[1],ls_gamma[1]]) - np.dot(beta2.T,beta2))

            return mean1,covariance1,mean2,covariance2,mean3,covariance3
        else:
            self.predict_multi_lvl(x)

    def neg_log_likelihood(self,param1,grad):
        if self.lvl == 1:
            K=self._compute_K(self.X[0],self.X[0],param1[0:2])
        else:
            K = self.compute_K(param1)
        self.K = np.copy(K)
        #print(self.is_invertible(self.K))
        jitter = 1e-4#self.options["nugget"]  # small number to ensure numerical stability.
        L = np.linalg.cholesky(self.K+ jitter*np.eye(self.K.shape[0]))
        beta = solve_triangular(L, self.y,lower=True)
        N=np.shape(self.y)[0]
        NMLL=1/2*(2*np.sum(np.log(np.diag(L)))+np.dot(beta.T,beta)+N*np.log(2*np.pi))
        nmll,=NMLL[0]
        return nmll

    def neg_log_likelihooda(self,param1):
        param1[0:2]=10**(param1[0:2])
        param1[2:8:3]=10**(param1[2:8:3])
        param1[3:8:3]=10**(param1[3:8:3])
        return self.neg_log_likelihood(param1,1)

    def log_likelihood_gradient(self,param1):
          X2 = np.copy(self.X0)
          X1 = np.copy(self.X1)
          param=param1[0:2]
          params_gamma=param1[2:4]
          rho=param1[4::][0]
          jitter = self.options["nugget"]  # small number to ensure numerical stability.
          L = np.linalg.cholesky(self.K+ jitter*np.eye(self.K.shape[0]))
          y=np.copy(self.y)
          betaa = solve_triangular(L,np.identity(np.shape(L)[0]),lower=True)
          InverseK = np.dot(betaa.T,betaa)
          alpha0 = solve_triangular(L,y,lower=True)
          alphaa = np.dot(betaa.T,alpha0)
          # Partial derivatives of the likelihood with respect to the hyperparameters
          k1,grad = self.SEKernel(X1,X1,param)
          k2,grad1 = self.SEKernel(X1,X2,param)
          _,grad2 = self.SEKernel(X2,X2,param)
          _,grad3 = self.SEKernel(X1, X1, params_gamma)

          a = rho* rho* ( grad[1] )
          b = rho*(  grad1[1] )
          c = grad2[1]
          dK_dl0 = np.vstack((np.concatenate((a,b.T)).T, np.concatenate((b,c)).T))
          a = rho*rho*grad[0]
          b = rho*grad1[0]
          c = grad2[0]
          dK_dv0 = np.vstack((np.concatenate((a,b.T)).T,
                              np.concatenate((b,c)).T))
          a = grad3[1]
          dK_dlg =  np.vstack((np.concatenate((a,np.zeros_like(b).T)).T,
                               np.concatenate((np.zeros_like(b),np.zeros_like(c))).T))
          a = grad3[0]
          dK_dvg = np.vstack((np.concatenate((a,np.zeros_like(b).T)).T,
                              np.concatenate((np.zeros_like(b),np.zeros_like(c))).T))
          a = 2*rho*k1
          b = k2
          dK_drho = np.vstack((np.concatenate((a,b.T)).T, np.concatenate((b,np.zeros_like(c))).T))
          mid_term = np.dot(alphaa,alphaa.T) - InverseK

          dL_dl0 = -0.5 * np.trace(np.dot(mid_term , dK_dl0))
          dL_dv0 = -0.5 * np.trace(np.dot(mid_term , dK_dv0))
          dL_dlg = -0.5 * np.trace(np.dot(mid_term , dK_dlg))
          dL_dvg = -0.5 * np.trace(np.dot(mid_term , dK_dvg))
          dL_drho = -0.5 * np.trace(np.dot(mid_term , dK_drho))

          return np.array([dL_dv0, dL_dl0, dL_dvg, dL_dlg, dL_drho])

    def compute_K(self, param1):
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
