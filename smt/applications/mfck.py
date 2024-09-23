"""
Multi-Fidelity Co-Kriging 
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
        #declare = self.options.declare
        self.name = "MFCK"
        
        self.params = {}
        self.doe = None
        self.doe_response = None
        self.K = None
        self.theta = None
        self.lvl = None
        self.X = []
    def set_training_values(self, doe, doe_response):
        """
        Parameters
        ----------
        doe : np.ndarray
            Input space with aditional column for indicate the level 
            of fidelity of the each observation.
        doe_response : np.ndarray
            evaluation of the inputs over the real function.
        Returns
        -------
        None.
        """
        self.doe= doe
        self.doe_response = doe_response
        self.lvl = np.int8(np.max(doe[:][:,-1])) + 1
        for unique in np.unique(self.doe[:][:,-1]):
            self.X.append(self.doe[:][self.doe[:][:,-1] == unique, 0:-1])
        if self.lvl == 1:
            self.params["all_params"] = np.zeros([self.lvl,2])
        else:
            self.params["all_params"] = np.zeros([self.lvl,3])

    def train(self):
        """
        Function for train the Hyper-parameters of the MFCK model
        Returns
        -------
        None.
        """
        if self.lvl == 1:
            theta_ini = np.copy(self.options["theta0"])
            theta_ini = np.vstack((1.0,theta_ini))
            lower_bounds = [self.options["theta_bounds"][0], self.options["theta_bounds"][0]]
            upper_bounds = [self.options["theta_bounds"][1], self.options["theta_bounds"][1]]
        else :
            for lvel in range(self.lvl):
                if lvel==0:
                    theta_ini = np.copy(self.options["theta0"])
                    theta_ini = np.vstack((1.0,theta_ini))
                    lower_bounds = [self.options["theta_bounds"][0], self.options["theta_bounds"][0]]
                    upper_bounds = [self.options["theta_bounds"][1], self.options["theta_bounds"][1]]
                elif lvel > 0:
                    theta_ini = np.vstack([ theta_ini, np.vstack([[0.5], self.options["theta0"],[1.1]]) ])
                    upper_bounds = np.hstack([upper_bounds,[self.options["theta_bounds"][1], self.options["theta_bounds"][1],2.0]])
                    lower_bounds = np.hstack([lower_bounds,[self.options["theta_bounds"][0], self.options["theta_bounds"][0],0.1]])
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
        self.theta = 10**(x_opt[:(len(x_opt))])
        
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
        Y= self.doe_response
        means=[]
        covariances=[]
        
        if self.lvl==1:
            k_XX = self._compute_K(self.X[0],self.X[0],param0)
            k_xX = self._compute_K(x,self.X[0],param0)
            k_xx = self._compute_K(x,x,param0)
            means.append( np.dot(k_xX, np.matmul(np.linalg.inv(k_XX + 1e-9*np.eye(k_XX.shape[0])), Y)))
            covariances.append(k_xx - np.matmul(k_xX, np.matmul(np.linalg.inv(k_XX+1e-9*np.eye(k_XX.shape[0])), k_xX.transpose())))
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
        Y= self.doe_response    
        if self.lvl==1:   
            k_XX = self._compute_K(self.X[0],self.X[0],param0)
            k_xX = self._compute_K(x,self.X[0],param0)
            k_xx = self._compute_K(x,x,param0)
            mean1 = np.dot(k_xX, np.matmul(np.linalg.inv(k_XX + 1e-9*np.eye(k_XX.shape[0])), Y))
            covariance1 =  k_xx - np.matmul(k_xX, np.matmul(np.linalg.inv(k_XX+1e-9*np.eye(k_XX.shape[0])), k_xX.transpose()))
            return mean1,covariance1
        elif self.lvl==2:
            X0=self.doe[:][self.doe[:][:,-1] == 0, 0:-1]
            X1=self.doe[:][self.doe[:][:,-1] == 1, 0:-1]
            param=self.theta[0:2]
            params_gamma=self.theta[2:4]
            rhoc=rhos[0]
            self.K = self.compute_K(self.theta)
            jitter = self.options["nugget"]  # small number to ensure numerical stability. tao of smt implementation?.
            L = np.linalg.cholesky(self.K+ jitter*np.eye(self.K.shape[0]))
            k1as = self._compute_K(x,X1,param)
            k2as = self._compute_K(x,X1,params_gamma)
            k3as = self._compute_K(x,X0,param)
            kxxas = self._compute_K(x,x,param)
            kxxas1 = self._compute_K(x,x,params_gamma)
            k11_ast = rhoc*rhoc*k1as + k2as
            k10_ast = rhoc * k3as
            k_xX=np.concatenate((k10_ast.T, k11_ast.T)).T
            k_xx = rhoc*rhoc* kxxas + kxxas1
            beta0 = solve_triangular(L, k_xX.T,lower=True)
            alpha0 = solve_triangular(L,Y,lower=True)
            mean1 = np.dot(beta0.T,alpha0)
            covariance1 = k_xx-np.dot(beta0.T,beta0)
            k01_ast = rhoc*k1as
            k00_ast = k3as
            k_xX = np.concatenate((k00_ast.T, k01_ast.T)).T
            k_xx = kxxas
            beta1 = solve_triangular(L, k_xX.T,lower=True)
            alpha1 = solve_triangular(L,Y,lower=True)
            mean2 = np.dot(beta1.T,alpha1)
            covariance2 = k_xx - np.dot(beta1.T,beta1)
            
            return mean1,covariance1,mean2,covariance2
        elif self.lvl==3:
            X2=self.doe[:][self.doe[:][:,-1] == 2, 0:-1]
            X1=self.doe[:][self.doe[:][:,-1] == 1, 0:-1]
            X0=self.doe[:][self.doe[:][:,-1] == 0, 0:-1]
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
            k11ast=(rhos[0]**2)*self._compute_K(x,X1,self.theta[0:2])+self._compute_K(x,X1,[sigmas_gamma[0],ls_gamma[0]])
            k12ast = (rhos[1]) * (rhos[0]**2) * self._compute_K(x,X2,self.theta[0:2]) + (rhos[1] * self._compute_K(x,X2,[sigmas_gamma[0],ls_gamma[0]])) 
            k_xX = np.concatenate((k01ast.T, k11ast.T,k12ast.T)).T
            beta1 = solve_triangular(L, k_xX.T,lower=True)
            alpha1 = solve_triangular(L,Y,lower=True)
            mean2 = np.dot(beta1.T,alpha1)
            covariance2 = (rhos[0]**2)*self._compute_K(x,x,self.theta[0:2])+self._compute_K(x,x,[sigmas_gamma[0],ls_gamma[0]]) - np.dot(beta1.T,beta1)
            k02ast = (rhos[1]) * (rhos[0]) * self._compute_K(x,X0,self.theta[0:2])
            k12ast = (rhos[1]) * (rhos[0]**2) * self._compute_K(x,X1,self.theta[0:2])+ (rhos[1] * self._compute_K(x,X1,[sigmas_gamma[0],ls_gamma[0]]))
            k22ast = (rhos[1]**2) * ((rhos[0]**2)*self._compute_K(x,X2,self.theta[0:2])+self._compute_K(x,X2,[sigmas_gamma[0],ls_gamma[0]])) + self._compute_K(x,X2,[sigmas_gamma[1],ls_gamma[1]])
            k_xX = np.concatenate((k02ast.T, k12ast.T,k22ast.T)).T
            beta2 = solve_triangular(L, k_xX.T,lower=True)
            alpha2 = solve_triangular(L,Y,lower=True)
            mean3 = np.dot(beta2.T,alpha2)
            covariance3 = (rhos[1]**2) * ((rhos[0]**2)*self._compute_K(x,x,self.theta[0:2])+self._compute_K(x,x,[sigmas_gamma[0],ls_gamma[0]]))+self._compute_K(x,x,[sigmas_gamma[1],ls_gamma[1]]) - np.dot(beta2.T,beta2)
            
            return mean1,covariance1,mean2,covariance2,mean3,covariance3
        else:
            self.predict_multi_lvl(x)
        
    def neg_log_likelihood(self,param1,grad):
        y = self.doe_response
        if self.lvl == 1:
            K=self._compute_K( self.X[0],self.X[0],param1[0:2] )
        else:
            K = self.compute_K(param1)
        self.K = np.copy(K)
        jitter = self.options["nugget"]  # small number to ensure numerical stability. 
        L = np.linalg.cholesky(self.K+ jitter*np.eye(self.K.shape[0]))
        beta = solve_triangular(L, y,lower=True)
        N=np.shape(y)[0]
        NMLL = 1/2*(   2*np.sum(np.log(np.diag(L))) + np.dot(beta.T,beta)  +  N * np.log(2*np.pi) )
        nmll, = NMLL[0]
        return nmll
    
    def neg_log_likelihooda(self,param1):
        param1[:(len(param1))] = 10**(param1[:(len(param1))])
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
          dK_dv0 = np.vstack((np.concatenate((a,b.T)).T, np.concatenate((b,c)).T))
          a = grad3[1]
          dK_dlg =  np.vstack((np.concatenate((a,np.zeros_like(b).T)).T, np.concatenate((np.zeros_like(b),np.zeros_like(c))).T))
          a = grad3[0]
          dK_dvg = np.vstack((np.concatenate((a,np.zeros_like(b).T)).T, np.concatenate((np.zeros_like(b),np.zeros_like(c))).T)) 
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
        param1 = np.asarray(param1)
        rhos = param1[4::3]
        params_smt = np.asarray(param1[0:2])  # Hyperparameters for the first level
        K_var = []
        K_cross = []
        n = self.doe.shape[0]
        # Precompute K matrices to avoid redundant calculations
        precomputed_K = {}
        def get_K(lvl, lvl1, params):
            """Helper function to compute or retrieve precomputed K matrices for (lvl, lvl1)."""
            # Convert params to a tuple if it's a numpy array
            if isinstance(params, np.ndarray):
                params = tuple(params)
            # Use tuple (lvl, lvl1) as the key for the precomputed matrices
            key = (lvl, lvl1, params)
            if key not in precomputed_K:
                precomputed_K[key] = self._compute_K(self.X[lvl], self.X[lvl1], params)
            return precomputed_K[key]
        # Fill K_var matrix
        for lvel in range(self.lvl):
            if lvel == 0:
                K_var.append(get_K(lvel , lvel, params_smt))
            else:
                param = param1[3 * lvel - 1: 3 * lvel + 2]
                param_ant = param1[3 * (lvel - 1) - 1: 3 * (lvel - 1) + 2] if lvel > 1 else None
    
                k1 = get_K(lvel,lvel, params_smt[:2])
                k2 = get_K(lvel,lvel, param_ant[:-1]) if param_ant is not None else None
                k3 = get_K(lvel,lvel, param[:-1])
                
                if lvel == 1:
                    K_var.append(param[-1]**2 * k1 + k3)
                else:
                    Var_ant = param_ant[-1] ** 2 * k1 + (k2 if k2 is not None else 0)
                    K_var.append(param[-1] ** 2 * Var_ant + k3)
        # Fill K_cross matrix
        for i in range(1, self.lvl):
            for j in range(i-1, -1, -1):
                # print(f"({i},{j})")
                r1 = get_K(i, j, params_smt)
                if i == 1:
                    K_cross.append(param1[-1] * r1.T)
                else:
                    rho_product = np.prod(rhos[:j+1])
                    K_cross.append(rho_product * r1.T)
        # Assemble the big K matrix
        K = np.zeros((n, n))
        row = 0
        # Fill diagonal blocks (K_var)
        for matrix in K_var:
            size = matrix.shape[0]
            K[row: row + size, row: row + size] = matrix
            row += size
        # Fill the off-diagonal blocks (K_cross)
        row, col = len(K_var[0]), 0
        shape_acum = []
        for lvl in range(1, self.lvl):
            for i in range(lvl):
                r, c = K_cross[lvl + i - 1].shape
                shape_acum.append(r)
                K[col:col + r, row:row + c] = K_cross[lvl + i - 1]
                K[row:row + c, col: col + r] = K_cross[lvl + i - 1].T
                if col > 0:
                    r_prev, _ = K_cross[lvl + i - 2].shape
                    col -= r_prev
            col += np.sum(shape_acum)
            row += c
            shape_acum = []
        return K
    
    def _compute_K(self, A: np.ndarray, B: np.ndarray, param):
        """
        Compute the covariance matrix K between A and B
            Modified for MFCK initial test (Same theta for each dimmension)
        """
        # Compute pairwise componentwise L1-distances between A and B
        dx = differences(A, B)
        d = componentwise_distance(dx,self.options["corr"],self.X[0].shape[1],power=self.options["pow_exp_power"])
        R = self._correlation_types[self.options["corr"]](   
            np.full(self.X[0].shape[1],param[1]),
            d).reshape(A.shape[0], B.shape[0])
        #Compute the covariance matrix K
        K = param[0] * R
        return K