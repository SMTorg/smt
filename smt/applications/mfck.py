"""
# Multi-Fidelity Co-Kriging 
Assuming that each level of fidelity is not a subset of the previous.
Mauricio CASTANO AGUIRRE
ONERA- UPHF
2024
"""
from scipy.spatial.distance import cdist
from scipy.linalg import solve_triangular
from smt.sampling_methods import LHS
import numpy as np
import nlopt
from smt.utils.kriging import differences
from scipy import optimize
from smt.utils.kriging import componentwise_distance

#from smt.surrogate_models import sgp # Exploring the option, for use sgp for add sparsity to the MFCK model 

from smt.surrogate_models.krg_based import KrgBased

class MFCK(KrgBased):
   
    def _initialize(self):
        super()._initialize()
        declare = self.options.declare
        self.name = "MFCK"
        self.params = {}
        self.doe = None
        self.doe_response = None
        self.K = None
        self.theta = None
        self.lvl = None
        self.X = []
        
    def set_training_values(self, doe, doe_response):
        self.doe= doe
        self.doe_response = doe_response
        self.lvl = np.int8(np.max(doe[:][:,-1])) + 1
        for unique in np.unique(self.doe[:][:,-1]):
            self.X.append(self.doe[:][self.doe[:][:,-1] == unique, 0:-1])
        
        print("\n MFCK \n")
        print(f'\n\n\n__________Experiment with {self.lvl} lvl and {self.X[0].shape[1]} Dim________\n\n\n')
        if self.lvl == 1:
            self.params["all_params"] = np.zeros([self.lvl,2])
        else: 
            self.params["all_params"] = np.zeros([self.lvl,3])
        
        
    def train(self):
        print("Training...")
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
                    random_state=1,
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
                        "tol": 1e-4,
                        "maxiter": 50,
                    },
                )
                
                x_opt_iter = optimal_theta_res_loop.x
                
                if j==0:
                    
                    x_opt=x_opt_iter
                    nll=optimal_theta_res_loop["fun"]
                    
                else:
                    
                    if optimal_theta_res_loop["fun"] < nll:
                        #print("Improvement, Likelihood=",optimal_theta_res_loop["fun"],"Params=",optimal_theta_res_loop.x)
                        
                        x_opt=x_opt_iter
                        
                        nll=optimal_theta_res_loop["fun"]
        
        elif self.options["Hyper_opt"]=="TNC":
            
            
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
                    method="TNC",
                    jac=self.log_likelihood_gradient
                    ###The hessian information is available but never used
                    #
                    ####hess=hessian_minus_reduced_likelihood_function,
                    #bounds=bounds_hyp,
                    #options={"maxfun": limit},
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
        
            # Set optimization parameters
            opt = nlopt.opt(nlopt.LD_LBFGS, np.shape(theta_ini)[0])  # Use COBYLA algorithm in 2 dimensions
            
            opt.set_lower_bounds(lower_bounds)  # Lower bounds for each dimension
            
            opt.set_upper_bounds(upper_bounds)  # Upper bounds for each dimension
            
            opt.set_min_objective(self.neg_log_likelihood)  # Set the objective function
            
            opt.set_xtol_rel(1e-4)  # Adjust tolerance level as needed
    
            # Set initial guess
            x0 = np.copy(theta_ini)  # Initial guess
    
            # Perform optimization
            x_opt = opt.optimize(x0)
        
        self.theta = x_opt
          
        
    def predict(self,x):
        
        print("Predicting...")
        
        sigmas_gamma = self.theta[2::3]
        
        ls_gamma = self.theta[3::3]

        rhos=self.theta[4::3]
        
        Y= self.doe_response
        
        if self.lvl==2:
            
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
        
    def neg_log_likelihood(self,param1,grad):
        
        y = self.doe_response
        
        try:
            if(param1[4::3].shape[0]+1 != self.lvl):
                print('Not enough number of Rhos for the levels of the input data')
                raise SyntaxError('=====error')
        except:
        
            pass
        
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
        
        #print(nmll)
        
        return nmll
    
    def neg_log_likelihooda(self,param1):
        
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
         
          #Calculation of the inverse of K
          betaa = solve_triangular(L,np.identity(np.shape(L)[0]),lower=True)
          InverseK = np.dot(betaa.T,betaa)
          
          #Calculation of \alpha = k inverse * y
          
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
          
          dK_dl0 = np.vstack((np.concatenate((a,b.T)).T, np.concatenate((b,c)).T))                 #variance * np.exp(-0.5 / lengthscale**2 * sqdist) * (0.5 / lengthscale**3) * sqdist
          
          a = rho*rho*grad[0]
          b = rho*grad1[0]
          c = grad2[0]
          
          dK_dv0 = np.vstack((np.concatenate((a,b.T)).T, np.concatenate((b,c)).T)) #2 * variance * np.exp(-0.5 / lengthscale**2 * sqdist)
          
          a = grad3[1]
          
          dK_dlg =  np.vstack((np.concatenate((a,np.zeros_like(b).T)).T, np.concatenate((np.zeros_like(b),np.zeros_like(c))).T))               #2 * noise * np.eye(N)
          
          a = grad3[0]
          
          dK_dvg = np.vstack((np.concatenate((a,np.zeros_like(b).T)).T, np.concatenate((np.zeros_like(b),np.zeros_like(c))).T)) 
          
          a = 2*rho*k1
          b = k2
          
    
          dK_drho = np.vstack((np.concatenate((a,b.T)).T, np.concatenate((b,np.zeros_like(c))).T)) 
          
          mid_term = np.dot(alphaa,alphaa.T) - InverseK
          
         
          
          dL_dl0 = -0.5 * np.trace( np.dot( mid_term , dK_dl0) )
          
          dL_dv0 = -0.5 * np.trace( np.dot( mid_term , dK_dv0) )
          
          dL_dlg = -0.5 * np.trace( np.dot( mid_term , dK_dlg) )
          
          dL_dvg = -0.5 * np.trace( np.dot( mid_term , dK_dvg) )
          
          dL_drho = -0.5 * np.trace( np.dot( mid_term , dK_drho) )
          
          
          return np.array([dL_dv0, dL_dl0, dL_dvg, dL_dlg, dL_drho])
    
    
    def _compute_K(self, A: np.ndarray, B: np.ndarray, param):
        """
        Compute the covariance matrix K between A and B
            Modified for MFCK initial test, Isotropic theta (Same theta for each dimmension)
        """
        # Compute pairwise componentwise L1-distances between A and B
        dx = differences(A, B)
        d = componentwise_distance(dx,self.options["corr"],self.X[0].shape[1],power=self.options["pow_exp_power"])
        R = self._correlation_types[self.options["corr"]](
            
            np.full(self.X[0].shape[1],param[1]),
            
            d).reshape(A.shape[0], B.shape[0])
        
        # Compute the covariance matrix K
        K = param[0] * R
        return K
    
    def compute_K(self,param1):
        """
        Compute the co-kriging piece-wise matrix  
        
        Parameters
        ----------
        param1 : array with the hyperparameters for the co-kriging
        
        if
            n_lvl==1 ---> param1 size = 2 (Sigma2, Theta (Related to length scale of the kernel in the case of square exponential) )
            n_level>1 ----> param1 size = 3* n_level -1 (Each new fidelity as a sigma2, Theta and \pho)
    
        Returns
        -------
        K : Returns the piece-wise matrix of the co-kriging model
        """
        #print("Params used:",param1)
        
        

        rhos = param1[4::3]
        
        params_smt = np.asarray(param1[0:2])
        
        K_cross = []
        
        K_var = []
        
        n = self.doe.shape[0]
        
        for lvel in range(self.lvl):
            
            #%%Fill kvar
            
                
            if lvel == 0:
                K_var.append(self._compute_K(self.X[lvel],self.X[lvel], params_smt))
                
            elif lvel > 0:
                if lvel == 1:
                    param = param1[3 * lvel -1 : 3 * lvel + 2]
                                    
                    k1 = self._compute_K(self.X[lvel],self.X[lvel], params_smt)
                    k2 = self._compute_K(self.X[lvel],self.X[lvel], param[:-1])
                    
                    K_var.append( ((param[-1] ** 2)  * k1) + k2  )
            
        #%%Fill K_cross
        
        if self.lvl ==2:
            
            r1 = self._compute_K(self.X[0], self.X[1], np.asarray(param1[0:2]))
            K_cross.append(  param1[-1] * r1 )
    
        #%%Fill big K
        
        K = np.zeros((n, n))
        
        row = 0
        
        for i, matrix in enumerate(K_var): 
            size = matrix.shape[0]
            K[row: row +size, row: row+size] = matrix
            row += size
            
        
        row, col = len(K_var[0]), 0
        
        shape_acum = []
        
        for lvl in range(1,self.lvl):
            
            for i in range(lvl):
            
                r,c = K_cross[lvl+i-1].shape
                
                shape_acum.append(r)
                
                K[ col:col + r, row:row + c ] = K_cross[lvl+i-1]
                
                K[row:row+ c, col: col + r ] = K_cross[lvl+i-1].T
                
                if col>0:
                    
                    r,_ = K_cross[lvl+i-2].shape
                    
                    col-=r
            
            col = col +  np.sum(shape_acum)
            row += c
            shape_acum = []
        
        
        
        
        return K
    
    
   
    def SEKernel(self, X1, X2, param,compt_grad=False): 
        sigma2, l = param
        
        dist2 = (cdist(X1, X2)/l)**2
        
        k = sigma2*np.exp(-0.5*dist2)
        
        
        if compt_grad:
            
            dk_ds2 = k/sigma2
            dk_dl =  k * (dist2 * (1/l))
            return k, np.array([dk_ds2, dk_dl])
        else:
            
            return k

    
    # def print_matrix(self,matrix):
    #     for row in matrix:
    #         print(" ".join("{:.2f}".format(val) for val in row))
    
    