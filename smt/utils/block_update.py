import numpy as np
from scipy.linalg import solve_triangular, cholesky
from smt.surrogate_models.krg import KRG
from smt.surrogate_models.krg_based.distances import differences


def update_cholesky_block(L_old, R12, R22, nugget=1e-10):
    """
    Cholesky block update.
    Returns the new Cholesky matrix along with the off-diagonal (L21) 
    and new diagonal (L22) blocks needed for the inverse update.
    """
    n_old = L_old.shape[0]
    n_new = R22.shape[0]
    
    L21_T = solve_triangular(L_old, R12, lower=True, check_finite=False)
    L21 = L21_T.T

    if n_new == 1:
        R22_eff = R22[0, 0] - np.sum(L21_T**2) + nugget
        if R22_eff <= 0.0:
            raise ValueError("Updated correlation matrix is not positive definite.")
        L22 = np.array([[np.sqrt(R22_eff)]])
        
    else:
        R22_eff = R22 - (L21 @ L21_T)
        np.fill_diagonal(R22_eff, R22_eff.diagonal() + nugget)
        L22 = cholesky(R22_eff, lower=True, check_finite=False)
    
    L_new = np.empty((n_old + n_new, n_old + n_new))
    L_new[:n_old, :n_old] = L_old
    L_new[:n_old, n_old:] = 0.0      
    L_new[n_old:, :n_old] = L21
    L_new[n_old:, n_old:] = L22
    
    return L_new, L21, L22


def update_inverse_cholesky_block(L_old_inv, L21, L22):
    """
    Updates the inverse of the Cholesky factor L^{-1}.
    Optimized for sequential additions.
    """
    n_old = L_old_inv.shape[0]
    n_new = L22.shape[0]
    
    L22_inv = np.array([[1.0 / L22[0, 0]]]) if n_new == 1 else solve_triangular(L22, np.eye(n_new), lower=True, check_finite=False)
    
    M21 = -L22_inv @ (L21 @ L_old_inv)
    
    L_new_inv = np.empty((n_old + n_new, n_old + n_new))
    L_new_inv[:n_old, :n_old] = L_old_inv
    L_new_inv[:n_old, n_old:] = 0.0
    L_new_inv[n_old:, :n_old] = M21
    L_new_inv[n_old:, n_old:] = L22_inv
    
    return L_new_inv


def predict_covariance(sm_model: KRG, X_new: np.ndarray, is_normalized: bool = False, compute_inverse:bool = True):
    """
    Computes the joint Cholesky factorization of the training data + new candidates.
    
    Parameters:
    - is_normalized: If True, bypasses the internal SMT normalization step.
    - compute_inverse: If True, computes the inverse of the Cholesky Gram matrix using the block update approach
    """
    if is_normalized:
        X_new_norma = X_new
    else:
        X_new_norma = (X_new - sm_model.X_offset) / sm_model.X_scale

    dx12 = differences(sm_model.X_norma, Y=X_new_norma)
    d12 = sm_model._componentwise_distance(dx12)
    R12 = sm_model.corr(d12).reshape(sm_model.nt, X_new.shape[0])

    dx22 = differences(X_new_norma, Y=X_new_norma)
    d22 = sm_model._componentwise_distance(dx22)
    R22 = sm_model.corr(d22).reshape(X_new.shape[0], X_new.shape[0])

    L_old = sm_model.optimal_par["C"]
    
    try:
        nugget = sm_model.options["nugget"]
    except KeyError:
        nugget = 1e-8
        
    L_new, L21, L22 = update_cholesky_block(L_old, R12, R22, nugget=nugget)

    if not compute_inverse:
        return L_new, None 
    else:
        L_old_inv = solve_triangular(L_old, np.eye(L_old.shape[0]), lower=True, check_finite=False) if "C_inv" not in sm_model.optimal_par else sm_model.optimal_par["C_inv"]
        
    L_new_inv = update_inverse_cholesky_block(L_old_inv, L21, L22)

    return L_new, L_new_inv


def update_smt_model(sm_model: KRG, X_new: np.ndarray, Y_new: np.ndarray) -> KRG:
    """
    Performs a Cholesky block update and injects the state back into the SMT model.
    Utilizes L^{-1} for heavily accelerated coefficient updates.
    """
    X_new_norma = (X_new - sm_model.X_offset) / sm_model.X_scale
    Y_new_norma = (Y_new - sm_model.y_mean) / sm_model.y_std

    L_new, L_new_inv = predict_covariance(sm_model, X_new_norma, is_normalized=True, compute_inverse=True)

    sm_model.X_train = np.vstack((sm_model.X_train, X_new))
    sm_model.training_points[None][0][0] = sm_model.X_train
    sm_model.training_points[None][0][1] = np.vstack((sm_model.training_points[None][0][1], Y_new.reshape(-1, 1)))
    
    sm_model.X_norma = np.vstack((sm_model.X_norma, X_new_norma))
    sm_model.y_norma = np.vstack((sm_model.y_norma, Y_new_norma.reshape(-1, 1)))
    sm_model.nt = sm_model.X_norma.shape[0]

    F_combined = sm_model._regression_types[sm_model.options['poly']](sm_model.X_norma)

    v_F = L_new_inv @ F_combined
    v_Y = L_new_inv @ sm_model.y_norma
    
    R_F = v_F.T @ v_F
    beta_new = np.linalg.solve(R_F, v_F.T @ v_Y)

    residuals = sm_model.y_norma - (F_combined @ beta_new)
    
    gamma_new = L_new_inv.T @ (L_new_inv @ residuals)

    sm_model.optimal_par["C"] = L_new
    sm_model.optimal_par["C_inv"] = L_new_inv  
    sm_model.optimal_par["beta"] = beta_new
    sm_model.optimal_par["gamma"] = gamma_new
    
    sm_model.optimal_par["Ft"] = v_F
    Q, G = np.linalg.qr(v_F)
    sm_model.optimal_par["Q"] = Q
    sm_model.optimal_par["G"] = G
    
    return sm_model