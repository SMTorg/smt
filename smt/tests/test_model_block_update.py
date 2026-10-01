import numpy as np
from smt.design_space import DesignSpace, FloatVariable
from smt.surrogate_models import KRG
from copy import deepcopy
from smt.surrogate_models.krg_based.block_update import update_smt_model 

def xsinx(x:np.ndarray)->np.floating:
    return x*np.sin(x)

def verify_block_update(base_model, X_new, Y_new, test_name):
    """
    Helper function to copy the base model, apply the block update, 
    and compare it against the mathematically true Cholesky decomposition.
    """
    model_block_update = deepcopy(base_model)
    model_block_update = update_smt_model(sm_model=model_block_update, X_new=X_new, Y_new=Y_new)
    C_block_update = model_block_update.optimal_par["C"]

    model_reference = deepcopy(base_model)
    X_new_training_matrix = np.vstack((base_model.X_train, X_new))
    Y_new_training_matrix = xsinx(X_new_training_matrix)
    model_reference.set_training_values(X_new_training_matrix, Y_new_training_matrix)
    model_reference.options["hyper_opt"] = "NoOp"
    model_reference.train()
    C_reference = model_reference.optimal_par["C"]

    print(f"--- Matrix Comparison: {test_name} ---")
    print(f"New points added: {X_new.shape[0]}")
    print(f"Resulting Matrix Shape: {C_block_update.shape}")
    print(f"Are C_new and C_true the same shape? {C_block_update.shape == C_reference.shape}")
    
    max_c_diff = np.max(np.abs(C_reference - C_block_update))
    print(f"Max absolute difference in C: {max_c_diff:.4e}")

    X_test = np.linspace(0, 25, 200).reshape(-1, 1)
    
    Y_upd = model_block_update.predict_values(X_test)
    Y_true = model_reference.predict_values(X_test)
    
    max_y_diff = np.max(np.abs(Y_upd - Y_true))
    
    print(f"Max absolute difference in Predictions (Y): {max_y_diff:.4e}")
    print("-" * 55 + "\n")

if __name__ == "__main__":
    lower_bound, upper_bound = 0, 25
    x_DOI = np.array([
            [3.5], 
            [8.0], 
            [12.0]
            ])
    y_doi = xsinx(x_DOI)

    ds = DesignSpace([FloatVariable(lower_bound, upper_bound)])

    model = KRG(
        design_space=ds,
        theta0=[1e-1],
        theta_bounds=[1e-2, 1e2],
        eval_noise=False,
        seed=42,
        hyper_opt="TNC",
        corr="squar_exp",
    )

    model.set_training_values(xt=x_DOI, yt=y_doi)
    model.train()

    x_eval_single = np.array([[23.0]])
    X_eval_multiple = np.array([[16.0], [19.0], [23.0], [25.0]])

    y_eval_single = model.predict_values(x_eval_single)
    Y_eval_multiple = model.predict_values(X_eval_multiple)

    verify_block_update(model, x_eval_single, y_eval_single, "Single Point Update (x_eval)")
    verify_block_update(model, X_eval_multiple, Y_eval_multiple, "Multiple Points Update (X_eval)")