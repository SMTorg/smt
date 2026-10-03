import numpy as np
from smt.design_space import DesignSpace, FloatVariable
from smt.surrogate_models import KRG
from copy import deepcopy
import unittest

def xsinx(x:np.ndarray)->np.floating:
    return x*np.sin(x)

def verify_block_update(base_model, X_new, Y_new, test_name):
    """
    Helper function to copy the base model, apply the block update,
    and compare it against the mathematically true Cholesky decomposition.
    """
    model_block_update = deepcopy(base_model)
    model_block_update.fast_update_model(X_new, Y_new)
    C_block_update = model_block_update.optimal_par["C"]

    model_reference = deepcopy(base_model)
    X_new_training_matrix = np.vstack((base_model.X_train, X_new))
    Y_new_training_matrix = np.vstack((base_model.training_points[None][0][1], Y_new))
    model_reference.set_training_values(X_new_training_matrix, Y_new_training_matrix)
    model_reference.options["hyper_opt"] = "NoOp"
    model_reference.theta0 = base_model.optimal_theta.tolist()
    model_reference.options["theta0"] = base_model.optimal_theta.tolist()

    # Monkey-patch the standardization function just for this test
    # so that the reference model uses the original model's offsets as the block update approach.
    import smt.surrogate_models.krg_based.krg_based
    from smt.utils.misc import standardization as real_standardization
    original_standardization = smt.surrogate_models.krg_based.krg_based.standardization

    def mocked_standardization(X, y):
        X_norma, y_norma, X_offset, y_mean, X_scale, y_std = real_standardization(X, y)

        X_offset = base_model.X_offset
        X_scale = base_model.X_scale
        y_mean = base_model.y_mean
        y_std = base_model.y_std

        X_norma = (X - X_offset) / X_scale
        y_norma = (y - y_mean) / y_std

        return X_norma, y_norma, X_offset, y_mean, X_scale, y_std

    smt.surrogate_models.krg_based.krg_based.standardization = mocked_standardization

    original_prepare_training_data = model_reference._prepare_training_data
    def mocked_prepare_training_data():
        X = model_reference.training_points[None][0][0]
        y = model_reference.training_points[None][0][1]

        is_acting = model_reference.is_acting_points.get(None)
        if is_acting is None and not model_reference.is_continuous:
            X, is_acting = model_reference.design_space.correct_get_acting(X)
            model_reference.training_points[None][0][0] = X
            model_reference.is_acting_points[None] = is_acting

        model_reference._check_param()
        model_reference.X_train = X
        model_reference.is_acting_train = is_acting
        from smt.surrogate_models.krg_based.krg_based import compute_X_cont
        _, model_reference.cat_features = compute_X_cont(model_reference.X_train, model_reference.design_space)

        (
            model_reference.X_norma,
            model_reference.y_norma,
            model_reference.X_offset,
            model_reference.y_mean,
            model_reference.X_scale,
            model_reference.y_std,
        ) = mocked_standardization(X.copy(), y.copy()) # mocked standardization inserted here

        if not model_reference._eval_noise:
            model_reference.optimal_noise = np.array(model_reference._noise0)

        return X, y, is_acting

    model_reference._prepare_training_data = mocked_prepare_training_data

    try:
        model_reference.train()
    finally:
        smt.surrogate_models.krg_based.krg_based.standardization = original_standardization
        model_reference._prepare_training_data = original_prepare_training_data

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

    return max_c_diff, max_y_diff

class TestModelBlockUpdate(unittest.TestCase):
    def test_block_update_cont_KRG(self):
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
            print_global=False,
        )

        model.set_training_values(xt=x_DOI, yt=y_doi)
        model.train()

        x_eval_single = np.array([[23.0]])
        X_eval_multiple = np.array([[16.0], [19.0], [23.0], [25.0]])

        y_eval_single = model.predict_values(x_eval_single)
        Y_eval_multiple = model.predict_values(X_eval_multiple)

        max_c_diff_single, max_y_diff_single = verify_block_update(
            model, x_eval_single, y_eval_single, "Single Point Update (x_eval)"
        )
        max_c_diff_multiple, max_y_diff_multiple = verify_block_update(
            model, X_eval_multiple, Y_eval_multiple, "Multiple Points Update (X_eval)"
        )

        self.assertLess(max_c_diff_single, 1e-6)
        self.assertLess(max_y_diff_single, 1e-6)
        self.assertLess(max_c_diff_multiple, 1e-6)
        self.assertLess(max_y_diff_multiple, 1e-6)

if __name__ == "__main__":
    unittest.main()
