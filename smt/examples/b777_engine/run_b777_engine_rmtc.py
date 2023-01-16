from smt.surrogate_models import RMTC
from smt.examples.b777_engine.b777_engine import get_b777_engine, plot_b777_engine

xt, yt, dyt_dxt, xlimits = get_b777_engine()

interp = RMTC(
    num_elements=6,
    xlimits=xlimits,
    nonlinear_maxiter=20,
    approx_order=2,
    energy_weight=0.0,
    regularization_weight=0.0,
    extrapolate=True,
)
interp.set_training_values(xt, yt)
interp.set_training_derivatives(xt, dyt_dxt[:, :, 0], 0)
interp.set_training_derivatives(xt, dyt_dxt[:, :, 1], 1)
interp.set_training_derivatives(xt, dyt_dxt[:, :, 2], 2)
interp.train()

plot_b777_engine(xt, yt, xlimits, interp)
