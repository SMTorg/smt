from smt.surrogate_models import RMTB
from smt.examples.one_D_step.one_D_step import get_one_d_step, plot_one_d_step

xt, yt, xlimits = get_one_d_step()

interp = RMTB(
    num_ctrl_pts=100,
    xlimits=xlimits,
    nonlinear_maxiter=20,
    solver_tolerance=1e-16,
    energy_weight=1e-14,
    regularization_weight=0.0,
)
interp.set_training_values(xt, yt)
interp.train()

plot_one_d_step(xt, yt, xlimits, interp)
