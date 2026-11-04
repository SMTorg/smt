from smt.examples.rans_crm_wing.rans_crm_wing import (
    get_rans_crm_wing,
    plot_rans_crm_wing_uq_combined,
)
from smt.surrogate_models import RMTBUQ

xt, yt, xlimits = get_rans_crm_wing()

# Same num_ctrl_pts/energy_weight as plain RMTB's own
# run_rans_crm_wing_rmtb.py example, for a direct side-by-side comparison;
# order=4/approx_order=2/solver="dense-lu" are RMTBUQ's own requirements
# (see its docstring) rather than a change to the published example's
# fit. With this little regularization relative to only 35 scattered
# points, RMTB fits each training point almost exactly (edf≈n), so
# RMTBUQ's Wahba sigma2 estimator will warn that it's in a
# near-exact-interpolation regime -- expected here, not a bug.
interp = RMTBUQ(
    num_ctrl_pts=20,
    xlimits=xlimits,
    order=4,
    approx_order=2,
    energy_weight=1e-8,
    regularization_weight=1e-14,
    solver="dense-lu",
)
interp.set_training_values(xt, yt)
interp.train()

plot_rans_crm_wing_uq_combined(xt, yt, xlimits, interp)
