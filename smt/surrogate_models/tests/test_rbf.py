"""
Regression tests of RBF against the legacy C++ implementation.

This package is distributed under New BSD license.
"""

import unittest

import numpy as np

from smt.surrogate_models import RBF
from smt.surrogate_models import rbf_idw_backend as bk

try:
    import jax
    import jax.numpy as jnp

    HAS_JAX = True
except ImportError:
    HAS_JAX = False


def _data(seed, nt, nx, ny, ne):
    rng = np.random.RandomState(seed)
    xt = rng.rand(nt, nx)
    yt = np.sin(3.0 * xt.sum(axis=1, keepdims=True)) + rng.rand(nt, ny)
    x = rng.rand(ne, nx)
    # one evaluation point coincides with a training point
    x[0] = xt[2]
    return xt, yt, x


# name -> (options, (seed, nt, nx, ny, n_eval))
CASES = {
    "rbf_1d_poly-1": ({"d0": 0.5}, (1, 6, 1, 1, 5)),
    "rbf_1d_poly1_reg": ({"d0": 0.3, "poly_degree": 1, "reg": 1e-06}, (2, 7, 1, 1, 5)),
    "rbf_2d_poly0_vec_d0": (
        {"d0": [0.7, 1.3], "poly_degree": 0, "reg": 1e-08},
        (3, 9, 2, 2, 5),
    ),
    "rbf_5d_poly1": ({"d0": 1.5, "poly_degree": 1, "reg": 1e-10}, (4, 14, 5, 1, 6)),
}

# Reference values captured from the legacy C++/Cython implementation (smt master dde28bd2).
REF = {
    "rbf_1d_poly-1": {
        "y": np.array(
            [
                [0.3971114161395235],
                [15.559896391712755],
                [0.7768775732183713],
                [-0.8140761915637995],
                [1.1327342240838334],
            ]
        ),
        "dy_dx0": np.array(
            [
                [18.745411165247788],
                [116.48929389677505],
                [9.560444119393651],
                [26.532724858348956],
                [-10.088803487131372],
            ]
        ),
        "dy_dyt": np.array(
            [
                [
                    [-3.2543539418838918e-07],
                    [2.0565494196489453e-08],
                    [9.9999949770062813e-01],
                    [8.6109503172338009e-07],
                    [-2.4116015993058681e-06],
                    [2.3561005946248770e-06],
                ],
                [
                    [-1.0227841005582377e01],
                    [2.2335168260801197e00],
                    [-6.7162472589925528e00],
                    [2.0416490209621770e01],
                    [-4.1610135249353334e01],
                    [3.6829099921094894e01],
                ],
                [
                    [-2.6692346218624152e-02],
                    [1.0831603829046799e-03],
                    [4.9477009816473583e-01],
                    [9.5349984927452169e-02],
                    [-5.7648313543177210e-01],
                    [1.0120955300808419e00],
                ],
                [
                    [1.7807022057986615e00],
                    [6.7009756132392795e-01],
                    [8.5015977583134372e-01],
                    [-3.0650638971528679e00],
                    [5.5982240146695403e00],
                    [-4.8283789184497437e00],
                ],
                [
                    [1.0038431045444793e00],
                    [6.8635499391279126e-05],
                    [8.6867109510427559e-04],
                    [-6.1937324595140808e-03],
                    [6.8669782047447083e-03],
                    [-5.4504868064997576e-03],
                ],
            ]
        ),
    },
    "rbf_1d_poly1_reg": {
        "y": np.array(
            [
                [1.263863947142454],
                [-6.497589655415368],
                [-7.865567021433662],
                [1.7461651796579645],
                [-7.767463189518708],
            ]
        ),
        "dy_dx0": np.array(
            [
                [-15.920599171742651],
                [-27.473650166983134],
                [-12.461990865965163],
                [-2.0769130807839247],
                [-14.010663586949704],
            ]
        ),
        "dy_dyt": np.array(
            [
                [
                    [1.5572215123555531e-03],
                    [7.7920327223975024e-06],
                    [9.9988006873070190e-01],
                    [1.1050179664188553e-03],
                    [-2.9249396680252338e-03],
                    [4.6608199499257268e-04],
                    [-9.1242607823005528e-05],
                ],
                [
                    [-8.9650099933279122e01],
                    [-7.5400598654893625e-01],
                    [1.0838127461380010e01],
                    [-7.5322759375536634e01],
                    [1.7575640837504153e02],
                    [-2.4602555500183101e01],
                    [4.7348849590897544e00],
                ],
                [
                    [-1.1609244660515257e02],
                    [-1.1782921820713905e00],
                    [1.3083875660066850e01],
                    [-9.7271827036400865e01],
                    [2.2907281362029204e02],
                    [-3.3378077886429232e01],
                    [6.7639544297099974e00],
                ],
                [
                    [2.1840987563683840e00],
                    [2.4929553144785004e-03],
                    [2.1687286473418155e-01],
                    [1.9053508323964499e00],
                    [-3.5197659945829889e00],
                    [2.3624765786154506e-01],
                    [-2.5297072197190573e-02],
                ],
                [
                    [-1.1374389184994398e02],
                    [-1.1323920490705939e00],
                    [1.2893324573038409e01],
                    [-9.5324318124484250e01],
                    [2.2430300526273788e02],
                    [-3.2559349494535930e01],
                    [6.5636216822438200e00],
                ],
            ]
        ),
    },
    "rbf_2d_poly0_vec_d0": {
        "y": np.array(
            [
                [-0.35239248703749415, -0.6359796250311334],
                [1.9219781120665194, 5.9027927933123205],
                [1.1805503213498478, 4.556065418814761],
                [1.6010220265733404, 3.5997532759722652],
                [-1.3447988310548065, -5.583271987503714],
            ]
        ),
        "dy_dx0": np.array(
            [
                [8.296841688011355, 36.277884672908556],
                [1.3789166260730503, 15.53304225683982],
                [-0.04734941262853947, 13.006087138572356],
                [2.3481044605937296, 25.47560458568516],
                [-1.7719877381223408, 12.916827703332793],
            ]
        ),
        "dy_dx1": np.array(
            [
                [-4.18140143400885, -0.35136382521062304],
                [-1.8338005079094333, -0.934962391840563],
                [-4.541647324008221, -15.994741238817651],
                [-2.093750782030517, -6.550200933881765],
                [-7.52040633881028, -36.23897437828816],
            ]
        ),
        "dy_dyt": np.array(
            [
                [
                    [-1.2397435966704506e-07, -1.2397435966704506e-07],
                    [-2.6438283384777606e-07, -2.6438283384777606e-07],
                    [9.9999984368269068e-01, 9.9999984368269068e-01],
                    [2.3559687178931199e-07, 2.3559687178931199e-07],
                    [9.4529605121351779e-07, 9.4529605121351779e-07],
                    [-1.6237336240010336e-06, -1.6237336240010336e-06],
                    [-2.0227315644660848e-07, -2.0227315644660848e-07],
                    [5.1524648370104842e-07, 5.1524648370104842e-07],
                    [6.7454413965606363e-07, 6.7454413965606363e-07],
                ],
                [
                    [-3.6887703163461083e-01, -3.6887703163461083e-01],
                    [1.4299591623548622e00, 1.4299591623548622e00],
                    [3.2923828253913712e-02, 3.2923828253913712e-02],
                    [1.0970105586811769e00, 1.0970105586811769e00],
                    [-6.3131615744423470e00, -6.3131615744423470e00],
                    [4.9835148819147435e00, 4.9835148819147435e00],
                    [3.2319565629711633e-01, 3.2319565629711633e-01],
                    [-7.4874755529890535e-02, -7.4874755529890535e-02],
                    [-1.0969072589614370e-01, -1.0969072589614370e-01],
                ],
                [
                    [-1.4362987320146203e-01, -1.4362987320146203e-01],
                    [1.4172944401366294e00, 1.4172944401366294e00],
                    [-3.9851769600602438e-02, -3.9851769600602438e-02],
                    [2.1761438940870903e-01, 2.1761438940870903e-01],
                    [-4.2373302543128375e00, -4.2373302543128375e00],
                    [4.1059597186795145e00, 4.1059597186795145e00],
                    [4.2115258805438316e-02, 4.2115258805438316e-02],
                    [2.3894155482548740e-01, 2.3894155482548740e-01],
                    [-6.0111346474104721e-01, -6.0111346474104721e-01],
                ],
                [
                    [-3.5416030059423065e-01, -3.5416030059423065e-01],
                    [1.1195895313994697e00, 1.1195895313994697e00],
                    [1.6734082115448246e-02, 1.6734082115448246e-02],
                    [7.5483027236145972e-01, 7.5483027236145972e-01],
                    [-2.7036280527099734e00, -2.7036280527099734e00],
                    [2.1500703947494912e00, 2.1500703947494912e00],
                    [6.9295781296659698e-02, 6.9295781296659698e-02],
                    [9.1078691197935768e-02, 9.1078691197935768e-02],
                    [-1.4381039981208232e-01, -1.4381039981208232e-01],
                ],
                [
                    [2.3877999710033366e00, 2.3877999710033366e00],
                    [-1.0319674075435279e00, -1.0319674075435279e00],
                    [1.1565022709650186e-01, 1.1565022709650186e-01],
                    [-1.0574657690769707e-01, -1.0574657690769707e-01],
                    [5.4368445948829631e00, 5.4368445948829631e00],
                    [-5.9772803821776428e00, -5.9772803821776428e00],
                    [3.4298478210248828e-01, 3.4298478210248828e-01],
                    [-1.4745781910728741e00, -1.4745781910728741e00],
                    [1.3062929826162417e00, 1.3062929826162417e00],
                ],
            ]
        ),
    },
    "rbf_5d_poly1": {
        "y": np.array(
            [
                [0.9612478972088603],
                [1.114790142432021],
                [1.0472041634722515],
                [1.0532351466924972],
                [1.2538800825976608],
                [1.0069951247586033],
            ]
        ),
        "dy_dx0": np.array(
            [
                [-0.008306301325271104],
                [-1.981367300808353],
                [1.3758095299616775],
                [0.19066184246994844],
                [-1.76034068011751],
                [0.8223161502401448],
            ]
        ),
        "dy_dx1": np.array(
            [
                [-1.4973562973369958],
                [1.2016667920983632],
                [1.5623105756091142],
                [1.9777790279208105],
                [1.0015307962243978],
                [0.46574578080164386],
            ]
        ),
        "dy_dx2": np.array(
            [
                [-1.4748329194667011],
                [0.4445230621534231],
                [0.9933871493979527],
                [0.8141464298284928],
                [0.47484490650715117],
                [1.627808465921941],
            ]
        ),
        "dy_dx3": np.array(
            [
                [-0.4432053503348996],
                [0.6427003066590151],
                [0.39705579861895846],
                [0.3829327666568778],
                [-0.017161315443074476],
                [0.23687700526911404],
            ]
        ),
        "dy_dx4": np.array(
            [
                [0.387023936747954],
                [-1.2959009570678033],
                [0.009859811595565926],
                [-1.0090126223671159],
                [-0.8175818636520938],
                [1.075696737553697],
            ]
        ),
        "dy_dyt": np.array(
            [
                [
                    [1.8584866978699210e-10],
                    [-7.3831163405202460e-11],
                    [9.9999999958300179e-01],
                    [-1.4651391211373266e-11],
                    [4.1323566790651967e-10],
                    [4.9230219900664451e-10],
                    [-2.4773783025011653e-10],
                    [3.1313618364947615e-11],
                    [1.2104806046409067e-10],
                    [-3.4203218035600003e-10],
                    [1.0769074521022048e-10],
                    [-4.6481396509534534e-10],
                    [-1.9710455489985179e-11],
                    [2.2834001356386580e-10],
                ],
                [
                    [4.7246809004293233e-02],
                    [2.6532082293572268e-01],
                    [-1.8780477093393089e-01],
                    [2.7897476466174354e-02],
                    [1.4297461829905611e-01],
                    [-3.8217900913558456e-01],
                    [-8.8111928712256038e-02],
                    [-7.0587336299532666e-01],
                    [1.7283761216122429e-01],
                    [7.9820506889444509e-01],
                    [1.9859780253242221e-01],
                    [-8.0839315574975146e-02],
                    [8.3710761199905548e-01],
                    [-4.5379434940331365e-02],
                ],
                [
                    [-2.0254614740274990e-01],
                    [-8.6257062764209858e-02],
                    [1.6772634581034573e-01],
                    [2.8074475456804393e-01],
                    [-6.2979913488643291e-03],
                    [5.3630594627423456e-01],
                    [-3.2406376145804217e-02],
                    [9.6284423827319188e-01],
                    [4.8149026025932073e-02],
                    [-1.8308142286490359e-01],
                    [-1.6782020645094420e-01],
                    [-3.2308775846604476e-01],
                    [-3.9458361391364605e-02],
                    [4.5185015883143720e-02],
                ],
                [
                    [-7.4964476269324010e-02],
                    [-5.5052452639198890e-02],
                    [6.2173927831625120e-02],
                    [3.8559177038016568e-01],
                    [-2.7234205249424548e-02],
                    [2.9825179218132636e-01],
                    [9.4397149048507600e-02],
                    [1.0122571355163359e00],
                    [-2.3658243810601221e-01],
                    [-1.0925087315696302e-02],
                    [-3.9221377778975519e-01],
                    [-1.7427597449048449e-01],
                    [2.4040877073180056e-01],
                    [-1.2183213382985336e-01],
                ],
                [
                    [1.7720379299796196e-01],
                    [2.5710563009370599e-03],
                    [-6.7173592089104894e-02],
                    [-5.2662562455772138e-02],
                    [-6.6299973458205902e-02],
                    [-2.8650648263988970e-01],
                    [-2.4985411410491579e-01],
                    [-1.0970569029894688e00],
                    [5.6963050761770728e-01],
                    [6.3001726455913254e-01],
                    [-1.7433891879531582e-01],
                    [2.1252872896567518e-01],
                    [9.0445062482512573e-01],
                    [4.9749057126614193e-01],
                ],
                [
                    [-1.6832452825259592e-01],
                    [-3.2075801008830944e-02],
                    [1.7149192022913584e-01],
                    [5.5518788538773636e-02],
                    [-6.6878469160748480e-02],
                    [3.1594351839632806e-01],
                    [-4.3745749698137326e-02],
                    [2.0696543064613127e-01],
                    [3.7627440407892447e-02],
                    [-5.5423191568947083e-01],
                    [-4.9306479461680404e-03],
                    [-1.1666995181934697e-01],
                    [3.0159681366075834e-02],
                    [1.1691502839909702e00],
                ],
            ]
        ),
    },
}


class TestRBFCppRegression(unittest.TestCase):
    def _check(self, name):
        opts, dspec = CASES[name]
        xt, yt, x = _data(*dspec)
        sm = RBF(print_global=False, **opts)
        sm.set_training_values(xt, yt)
        sm.train()
        ref = REF[name]
        np.testing.assert_allclose(sm.predict_values(x), ref["y"], rtol=1e-9, atol=1e-9)
        for k in range(xt.shape[1]):
            np.testing.assert_allclose(
                sm.predict_derivatives(x, k), ref["dy_dx%d" % k], rtol=1e-9, atol=1e-9
            )
        # output_derivatives uses inv(self.mtx), which has cond ~6e6 on rbf_1d_poly1_reg;
        # allow 1e-8 to accommodate BLAS/LAPACK rounding differences across platforms.
        np.testing.assert_allclose(
            sm.predict_output_derivatives(x)[None],
            ref["dy_dyt"],
            rtol=1e-8,
            atol=1e-8,
        )

    def test_cpp_regression_rbf(self):
        """Regression test against outputs captured from legacy C++ RBF (smt master dde28bd2)"""
        for name in CASES:
            with self.subTest(case=name):
                self._check(name)

    def test_d0_broadcast_and_validation(self):
        rng = np.random.RandomState(42)
        xt = rng.rand(8, 2)
        yt = np.sin(xt[:, :1]) + xt[:, 1:2]
        x = rng.rand(4, 2)

        ref = RBF(d0=1.5, print_global=False)
        ref.set_training_values(xt, yt)
        ref.train()
        y_ref = ref.predict_values(x)
        dy_ref = ref.predict_derivatives(x, 1)

        for d0_val in (np.float32(1.5), np.array(1.5), [1.5]):
            sm = RBF(d0=d0_val, print_global=False)
            sm.set_training_values(xt, yt)
            sm.train()
            np.testing.assert_allclose(
                sm.predict_values(x), y_ref, rtol=1e-6, atol=1e-6
            )
            np.testing.assert_allclose(
                sm.predict_derivatives(x, 1), dy_ref, rtol=1e-6, atol=1e-6
            )

        for bad_d0 in ([1.0, 2.0, 3.0], 0.0, [-1.0, 1.0]):
            sm = RBF(d0=bad_d0, print_global=False)
            sm.set_training_values(xt, yt)
            with self.assertRaises(ValueError):
                sm.train()


class TestRBFJAX(unittest.TestCase):
    @unittest.skipUnless(HAS_JAX, "JAX not installed")
    def test_jax_predictor_and_autodiff(self):
        rng = np.random.RandomState(42)
        xt = rng.rand(12, 2)
        yt = np.sin(3.0 * xt[:, :1]) + 0.5 * np.cos(2.0 * xt[:, 1:2])
        x_eval_np = np.array([[0.23, 0.77], [0.81, 0.19]])

        prev_x64 = bool(jax.config.jax_enable_x64)
        for poly_degree in (-1, 0, 1):
            sm = RBF(d0=[1.2, 0.8], poly_degree=poly_degree, print_global=False)
            sm.set_training_values(xt, yt)
            sm.train()

            y_np = sm.predict_values(x_eval_np)
            pred_default = sm.get_jax_predictor()
            self.assertEqual(bool(jax.config.jax_enable_x64), prev_x64)
            y_default = np.asarray(jax.jit(pred_default)(jnp.asarray(x_eval_np)))
            np.testing.assert_allclose(y_default, y_np, atol=1e-4)

            val_1d_default = pred_default(jnp.asarray(x_eval_np[0]))
            self.assertEqual(val_1d_default.ndim, 0)
            np.testing.assert_allclose(float(val_1d_default), y_np[0, 0], atol=1e-4)

            with bk._enable_x64(jax):
                pred_jax = sm.get_jax_predictor()
                x_eval = jnp.array(x_eval_np, dtype=jnp.float64)
                np.testing.assert_allclose(
                    np.asarray(pred_jax(x_eval)), y_np, atol=1e-12
                )

                vmap_vals = np.asarray(jax.vmap(pred_jax)(x_eval))
                np.testing.assert_allclose(vmap_vals, y_np[:, 0], atol=1e-12)

                grad_fn = jax.jacobian(pred_jax)
                hess_fn = jax.hessian(pred_jax)
                eps = 1e-6
                for idx in range(x_eval_np.shape[0]):
                    g = np.asarray(grad_fn(x_eval[idx]))
                    dy_0 = sm.predict_derivatives(x_eval_np[idx : idx + 1], 0)[0, 0]
                    dy_1 = sm.predict_derivatives(x_eval_np[idx : idx + 1], 1)[0, 0]
                    np.testing.assert_allclose(g, [dy_0, dy_1], atol=1e-10)

                    h = np.asarray(hess_fn(x_eval[idx]))
                    h_fd = np.zeros((2, 2))
                    for col in range(2):
                        step = np.zeros((1, 2))
                        step[0, col] = eps
                        x_p = x_eval_np[idx : idx + 1] + step
                        x_m = x_eval_np[idx : idx + 1] - step
                        for row in range(2):
                            h_fd[row, col] = (
                                sm.predict_derivatives(x_p, row)[0, 0]
                                - sm.predict_derivatives(x_m, row)[0, 0]
                            ) / (2.0 * eps)
                    np.testing.assert_allclose(h, h_fd, atol=1e-7)

        self.assertEqual(bool(jax.config.jax_enable_x64), prev_x64)

    @unittest.skipUnless(HAS_JAX, "JAX not installed")
    def test_jax_predictor_multioutput_and_untrained(self):
        sm_untrained = RBF(print_global=False)
        with self.assertRaises(RuntimeError):
            sm_untrained.get_jax_predictor()

        rng = np.random.RandomState(11)
        xt = rng.rand(10, 2)
        yt = np.column_stack([np.sin(xt[:, 0]), np.cos(xt[:, 1])])

        sm = RBF(d0=0.9, poly_degree=1, print_global=False)
        sm.set_training_values(xt, yt)
        sm.train()

        with bk._enable_x64(jax):
            pred = sm.get_jax_predictor()
            x_pt = np.array([0.4, 0.6])
            out_1d = np.asarray(pred(jnp.asarray(x_pt)))
            self.assertEqual(out_1d.shape, (2,))
            np.testing.assert_allclose(
                out_1d, sm.predict_values(x_pt.reshape(1, -1))[0], atol=1e-12
            )
            with self.assertRaises(ValueError):
                pred(0.5)

            # 0D scalar input & vmap for 1D model
            sm_1d = RBF(d0=0.8, poly_degree=1, print_global=False)
            sm_1d.set_training_values(
                np.array([[0.0], [0.5], [1.0]]), np.array([[0.0], [1.0], [2.0]])
            )
            sm_1d.train()
            pred_1d = sm_1d.get_jax_predictor()
            out_0d = pred_1d(0.5)
            self.assertEqual(out_0d.ndim, 0)
            np.testing.assert_allclose(float(out_0d), 1.0, atol=1e-10)
            vmap_1d = np.asarray(jax.vmap(pred_1d)(jnp.array([0.0, 0.5, 1.0])))
            np.testing.assert_allclose(vmap_1d, [0.0, 1.0, 2.0], atol=1e-10)

            # Single training point (nt = 1, poly_degree = -1) with vmap and Hessian
            sm_single = RBF(d0=0.8, poly_degree=-1, print_global=False)
            sm_single.set_training_values(np.array([[0.5]]), np.array([[1.5]]))
            sm_single.train()
            pred_single = sm_single.get_jax_predictor()
            x_single = jnp.array([0.3, 0.5, 0.8])
            np.testing.assert_allclose(
                np.asarray(jax.vmap(pred_single)(x_single)),
                sm_single.predict_values(np.array([[0.3], [0.5], [0.8]]))[:, 0],
                atol=1e-12,
            )
            hess_center = float(jax.hessian(pred_single)(jnp.array(0.5)))
            np.testing.assert_allclose(hess_center, -2.0 * 1.5 / (0.8**2), atol=1e-12)


if __name__ == "__main__":
    unittest.main()
