"""
Regression tests of IDW against the legacy C++ implementation.

This package is distributed under New BSD license.
"""

import unittest

import numpy as np

from smt.surrogate_models import IDW
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
    # one evaluation point coincides with a training point (exercises the exact-hit branch of IDW)
    x[0] = xt[2]
    return xt, yt, x


# name -> (options, (seed, nt, nx, ny, n_eval))
CASES = {
    "idw_1d_p2.5": ({"p": 2.5}, (5, 6, 1, 1, 5)),
    "idw_2d_p1": ({"p": 1.0}, (6, 9, 2, 2, 5)),
    "idw_4d_p3.5": ({"p": 3.5}, (7, 12, 4, 1, 6)),
}

# Reference values captured from the legacy C++/Cython implementation (smt master dde28bd2).
REF = {
    "idw_1d_p2.5": {
        "y": np.array(
            [
                [0.8779638131093891],
                [1.0488229438506576],
                [1.0112317728524547],
                [1.208863164305348],
                [1.1246202228653144],
            ]
        ),
        "dy_dx0": np.array(
            [
                [0.0],
                [-1.440420578976031],
                [-4.0014958901838025],
                [-1.2452919966924851],
                [-0.8997736579833506],
            ]
        ),
        "dy_dyt": np.array(
            [
                [
                    [0.0000000000000000e00],
                    [0.0000000000000000e00],
                    [1.0000000000000000e00],
                    [0.0000000000000000e00],
                    [0.0000000000000000e00],
                    [0.0000000000000000e00],
                ],
                [
                    [3.3188680233859053e-01],
                    [7.9288457026979709e-04],
                    [6.5876756917323931e-01],
                    [6.7389150018397185e-04],
                    [5.4254502357563078e-03],
                    [2.4534021819600997e-03],
                ],
                [
                    [2.2520000690291579e-05],
                    [9.7277734479834599e-01],
                    [2.1264315026116685e-05],
                    [2.6884144680484928e-02],
                    [8.2440271622371639e-05],
                    [2.1228593383023251e-04],
                ],
                [
                    [6.3759349810843780e-01],
                    [1.4361731891684019e-03],
                    [3.3525510790049162e-01],
                    [1.1841313503438916e-03],
                    [1.8570246470631612e-02],
                    [5.9608429809266776e-03],
                ],
                [
                    [7.2575417819555149e-02],
                    [8.3524497979582963e-03],
                    [5.9948886439480979e-02],
                    [6.5091395537221841e-03],
                    [7.8478079638393627e-01],
                    [6.7833310005347147e-02],
                ],
            ]
        ),
    },
    "idw_2d_p1": {
        "y": np.array(
            [
                [1.6612489762989338, 1.5954850576858972],
                [0.6062338849355763, 0.8042375993365201],
                [0.3226309973942431, 0.5105483894359021],
                [0.2763748145500884, 0.48229176600501533],
                [0.6143132918616588, 0.7350692430690066],
            ]
        ),
        "dy_dx0": np.array(
            [
                [0.0, 0.0],
                [-0.5473379253127285, -0.48088657326863493],
                [-0.0775027172897266, -0.07993501927000193],
                [-1.0126779292559167, -0.8932280395731309],
                [0.9096938789533375, -0.036334678881428155],
            ]
        ),
        "dy_dx1": np.array(
            [
                [0.0, 0.0],
                [0.21615770219601568, 0.4205145760755773],
                [-0.7101244692744453, -0.5393278078725179],
                [-0.26047659214338026, -0.10389147282040569],
                [0.4546035469796154, 0.18111651825333686],
            ]
        ),
        "dy_dyt": np.array(
            [
                [
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [1.0, 1.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                ],
                [
                    [0.0647729353438273, 0.0647729353438273],
                    [0.06193674852703575, 0.06193674852703575],
                    [0.1723773790226454, 0.1723773790226454],
                    [0.15056265287655768, 0.15056265287655768],
                    [0.17744639408668375, 0.17744639408668375],
                    [0.10931562700211467, 0.10931562700211467],
                    [0.13105930342877112, 0.13105930342877112],
                    [0.05897490555838377, 0.05897490555838377],
                    [0.07355405415398054, 0.07355405415398054],
                ],
                [
                    [0.09959778679817011, 0.09959778679817011],
                    [0.06072232420075521, 0.06072232420075521],
                    [0.06395532604824777, 0.06395532604824777],
                    [0.11134618753424369, 0.11134618753424369],
                    [0.09843435306612322, 0.09843435306612322],
                    [0.13507326704004954, 0.13507326704004954],
                    [0.15397327105252984, 0.15397327105252984],
                    [0.14639562953360696, 0.14639562953360696],
                    [0.13050185472627365, 0.13050185472627365],
                ],
                [
                    [0.05976926022682957, 0.05976926022682957],
                    [0.04725473015709633, 0.04725473015709633],
                    [0.09886098134530594, 0.09886098134530594],
                    [0.08374100917897577, 0.08374100917897577],
                    [0.13085936649407137, 0.13085936649407137],
                    [0.19134188410005384, 0.19134188410005384],
                    [0.11489950283319368, 0.11489950283319368],
                    [0.2043499390751782, 0.2043499390751782],
                    [0.06892332658929524, 0.06892332658929524],
                ],
                [
                    [0.06828895601728077, 0.06828895601728077],
                    [0.06096253553971448, 0.06096253553971448],
                    [0.07436460764293164, 0.07436460764293164],
                    [0.3327329962958534, 0.3327329962958534],
                    [0.11072122488764781, 0.11072122488764781],
                    [0.0837549224439005, 0.0837549224439005],
                    [0.14055143782161691, 0.14055143782161691],
                    [0.04693583461197621, 0.04693583461197621],
                    [0.08168748473907828, 0.08168748473907828],
                ],
            ]
        ),
    },
    "idw_4d_p3.5": {
        "y": np.array(
            [
                [0.8758720452767641],
                [0.4827989132941734],
                [0.3441487478121816],
                [0.07751149549192671],
                [0.05032148794526071],
                [0.30635140843131253],
            ]
        ),
        "dy_dx0": np.array(
            [
                [0.0],
                [0.14247382596501112],
                [0.9812128606027185],
                [-1.5671834750322928],
                [-0.37659909792019586],
                [-0.8238450707279003],
            ]
        ),
        "dy_dx1": np.array(
            [
                [0.0],
                [0.3748032871352298],
                [0.49448117927413227],
                [1.4027746629757254],
                [0.24203584743533868],
                [0.4626486692993708],
            ]
        ),
        "dy_dx2": np.array(
            [
                [0.0],
                [-0.16602854616698343],
                [-0.5727454464175606],
                [0.5009393411105351],
                [-0.0959345992117021],
                [2.722559267820578],
            ]
        ),
        "dy_dx3": np.array(
            [
                [0.0],
                [0.1665226836306667],
                [-0.4591258861871513],
                [2.855784751273801],
                [0.18664754291659402],
                [-0.1269814785985111],
            ]
        ),
        "dy_dyt": np.array(
            [
                [
                    [0.0],
                    [0.0],
                    [1.0],
                    [0.0],
                    [0.0],
                    [0.0],
                    [0.0],
                    [0.0],
                    [0.0],
                    [0.0],
                    [0.0],
                    [0.0],
                ],
                [
                    [0.033680476094531245],
                    [0.01944406439632494],
                    [0.25989949292419195],
                    [0.08262845076499717],
                    [0.025686997875920808],
                    [0.019713802649406982],
                    [0.1781653183090249],
                    [0.03692930891462784],
                    [0.10006093194313918],
                    [0.06015780311887071],
                    [0.10753882142544781],
                    [0.07609453158351659],
                ],
                [
                    [0.0075635518901172715],
                    [0.32466241676157104],
                    [0.014946626019234],
                    [0.006651039978384237],
                    [0.047185608820847234],
                    [0.014278552102920995],
                    [0.025339987064800652],
                    [0.02826912480347676],
                    [0.011401534051018692],
                    [0.3999525082375731],
                    [0.10923066006550519],
                    [0.010518390204550962],
                ],
                [
                    [0.029400645424169616],
                    [0.007094667119306585],
                    [0.17889255100570992],
                    [0.010414246707666705],
                    [0.03908291212383576],
                    [0.009213324712937256],
                    [0.009243036371848633],
                    [0.016728357159699187],
                    [0.036076385121502144],
                    [0.028839412433250937],
                    [0.6146121743879432],
                    [0.02040228743212995],
                ],
                [
                    [0.09671575833582795],
                    [0.0253982310303992],
                    [0.08311905408016458],
                    [0.059610728409093476],
                    [0.07070460663402987],
                    [0.03851581540304843],
                    [0.02274853921562564],
                    [0.11532183449383526],
                    [0.10719988882677697],
                    [0.07057970414520602],
                    [0.267886974582352],
                    [0.04219886484364059],
                ],
                [
                    [0.08930063883968742],
                    [0.0022744102591750323],
                    [0.29967989664299793],
                    [0.04537466954956699],
                    [0.004442755978845253],
                    [0.00792417085590536],
                    [0.006583487318058303],
                    [0.015178338741891468],
                    [0.44566628077461873],
                    [0.006588926677915831],
                    [0.025609119472891254],
                    [0.05137730488844626],
                ],
            ]
        ),
    },
}


class TestIDWCppRegression(unittest.TestCase):
    def _check(self, name):
        opts, dspec = CASES[name]
        xt, yt, x = _data(*dspec)
        sm = IDW(print_global=False, **opts)
        sm.set_training_values(xt, yt)
        sm.train()
        ref = REF[name]
        np.testing.assert_allclose(
            sm.predict_values(x), ref["y"], rtol=1e-10, atol=1e-10
        )
        for k in range(xt.shape[1]):
            np.testing.assert_allclose(
                sm.predict_derivatives(x, k), ref["dy_dx%d" % k], rtol=1e-10, atol=1e-10
            )
        np.testing.assert_allclose(
            sm.predict_output_derivatives(x)[None],
            ref["dy_dyt"],
            rtol=1e-10,
            atol=1e-10,
        )

    def test_cpp_regression_idw(self):
        """Regression test against outputs captured from legacy C++ IDW (smt master dde28bd2)"""
        for name in CASES:
            with self.subTest(case=name):
                self._check(name)


class TestIDWJAX(unittest.TestCase):
    @unittest.skipUnless(HAS_JAX, "JAX not installed")
    def test_jax_predictor_and_autodiff(self):
        rng = np.random.RandomState(42)
        xt = rng.rand(12, 2)
        yt = np.sin(3.0 * xt[:, :1]) + 0.5 * np.cos(2.0 * xt[:, 1:2])

        sm = IDW(p=2.5, print_global=False)
        sm.set_training_values(xt, yt)
        sm.train()

        prev_x64 = bool(jax.config.jax_enable_x64)
        pred_default = sm.get_jax_predictor()
        self.assertEqual(bool(jax.config.jax_enable_x64), prev_x64)

        x_eval_np = np.array([[0.23, 0.77], [0.81, 0.19]])
        y_np = sm.predict_values(x_eval_np)
        y_default = np.asarray(jax.jit(pred_default)(jnp.asarray(x_eval_np)))
        np.testing.assert_allclose(y_default, y_np, atol=1e-5)

        val_1d_default = pred_default(jnp.asarray(x_eval_np[0]))
        self.assertEqual(val_1d_default.ndim, 0)
        np.testing.assert_allclose(float(val_1d_default), y_np[0, 0], atol=1e-5)

        with bk._enable_x64(jax):
            pred_jax = sm.get_jax_predictor()
            x_eval = jnp.array(x_eval_np, dtype=jnp.float64)
            np.testing.assert_allclose(np.asarray(pred_jax(x_eval)), y_np, atol=1e-12)

            def scalar_pred(x_vec):
                return pred_jax(x_vec)

            grad_fn = jax.jacobian(scalar_pred)
            for idx in range(x_eval_np.shape[0]):
                g = np.asarray(grad_fn(x_eval[idx]))
                dy_0 = sm.predict_derivatives(x_eval_np[idx : idx + 1], 0)[0, 0]
                dy_1 = sm.predict_derivatives(x_eval_np[idx : idx + 1], 1)[0, 0]
                np.testing.assert_allclose(g, [dy_0, dy_1], atol=1e-10)

        self.assertEqual(bool(jax.config.jax_enable_x64), prev_x64)

    @unittest.skipUnless(HAS_JAX, "JAX not installed")
    def test_jax_predictor_multioutput_and_edge_cases(self):
        sm_untrained = IDW(p=2.5, print_global=False)
        with self.assertRaises(RuntimeError):
            sm_untrained.get_jax_predictor()

        rng = np.random.RandomState(10)
        xt = rng.rand(8, 2)
        yt = np.column_stack([xt[:, 0] ** 2, np.sin(xt[:, 1])])
        xt_dup = np.vstack([xt, xt[0:1]])
        yt_dup = np.vstack([yt, yt[0:1] + 5.0])  # first exact match must win

        sm = IDW(p=2.5, print_global=False)
        sm.set_training_values(xt_dup, yt_dup)
        sm.train()

        with bk._enable_x64(jax):
            pred = sm.get_jax_predictor()
            out_1d = np.asarray(pred(jnp.asarray(xt_dup[0])))
            self.assertEqual(out_1d.shape, (2,))
            np.testing.assert_allclose(out_1d, yt_dup[0], atol=1e-12)

            def first_out(x_vec):
                return pred(jnp.expand_dims(x_vec, 0))[0, 0]

            g_at_node = np.asarray(jax.jacobian(first_out)(jnp.asarray(xt_dup[0])))
            self.assertTrue(np.all(np.isfinite(g_at_node)))
            np.testing.assert_allclose(g_at_node, 0.0, atol=1e-12)

            # Single training point (nt = 1)
            sm_single = IDW(p=2.0, print_global=False)
            sm_single.set_training_values(np.array([[0.5, 0.5]]), np.array([[3.0]]))
            sm_single.train()
            pred_single = sm_single.get_jax_predictor()
            np.testing.assert_allclose(
                np.asarray(pred_single(jnp.array([[0.1, 0.2], [0.5, 0.5]]))),
                [[3.0], [3.0]],
                atol=1e-12,
            )

            # Distance underflow fallback to nearest training point
            sm_far = IDW(p=100.0, print_global=False)
            sm_far.set_training_values(
                np.array([[0.0, 0.0], [10.0, 10.0]]), np.array([[1.0], [7.0]])
            )
            sm_far.train()
            pred_far = sm_far.get_jax_predictor()
            np.testing.assert_allclose(
                np.asarray(pred_far(jnp.array([[1e4, 1e4]]))), [[7.0]], atol=1e-12
            )
            np.testing.assert_allclose(
                bk.idw_jac(
                    np.array([[1e4, 1e4]]),
                    np.array([[0.0, 0.0], [10.0, 10.0]]),
                    100.0,
                    kx=0,
                    _backend="jax",
                ),
                [[0.0, 0.0]],
                atol=1e-12,
            )

            # 0D scalar input for 1D model, vmap batching & mismatched feature count
            sm_1d = IDW(p=2.0, print_global=False)
            sm_1d.set_training_values(
                np.array([[0.0], [1.0]]), np.array([[1.0], [3.0]])
            )
            sm_1d.train()
            pred_1d = sm_1d.get_jax_predictor()
            out_0d = pred_1d(0.5)
            self.assertEqual(out_0d.ndim, 0)
            np.testing.assert_allclose(float(out_0d), 2.0, atol=1e-12)
            vmap_out = np.asarray(jax.vmap(pred_1d)(jnp.array([0.0, 0.5, 1.0])))
            np.testing.assert_allclose(vmap_out, [1.0, 2.0, 3.0], atol=1e-12)
            with self.assertRaises(ValueError):
                pred(0.5)

            # FP64 near-hit overflow prevention (both predictor/grad and idw_jac)
            sm_near64 = IDW(p=4.0, print_global=False)
            sm_near64.set_training_values(
                np.array([[0.0], [1.0]]), np.array([[1.0], [3.0]])
            )
            sm_near64.train()
            pred_near64 = sm_near64.get_jax_predictor()
            x_near64 = jnp.array(1e-100, dtype=jnp.float64)
            val_near64 = pred_near64(x_near64)
            grad_near64 = jax.grad(pred_near64)(x_near64)
            self.assertTrue(np.isfinite(float(val_near64)))
            self.assertTrue(np.isfinite(float(grad_near64)))
            np.testing.assert_allclose(float(val_near64), 1.0, atol=1e-12)
            np.testing.assert_allclose(float(grad_near64), 0.0, atol=1e-12)
            np.testing.assert_allclose(
                bk.idw_jac(
                    np.array([[1e-100], [0.0]]),
                    np.array([[0.0], [1.0]]),
                    4.0,
                    _backend="jax",
                ),
                [[1.0, 0.0], [1.0, 0.0]],
                atol=1e-12,
            )
            np.testing.assert_allclose(
                bk.idw_jac(
                    np.array([[1e-100], [0.0]]),
                    np.array([[0.0], [1.0]]),
                    4.0,
                    kx=0,
                    _backend="jax",
                ),
                [[0.0, 0.0], [0.0, 0.0]],
                atol=1e-12,
            )

            sm_pair64 = IDW(p=4.0, print_global=False)
            sm_pair64.set_training_values(
                np.array([[0.0], [1e-100]]), np.array([[1.0], [3.0]])
            )
            sm_pair64.train()
            pred_pair64 = sm_pair64.get_jax_predictor()
            x_mid64 = jnp.array(0.5e-100, dtype=jnp.float64)
            np.testing.assert_allclose(float(pred_pair64(x_mid64)), 2.0, atol=1e-12)
            np.testing.assert_allclose(
                float(jax.grad(pred_pair64)(x_mid64)), 8e100, rtol=1e-12
            )
            np.testing.assert_allclose(
                float(jax.grad(pred_pair64)(jnp.array(0.0, dtype=jnp.float64))),
                0.0,
                atol=1e-12,
            )

    @unittest.skipUnless(HAS_JAX, "JAX not installed")
    def test_jax_near_hit_fp32_overflow_guard(self):
        sm = IDW(p=4.0, print_global=False)
        sm.set_training_values(np.array([[0.0], [1.0]]), np.array([[1.0], [3.0]]))
        sm.train()

        prev_x64 = bool(jax.config.jax_enable_x64)
        jax.config.update("jax_enable_x64", False)
        try:
            pred = sm.get_jax_predictor()
            x_near = jnp.array(1e-10, dtype=jnp.float32)
            val_near = pred(x_near)
            grad_near = jax.grad(pred)(x_near)
            self.assertTrue(np.isfinite(float(val_near)))
            self.assertTrue(np.isfinite(float(grad_near)))
            np.testing.assert_allclose(float(val_near), 1.0, atol=1e-6)
            np.testing.assert_allclose(float(grad_near), 0.0, atol=1e-6)

            sm_pair = IDW(p=4.0, print_global=False)
            sm_pair.set_training_values(
                np.array([[0.0], [1e-10]]), np.array([[1.0], [3.0]])
            )
            sm_pair.train()
            pred_pair = sm_pair.get_jax_predictor()
            x_mid = jnp.array(0.5e-10, dtype=jnp.float32)
            np.testing.assert_allclose(float(pred_pair(x_mid)), 2.0, atol=1e-6)
            np.testing.assert_allclose(
                float(jax.grad(pred_pair)(x_mid)), 8e10, rtol=1e-5
            )
            np.testing.assert_allclose(
                float(jax.grad(pred_pair)(jnp.array(0.0, dtype=jnp.float32))),
                0.0,
                atol=1e-6,
            )
        finally:
            jax.config.update("jax_enable_x64", prev_x64)


if __name__ == "__main__":
    unittest.main()
