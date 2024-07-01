from math import pi as π
import numpy as np
from smt.surrogate_models import RMTB


dimension = 2


def generate_sine_surface_data(samples=20):
    training_points = np.full(shape=(samples, dimension), fill_value=np.nan)
    training_values = np.full(shape=samples, fill_value=np.nan)

    rng = np.random.default_rng(12345)
    for i in range(samples):
        x = rng.uniform(-1, 1, dimension)
        training_points[i, :] = x
        v = 1
        for j in range(dimension):
            v *= np.sin(π * x[j])
        training_values[i] = v
    return training_points, training_values


def test_rmtb_surrogate_model():
    training_points, training_values = generate_sine_surface_data()
    limits = np.full(shape=(dimension, 2), fill_value=np.nan)
    for i in range(dimension):
        limits[i, 0] = -1
        limits[i, 1] = 1

    model = RMTB(
        xlimits=limits,
    )
    model.set_training_values(training_points, training_values)
    model.train()
    computed_values = model.predict_values(training_points)
    for i in range(len(computed_values)):
        expected = training_values[i]
        # TODO: Fix the shape of the results:
        computed = computed_values[i][0]
        assert np.isclose(expected, computed, atol=0.2)
