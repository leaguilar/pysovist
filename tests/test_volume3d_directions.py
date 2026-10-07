"""Direction sets on the sphere and on the horizontal ring."""

import math

import numpy as np
import pytest

from pysovist.directions import equiangular, fibonacci_sphere, random_sphere, solid_angle_weights


@pytest.mark.parametrize("n", [1, 2, 7, 1024, 20070])
def test_fibonacci_unit_vectors_with_zero_mean(n):
    d = fibonacci_sphere(n)
    assert d.shape == (n, 3)
    np.testing.assert_allclose(np.linalg.norm(d, axis=1), 1.0, rtol=0, atol=1e-14)
    if n > 1:
        # z is exactly symmetric, x and y cancel up to O(1/n).
        assert abs(d[:, 2].mean()) < 1e-14
        assert np.abs(d[:, :2].mean(axis=0)).max() < 2.0 / n


def test_fibonacci_is_deterministic_and_equal_area():
    a, b = fibonacci_sphere(4096), fibonacci_sphere(4096)
    np.testing.assert_array_equal(a, b)
    # Equal area: the fraction of directions in a polar cap equals the cap's area fraction.
    for h in (0.1, 0.5, 1.3):
        frac = (a[:, 2] > 1 - h).mean()
        assert frac == pytest.approx(h / 2, abs=1.0 / 4096)


@pytest.mark.parametrize("n", [1000, 20070])
def test_random_unit_vectors_with_zero_mean(n):
    d = random_sphere(n, seed=1337)
    assert d.shape == (n, 3)
    np.testing.assert_allclose(np.linalg.norm(d, axis=1), 1.0, rtol=0, atol=1e-14)
    # Each coordinate of a uniform unit vector has variance 1/3.
    assert np.abs(d.mean(axis=0)).max() < 4 * math.sqrt(1 / 3 / n)
    np.testing.assert_allclose(d.var(axis=0), 1 / 3, atol=8 * math.sqrt(4 / 45 / n))


def test_random_is_reproducible_with_a_seed():
    np.testing.assert_array_equal(random_sphere(100, seed=7), random_sphere(100, seed=7))
    assert not np.array_equal(random_sphere(100, seed=7), random_sphere(100, seed=8))


@pytest.mark.parametrize("n", [1, 1024, 20070])
def test_equal_weights_sum_to_full_solid_angle(n):
    w = solid_angle_weights(n)
    assert w.shape == (n,)
    assert np.all(w == w[0])
    assert w.sum() == pytest.approx(4 * math.pi, rel=1e-14)


def test_equiangular_ring():
    d = equiangular(8)
    assert d.shape == (8, 3)
    np.testing.assert_allclose(d[:, 2], 0.0)
    np.testing.assert_allclose(np.linalg.norm(d, axis=1), 1.0, atol=1e-15)
    np.testing.assert_allclose(d[0], [1, 0, 0], atol=1e-15)
    np.testing.assert_allclose(d[2], [0, 1, 0], atol=1e-15)
    np.testing.assert_allclose(d.sum(axis=0), 0.0, atol=1e-14)
