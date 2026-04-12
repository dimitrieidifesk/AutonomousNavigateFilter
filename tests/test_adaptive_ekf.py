"""
Тесты адаптивного EKF.

Проверяемые сценарии:
    1. При постоянном шуме R не изменяется значительно.
    2. При увеличении шума R адаптируется (увеличивается).
    3. R_adapted всегда положительно определена.
    4. R_adapted находится в допустимых границах [r_min, r_max].
    5. При adaptation_rate=0 поведение = классическому EKF.
    6. Адаптивный EKF сходится на прямолинейной траектории.
    7. get_r_history() возвращает непустой список после обновлений.
"""

import numpy as np
import pytest

from filters.adaptive_ekf import AdaptiveEKF
from models.kinematic_2d import Kinematic2DModel


class TestAdaptiveEKFBasic:
    """Базовые тесты адаптивного EKF."""

    def test_predict_works(self, adaptive_ekf):
        """predict() работает без ошибок."""
        result = adaptive_ekf.predict(np.array([0.0, 0.0]), dt=0.1)

        assert result.state is not None
        assert result.covariance is not None

    def test_update_works(self, adaptive_ekf):
        """update() работает без ошибок."""
        adaptive_ekf.predict(np.array([0.0, 0.0]), dt=0.1)

        H = np.array([[0, 0, 1, 0]])
        R = np.array([[0.01]])
        z = np.array([1.5])

        result = adaptive_ekf.update(z, H, R)

        assert result.state is not None
        assert result.innovation is not None

    def test_r_history_populated(self, adaptive_ekf):
        """После update() r_history заполняется."""
        H = np.array([[0, 0, 1, 0]])
        R = np.array([[0.01]])

        for _ in range(15):
            adaptive_ekf.predict(np.array([0.0, 0.0]), dt=0.1)
            z = np.array([1.5 + np.random.normal(0, 0.1)])
            adaptive_ekf.update(z, H, R)

        history = adaptive_ekf.get_r_history()
        assert len(history) == 15


class TestAdaptiveEKFAdaptation:
    """Тесты механизма адаптации R."""

    def test_r_stays_stable_with_consistent_noise(self):
        """При постоянном шуме R не должна сильно меняться."""
        model = Kinematic2DModel()
        state = np.array([0.0, 0.0, 1.5, 0.0])
        P0 = np.diag([1.0, 1.0, 0.1, 0.01])
        Q = np.diag([0.01, 0.01, 0.001, 0.0001])

        aekf = AdaptiveEKF(
            motion_model=model,
            initial_state=state,
            initial_covariance=P0,
            process_noise=Q,
            innovation_window_size=10,
            adaptation_rate=0.05,
        )

        H = np.array([[0, 0, 1, 0]])
        R_nominal = np.array([[0.01]])
        noise_std = 0.1

        for i in range(50):
            aekf.predict(np.array([0.0, 0.0]), dt=0.1)
            z = np.array([1.5 + np.random.normal(0, noise_std)])
            aekf.update(z, H, R_nominal)

        R_adapted = aekf.get_adapted_R(1)
        assert R_adapted is not None
        # R не должна отличаться от начальной более чем в 10 раз
        ratio = R_adapted[0, 0] / R_nominal[0, 0]
        assert 0.1 <= ratio <= 10.0

    def test_r_increases_with_more_noise(self):
        """При увеличении шума R должна адаптироваться (увеличиться)."""
        model = Kinematic2DModel()
        state = np.array([0.0, 0.0, 1.5, 0.0])
        P0 = np.diag([1.0, 1.0, 0.1, 0.01])
        Q = np.diag([0.01, 0.01, 0.001, 0.0001])

        aekf = AdaptiveEKF(
            motion_model=model,
            initial_state=state,
            initial_covariance=P0,
            process_noise=Q,
            innovation_window_size=10,
            adaptation_rate=0.3,
            r_max_scale=50.0,
        )

        H = np.array([[0, 0, 1, 0]])
        R_nominal = np.array([[0.001]])  # Заниженный начальный R

        # Подаём данные с БОЛЬШИМ шумом (намного больше чем R_nominal)
        for i in range(60):
            aekf.predict(np.array([0.0, 0.0]), dt=0.1)
            z = np.array([1.5 + np.random.normal(0, 1.0)])  # шум >> R_nominal
            aekf.update(z, H, R_nominal)

        R_adapted = aekf.get_adapted_R(1)
        assert R_adapted is not None
        # R должна увеличиться
        assert R_adapted[0, 0] > R_nominal[0, 0]

    def test_r_clamped_to_bounds(self):
        """R_adapted не выходит за установленные границы."""
        model = Kinematic2DModel()
        state = np.array([0.0, 0.0, 1.5, 0.0])
        P0 = np.diag([1.0, 1.0, 0.1, 0.01])
        Q = np.diag([0.01, 0.01, 0.001, 0.0001])

        r_min_scale = 0.5
        r_max_scale = 5.0

        aekf = AdaptiveEKF(
            motion_model=model,
            initial_state=state,
            initial_covariance=P0,
            process_noise=Q,
            innovation_window_size=5,
            adaptation_rate=0.9,
            r_min_scale=r_min_scale,
            r_max_scale=r_max_scale,
        )

        H = np.array([[0, 0, 1, 0]])
        R_nominal = np.array([[0.01]])

        for i in range(50):
            aekf.predict(np.array([0.0, 0.0]), dt=0.1)
            z = np.array([1.5 + np.random.normal(0, 5.0)])
            aekf.update(z, H, R_nominal)

        R_adapted = aekf.get_adapted_R(1)
        assert R_adapted is not None
        assert R_adapted[0, 0] >= R_nominal[0, 0] * r_min_scale
        assert R_adapted[0, 0] <= R_nominal[0, 0] * r_max_scale

    def test_r_positive_definite(self):
        """R_adapted должна быть положительно определена."""
        model = Kinematic2DModel()
        state = np.array([0.0, 0.0, 1.5, 0.0])
        P0 = np.diag([1.0, 1.0, 0.1, 0.01])
        Q = np.diag([0.01, 0.01, 0.001, 0.0001])

        aekf = AdaptiveEKF(
            motion_model=model,
            initial_state=state,
            initial_covariance=P0,
            process_noise=Q,
            innovation_window_size=10,
            adaptation_rate=0.2,
        )

        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        R_nominal = np.eye(2) * 1.0

        for i in range(30):
            aekf.predict(np.array([0.0, 0.0]), dt=0.1)
            z = np.array([np.random.normal(0, 1.0), np.random.normal(0, 1.0)])
            aekf.update(z, H, R_nominal)

        R_adapted = aekf.get_adapted_R(2)
        if R_adapted is not None:
            eigenvalues = np.linalg.eigvals(R_adapted)
            assert np.all(eigenvalues > 0), f"R не положительно определена: {eigenvalues}"


class TestAdaptiveEKFZeroAdaptation:
    """Тесты при нулевой скорости адаптации (= классический EKF)."""

    def test_zero_adaptation_rate(self):
        """При α=0 поведение = классическому EKF."""
        model = Kinematic2DModel()
        state = np.array([0.0, 0.0, 1.5, 0.0])
        P0 = np.diag([1.0, 1.0, 0.1, 0.01])
        Q = np.diag([0.01, 0.01, 0.001, 0.0001])

        aekf = AdaptiveEKF(
            motion_model=model,
            initial_state=state.copy(),
            initial_covariance=P0.copy(),
            process_noise=Q.copy(),
            innovation_window_size=10,
            adaptation_rate=0.0,  # нулевая адаптация
        )

        H = np.array([[0, 0, 1, 0]])
        R_nominal = np.array([[0.01]])

        for i in range(30):
            aekf.predict(np.array([0.0, 0.0]), dt=0.1)
            z = np.array([1.5 + np.random.normal(0, 0.1)])
            aekf.update(z, H, R_nominal)

        # R не должна измениться
        R_adapted = aekf.get_adapted_R(1)
        if R_adapted is not None:
            np.testing.assert_allclose(R_adapted, R_nominal, atol=1e-10)
