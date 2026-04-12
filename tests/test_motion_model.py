"""
Тесты модели движения (Kinematic2DModel).

Проверяемые сценарии:
    1. Прямолинейное движение: при ψ=0, ω=0 — x растёт, y не меняется.
    2. Нулевое управление: координаты меняются равномерно.
    3. Нормализация курса: углы остаются в [-π, π].
    4. Численная проверка Якобиана: конечные разности vs аналитика.
    5. Размерности выходов predict() и jacobian().
"""

import numpy as np
import pytest

from models.kinematic_2d import Kinematic2DModel
from models.base_motion_model import BaseMotionModel


class TestKinematic2DPredict:
    """Тесты шага предсказания."""

    def test_straight_motion_x_direction(self, motion_model):
        """При ψ=0, ω=0 — аппарат движется строго по X."""
        state = np.array([0.0, 0.0, 1.5, 0.0])  # v=1.5, ψ=0
        control = np.array([0.0, 0.0])            # ω=0, a=0
        dt = 0.1

        new_state = motion_model.predict(state, control, dt)

        assert new_state[0] == pytest.approx(0.15, abs=1e-10)  # x = 0 + 1.5*cos(0)*0.1
        assert new_state[1] == pytest.approx(0.0, abs=1e-10)   # y не меняется
        assert new_state[2] == pytest.approx(1.5, abs=1e-10)   # v не меняется
        assert new_state[3] == pytest.approx(0.0, abs=1e-10)   # ψ не меняется

    def test_straight_motion_y_direction(self, motion_model):
        """При ψ=π/2 — аппарат движется строго по Y."""
        state = np.array([0.0, 0.0, 1.5, np.pi / 2])
        control = np.array([0.0, 0.0])
        dt = 0.1

        new_state = motion_model.predict(state, control, dt)

        assert new_state[0] == pytest.approx(0.0, abs=1e-10)   # x не меняется
        assert new_state[1] == pytest.approx(0.15, abs=1e-10)  # y растёт
        assert new_state[2] == pytest.approx(1.5, abs=1e-10)

    def test_acceleration(self, motion_model):
        """Проверка ускорения: v должно измениться."""
        state = np.array([0.0, 0.0, 1.0, 0.0])
        control = np.array([0.0, 0.5])  # ускорение 0.5 м/с²
        dt = 1.0

        new_state = motion_model.predict(state, control, dt)

        assert new_state[2] == pytest.approx(1.5, abs=1e-10)  # v = 1.0 + 0.5*1.0

    def test_rotation(self, motion_model):
        """Проверка вращения: ψ должно измениться."""
        state = np.array([0.0, 0.0, 1.0, 0.0])
        control = np.array([0.1, 0.0])  # ω = 0.1 рад/с
        dt = 1.0

        new_state = motion_model.predict(state, control, dt)

        assert new_state[3] == pytest.approx(0.1, abs=1e-10)

    def test_angle_normalization(self, motion_model):
        """Курс должен нормализоваться в [-π, π]."""
        state = np.array([0.0, 0.0, 1.0, 3.0])
        control = np.array([0.5, 0.0])
        dt = 1.0

        new_state = motion_model.predict(state, control, dt)

        assert -np.pi <= new_state[3] <= np.pi

    def test_output_dimensions(self, motion_model):
        """predict() возвращает вектор размерности 4."""
        state = np.array([1.0, 2.0, 1.5, 0.5])
        control = np.array([0.1, 0.0])
        dt = 0.02

        result = motion_model.predict(state, control, dt)

        assert result.shape == (4,)


class TestKinematic2DJacobian:
    """Тесты матрицы Якоби."""

    def test_jacobian_dimensions(self, motion_model):
        """Якобиан должен быть матрицей 4×4."""
        state = np.array([0.0, 0.0, 1.5, 0.0])
        control = np.array([0.0, 0.0])
        dt = 0.02

        F = motion_model.jacobian(state, control, dt)

        assert F.shape == (4, 4)

    def test_jacobian_identity_diagonal(self, motion_model):
        """Диагональ Якобиана содержит единицы."""
        state = np.array([0.0, 0.0, 1.5, 0.0])
        control = np.array([0.0, 0.0])
        dt = 0.02

        F = motion_model.jacobian(state, control, dt)

        np.testing.assert_array_equal(np.diag(F), [1, 1, 1, 1])

    def test_jacobian_vs_numerical(self, motion_model):
        """Аналитический Якобиан ≈ численному (конечные разности)."""
        state = np.array([10.0, 5.0, 1.5, 0.3])
        control = np.array([0.1, 0.05])
        dt = 0.1

        F_analytical = motion_model.jacobian(state, control, dt)

        # Численный Якобиан
        eps = 1e-7
        n = len(state)
        F_numerical = np.zeros((n, n))

        for j in range(n):
            state_plus = state.copy()
            state_minus = state.copy()
            state_plus[j] += eps
            state_minus[j] -= eps

            f_plus = motion_model.predict(state_plus, control, dt)
            f_minus = motion_model.predict(state_minus, control, dt)

            F_numerical[:, j] = (f_plus - f_minus) / (2 * eps)

        np.testing.assert_allclose(F_analytical, F_numerical, atol=1e-5)


class TestNormalizeAngle:
    """Тесты нормализации углов."""

    @pytest.mark.parametrize("angle,expected", [
        (0.0, 0.0),
        (np.pi, -np.pi),  # граничный случай
        (2 * np.pi, 0.0),
        (-2 * np.pi, 0.0),
        (3 * np.pi, -np.pi),
        (np.pi / 2, np.pi / 2),
    ])
    def test_normalize_angle(self, angle, expected):
        """Нормализация углов в [-π, π]."""
        result = BaseMotionModel.normalize_angle(angle)
        assert result == pytest.approx(expected, abs=1e-10)
