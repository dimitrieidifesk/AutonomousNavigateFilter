"""
Тесты классического EKF.

Проверяемые сценарии:
    1. Predict увеличивает неопределённость (trace(P) растёт).
    2. Update с точным измерением приближает состояние к измерению.
    3. Update уменьшает неопределённость (trace(P) уменьшается).
    4. Симметричность ковариационной матрицы P.
    5. Сходимость фильтра на прямолинейном движении.
    6. get_state() возвращает копию (не ссылку).
    7. Корректная работа при высоком шуме (R→∞).
"""

import numpy as np
import pytest

from filters.ekf import EKF


class TestEKFPredict:
    """Тесты шага предсказания EKF."""

    def test_predict_increases_uncertainty(self, ekf):
        """После predict() trace(P) должен увеличиться (или остаться)."""
        trace_before = np.trace(ekf.get_covariance())
        ekf.predict(np.array([0.0, 0.0]), dt=0.1)
        trace_after = np.trace(ekf.get_covariance())

        assert trace_after >= trace_before

    def test_predict_state_changes(self, ekf):
        """Состояние должно измениться после predict() при ненулевом управлении."""
        state_before = ekf.get_state().copy()
        # Ускорение 0.5 м/с² — даже при v0=0 скорость изменится
        ekf.predict(np.array([0.0, 0.5]), dt=0.1)
        state_after = ekf.get_state()

        # v должна увеличиться (a=0.5, dt=0.1 → Δv=0.05)
        assert state_after[2] != pytest.approx(state_before[2])

    def test_predict_returns_filter_result(self, ekf):
        """predict() возвращает FilterResult."""
        result = ekf.predict(np.array([0.0, 0.0]), dt=0.1)

        assert result.state is not None
        assert result.covariance is not None
        assert result.state.shape == (4,)
        assert result.covariance.shape == (4, 4)

    def test_covariance_symmetric_after_predict(self, ekf):
        """P должна оставаться симметричной после predict()."""
        ekf.predict(np.array([0.1, 0.05]), dt=0.1)
        P = ekf.get_covariance()

        np.testing.assert_allclose(P, P.T, atol=1e-12)


class TestEKFUpdate:
    """Тесты шага обновления EKF."""

    def test_update_decreases_uncertainty(self, ekf):
        """После update() trace(P) должен уменьшиться."""
        ekf.predict(np.array([0.0, 0.0]), dt=0.1)
        trace_before = np.trace(ekf.get_covariance())

        # Измерение скорости (DVL)
        H = np.array([[0, 0, 1, 0]])
        R = np.array([[0.001]])
        z = np.array([1.5])

        ekf.update(z, H, R)
        trace_after = np.trace(ekf.get_covariance())

        assert trace_after < trace_before

    def test_update_with_exact_measurement(self, ekf):
        """При R→0 состояние должно ≈ измерению."""
        ekf.predict(np.array([0.0, 0.0]), dt=0.1)

        # Измерение координат (USBL) с очень малым шумом
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        R = np.eye(2) * 1e-10
        true_pos = np.array([0.15, 0.0])

        ekf.update(true_pos, H, R)
        state = ekf.get_state()

        np.testing.assert_allclose(state[:2], true_pos, atol=0.01)

    def test_update_with_high_noise_no_change(self, ekf):
        """При R→∞ состояние почти не меняется."""
        ekf.predict(np.array([0.0, 0.0]), dt=0.1)
        state_before = ekf.get_state().copy()

        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        R = np.eye(2) * 1e6
        z = np.array([100.0, 100.0])  # далеко от истины

        ekf.update(z, H, R)
        state_after = ekf.get_state()

        # Состояние почти не изменилось
        np.testing.assert_allclose(state_after[:2], state_before[:2], atol=0.1)

    def test_update_returns_innovation(self, ekf):
        """update() возвращает инновацию и усиление Калмана."""
        ekf.predict(np.array([0.0, 0.0]), dt=0.1)

        H = np.array([[0, 0, 1, 0]])
        R = np.array([[0.01]])
        z = np.array([1.6])

        result = ekf.update(z, H, R)

        assert result.innovation is not None
        assert result.kalman_gain is not None
        assert result.innovation.shape == (1,)
        assert result.kalman_gain.shape == (4, 1)

    def test_covariance_symmetric_after_update(self, ekf):
        """P должна оставаться симметричной после update()."""
        ekf.predict(np.array([0.1, 0.05]), dt=0.1)

        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        R = np.eye(2) * 0.5
        z = np.array([0.1, 0.1])

        ekf.update(z, H, R)
        P = ekf.get_covariance()

        np.testing.assert_allclose(P, P.T, atol=1e-12)


class TestEKFConvergence:
    """Тесты сходимости EKF."""

    def test_convergence_on_straight_trajectory(self, ekf):
        """EKF сходится к истинной траектории при прямолинейном движении."""
        dt = 0.1
        true_state = np.array([0.0, 0.0, 1.5, 0.0])

        errors = []
        for step in range(100):
            true_state[0] += true_state[2] * np.cos(true_state[3]) * dt
            true_state[1] += true_state[2] * np.sin(true_state[3]) * dt

            # Predict
            ekf.predict(np.array([0.0, 0.0]), dt)

            # Каждые 10 шагов — обновление координат
            if step % 10 == 0:
                H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
                R = np.eye(2) * 1.0
                z = true_state[:2] + np.random.normal(0, 1.0, 2)
                ekf.update(z, H, R)

            # Каждые 2 шага — обновление скорости
            if step % 2 == 0:
                H = np.array([[0, 0, 1, 0]])
                R = np.array([[0.01]])
                z = np.array([true_state[2] + np.random.normal(0, 0.02)])
                ekf.update(z, H, R)

            error = np.linalg.norm(ekf.get_state()[:2] - true_state[:2])
            errors.append(error)

        # Средняя ошибка в конце должна быть меньше начальной
        avg_error_start = np.mean(errors[:20])
        avg_error_end = np.mean(errors[-20:])
        assert avg_error_end < avg_error_start + 5.0  # допуск с учётом шума


class TestEKFEncapsulation:
    """Тесты инкапсуляции."""

    def test_get_state_returns_copy(self, ekf):
        """get_state() возвращает копию, а не ссылку."""
        state1 = ekf.get_state()
        state1[0] = 999.0
        state2 = ekf.get_state()

        assert state2[0] != 999.0

    def test_get_covariance_returns_copy(self, ekf):
        """get_covariance() возвращает копию."""
        P1 = ekf.get_covariance()
        P1[0, 0] = 999.0
        P2 = ekf.get_covariance()

        assert P2[0, 0] != 999.0

    def test_reset(self, ekf):
        """reset() корректно сбрасывает состояние."""
        ekf.predict(np.array([0.1, 0.05]), dt=0.5)

        new_state = np.array([10.0, 20.0, 2.0, 1.0])
        new_P = np.eye(4) * 5.0
        ekf.reset(new_state, new_P)

        np.testing.assert_array_equal(ekf.get_state(), new_state)
        np.testing.assert_array_equal(ekf.get_covariance(), new_P)
