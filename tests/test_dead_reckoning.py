"""
Тесты Dead Reckoning (наивное счисление пути).

Проверяемые сценарии (раздел 4.5):
    1. predict() корректно интегрирует кинематику.
    2. update() НЕ изменяет состояние (игнорирует измерения).
    3. Ковариация растёт со временем (нет коррекции).
    4. Ошибка DR значительно больше ошибки EKF на длинной траектории.
    5. Совместимость с SimulationRunner (интерфейс BaseFilter).

Простыми словами:
    Эти тесты проверяют, что Dead Reckoning ведёт себя как
    «слепой» навигатор — он интегрирует показания ИНС,
    но полностью игнорирует все внешние датчики.
"""

import numpy as np
import pytest

from filters.dead_reckoning import DeadReckoning
from models.kinematic_2d import Kinematic2DModel


class TestDeadReckoningPredict:
    """Тесты шага предсказания DR."""

    def test_predict_changes_state(self):
        """predict() изменяет состояние при ненулевом управлении."""
        model = Kinematic2DModel()
        state = np.array([0.0, 0.0, 1.5, 0.0])
        P0 = np.diag([1.0, 1.0, 0.1, 0.01])
        Q = np.diag([0.0, 0.0, 0.001, 0.0001])

        dr = DeadReckoning(
            motion_model=model,
            initial_state=state.copy(),
            initial_covariance=P0.copy(),
            process_noise=Q.copy(),
        )

        state_before = dr.get_state().copy()
        dr.predict(np.array([0.0, 0.0]), dt=0.1)
        state_after = dr.get_state()

        # x должен измениться (v=1.5, ψ=0 → x += 0.15)
        assert state_after[0] != pytest.approx(state_before[0])

    def test_predict_increases_uncertainty(self):
        """predict() увеличивает trace(P) — неопределённость растёт."""
        model = Kinematic2DModel()
        state = np.array([0.0, 0.0, 1.5, 0.0])
        P0 = np.diag([1.0, 1.0, 0.1, 0.01])
        Q = np.diag([0.0, 0.0, 0.001, 0.0001])

        dr = DeadReckoning(
            motion_model=model,
            initial_state=state.copy(),
            initial_covariance=P0.copy(),
            process_noise=Q.copy(),
        )

        trace_before = np.trace(dr.get_covariance())
        dr.predict(np.array([0.0, 0.0]), dt=0.1)
        trace_after = np.trace(dr.get_covariance())

        assert trace_after >= trace_before

    def test_predict_returns_filter_result(self):
        """predict() возвращает FilterResult."""
        model = Kinematic2DModel()
        state = np.array([0.0, 0.0, 1.0, 0.0])
        P0 = np.eye(4)
        Q = np.eye(4) * 0.01

        dr = DeadReckoning(model, state, P0, Q)
        result = dr.predict(np.array([0.0, 0.0]), dt=0.1)

        assert result.state is not None
        assert result.covariance is not None
        assert result.state.shape == (4,)
        assert result.covariance.shape == (4, 4)


class TestDeadReckoningUpdate:
    """Тесты шага обновления DR (должен быть пустым)."""

    def test_update_does_not_change_state(self):
        """update() НЕ изменяет состояние — это главное свойство DR.

        Простыми словами: Dead Reckoning «не слышит» датчики.
        Даже если дать ему точное измерение, он его проигнорирует.
        """
        model = Kinematic2DModel()
        state = np.array([0.0, 0.0, 1.5, 0.0])
        P0 = np.eye(4)
        Q = np.eye(4) * 0.01

        dr = DeadReckoning(model, state.copy(), P0, Q)
        dr.predict(np.array([0.0, 0.0]), dt=0.1)
        state_before = dr.get_state().copy()

        # Даём «идеальное» измерение — DR должен его проигнорировать
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        R = np.eye(2) * 0.001
        z = np.array([100.0, 200.0])

        dr.update(z, H, R)
        state_after = dr.get_state()

        np.testing.assert_array_equal(state_after, state_before)

    def test_update_does_not_change_covariance(self):
        """update() НЕ изменяет ковариацию."""
        model = Kinematic2DModel()
        state = np.array([0.0, 0.0, 1.5, 0.0])
        P0 = np.eye(4) * 10.0
        Q = np.eye(4) * 0.01

        dr = DeadReckoning(model, state.copy(), P0, Q)
        dr.predict(np.array([0.0, 0.0]), dt=0.1)
        P_before = dr.get_covariance().copy()

        H = np.array([[0, 0, 1, 0]])
        R = np.array([[0.001]])
        z = np.array([1.5])

        dr.update(z, H, R)
        P_after = dr.get_covariance()

        np.testing.assert_array_equal(P_after, P_before)

    def test_update_returns_filter_result(self):
        """update() возвращает FilterResult с текущим состоянием."""
        model = Kinematic2DModel()
        dr = DeadReckoning(model, np.zeros(4), np.eye(4), np.eye(4) * 0.01)

        H = np.array([[1, 0, 0, 0]])
        R = np.array([[1.0]])
        z = np.array([5.0])

        result = dr.update(z, H, R)
        assert result.state is not None
        assert result.innovation is None  # DR не вычисляет инновацию
        assert result.kalman_gain is None


class TestDeadReckoningDrift:
    """Тесты накопления ошибки DR (раздел 4.5).

    Простыми словами: DR копит ошибку, потому что не корректируется.
    После 100 шагов с зашумлённым управлением позиция DR должна
    значительно отклониться от истинной.
    """

    def test_error_grows_over_time(self):
        """Ошибка DR растёт со временем (дрейф)."""
        model = Kinematic2DModel()
        state = np.array([0.0, 0.0, 1.5, 0.0])
        P0 = np.eye(4) * 0.01
        Q = np.diag([0.0, 0.0, 0.001, 0.0001])

        dr = DeadReckoning(model, state.copy(), P0, Q)
        np.random.seed(42)

        true_state = state.copy()
        errors = []

        for _ in range(200):
            # Истинная эволюция (без шума)
            true_state = model.predict(true_state, np.array([0.0, 0.0]), 0.1)

            # DR с зашумлённым управлением
            noisy_control = np.array([
                np.random.normal(0.0, 0.01),   # шум гироскопа
                np.random.normal(0.0, 0.05),   # шум акселерометра
            ])
            dr.predict(noisy_control, dt=0.1)

            error = np.linalg.norm(dr.get_state()[:2] - true_state[:2])
            errors.append(error)

        # Ошибка в конце должна быть больше, чем в начале
        avg_error_start = np.mean(errors[:20])
        avg_error_end = np.mean(errors[-20:])
        assert avg_error_end > avg_error_start

    def test_covariance_grows_monotonically(self):
        """trace(P) растёт монотонно (нет коррекции)."""
        model = Kinematic2DModel()
        state = np.array([0.0, 0.0, 1.5, 0.0])
        P0 = np.eye(4) * 1.0
        Q = np.diag([0.0, 0.0, 0.001, 0.0001])

        dr = DeadReckoning(model, state.copy(), P0, Q)
        traces = [np.trace(dr.get_covariance())]

        for _ in range(50):
            dr.predict(np.array([0.0, 0.0]), dt=0.1)
            traces.append(np.trace(dr.get_covariance()))

        # Монотонный рост (с учётом числ. погрешностей)
        for i in range(1, len(traces)):
            assert traces[i] >= traces[i - 1] - 1e-10
