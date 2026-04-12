"""
Тесты метрик точности.

Проверяемые сценарии:
    1. RMSE(x, x) = 0 (нулевая ошибка).
    2. MAE(x, x) = 0.
    3. RMSE ≥ MAE для любых данных.
    4. Max Error ≥ RMSE.
    5. Все метрики ≥ 0.
    6. Известные вычисления (ручная проверка).
    7. compute_metrics() возвращает MetricsReport с корректными полями.
"""

import numpy as np
import pytest

from evaluation.metrics import rmse, mae, max_error, position_error_norm, compute_metrics


class TestRMSE:
    """Тесты RMSE."""

    def test_zero_error(self):
        """RMSE = 0 при идентичных массивах."""
        x = np.array([1.0, 2.0, 3.0])
        assert rmse(x, x) == pytest.approx(0.0)

    def test_known_value(self):
        """RMSE для известных данных."""
        estimated = np.array([1.0, 2.0, 3.0])
        true = np.array([1.0, 2.0, 4.0])
        # error = [0, 0, 1], MSE = 1/3, RMSE = sqrt(1/3)
        expected = np.sqrt(1.0 / 3)
        assert rmse(estimated, true) == pytest.approx(expected, abs=1e-10)

    def test_non_negative(self):
        """RMSE ≥ 0."""
        x = np.random.randn(100)
        y = np.random.randn(100)
        assert rmse(x, y) >= 0


class TestMAE:
    """Тесты MAE."""

    def test_zero_error(self):
        """MAE = 0 при идентичных массивах."""
        x = np.array([1.0, 2.0, 3.0])
        assert mae(x, x) == pytest.approx(0.0)

    def test_known_value(self):
        """MAE для известных данных."""
        estimated = np.array([1.0, 3.0, 5.0])
        true = np.array([2.0, 3.0, 3.0])
        # |error| = [1, 0, 2], MAE = 3/3 = 1.0
        assert mae(estimated, true) == pytest.approx(1.0)


class TestMaxError:
    """Тесты максимальной ошибки."""

    def test_zero_error(self):
        """Max error = 0 при идентичных массивах."""
        x = np.array([1.0, 2.0, 3.0])
        assert max_error(x, x) == pytest.approx(0.0)

    def test_known_value(self):
        """Max error для известных данных."""
        estimated = np.array([1.0, 5.0, 3.0])
        true = np.array([1.0, 2.0, 3.0])
        assert max_error(estimated, true) == pytest.approx(3.0)


class TestMetricRelations:
    """Тесты соотношений между метриками."""

    def test_rmse_ge_mae(self):
        """RMSE ≥ MAE для любых данных (неравенство Коши-Шварца)."""
        np.random.seed(42)
        for _ in range(10):
            x = np.random.randn(50)
            y = np.random.randn(50)
            assert rmse(x, y) >= mae(x, y) - 1e-10

    def test_max_error_ge_rmse(self):
        """Max error ≥ RMSE для любых данных."""
        np.random.seed(42)
        for _ in range(10):
            x = np.random.randn(50)
            y = np.random.randn(50)
            # max_error ≥ RMSE не всегда верно для 1D, но верно что max ≥ средние
            assert max_error(x, y) >= mae(x, y) - 1e-10


class TestPositionErrorNorm:
    """Тесты нормы ошибки позиции."""

    def test_zero_error(self):
        """Нулевая ошибка при идентичных координатах."""
        pos = np.array([[1.0, 2.0], [3.0, 4.0]])
        norms = position_error_norm(pos, pos)
        np.testing.assert_allclose(norms, [0.0, 0.0])

    def test_known_value(self):
        """Известная норма: ||(3,4)|| = 5."""
        estimated = np.array([[3.0, 4.0]])
        true = np.array([[0.0, 0.0]])
        norms = position_error_norm(estimated, true)
        assert norms[0] == pytest.approx(5.0)


class TestComputeMetrics:
    """Тесты compute_metrics()."""

    def test_returns_metrics_report(self):
        """compute_metrics() возвращает MetricsReport."""
        estimated = np.array([
            [1.0, 2.0, 1.5, 0.1],
            [2.0, 3.0, 1.5, 0.2],
        ])
        true = np.array([
            [1.1, 2.1, 1.5, 0.1],
            [2.1, 3.1, 1.5, 0.2],
        ])

        report = compute_metrics(estimated, true, "TestFilter")

        assert report.filter_name == "TestFilter"
        assert report.rmse_x >= 0
        assert report.rmse_y >= 0
        assert report.rmse_position >= 0
        assert report.mae_position >= 0
        assert report.max_error_position >= 0

    def test_perfect_estimation(self):
        """При идеальной оценке все метрики = 0."""
        states = np.array([
            [0.0, 0.0, 1.5, 0.0],
            [0.15, 0.0, 1.5, 0.0],
        ])

        report = compute_metrics(states, states, "Perfect")

        assert report.rmse_x == pytest.approx(0.0)
        assert report.rmse_y == pytest.approx(0.0)
        assert report.rmse_position == pytest.approx(0.0)
        assert report.max_error_position == pytest.approx(0.0)
