"""
Тесты для модуля робастного анализа (evaluation/robustness.py).

Проверяем:
    1. Корректность вычисления метрик на синтетических данных
       с известными ответами.
    2. Обработку граничных случаев (сбой в начале, не восстановился).
    3. Генерацию текста для диплома.
"""

import math

import numpy as np
import pytest

from evaluation.robustness import (
    FilterFailureMetrics,
    FailureRobustnessReport,
    _compute_filter_failure_metrics,
    analyze_robustness,
    print_robustness_report,
    generate_robustness_text,
)
from simulation.runner import SimulationResult, FilterRunResult
from simulation.sensor_failure import FailureScenario


# ─────────────────── Утилиты ───────────────────


def _make_sim_result(
    timestamps: np.ndarray,
    true_xy: np.ndarray,
    filter_estimates: dict[str, np.ndarray],
) -> SimulationResult:
    """Вспомогательная функция: создаёт SimulationResult из xy-данных.

    true_xy и filter_estimates — массивы (N, 2).
    Остальные компоненты (v, psi) заполняются нулями.
    """
    N = len(timestamps)
    true_states = np.zeros((N, 4))
    true_states[:, :2] = true_xy

    result = SimulationResult(
        true_states=true_states,
        timestamps=timestamps,
    )

    for name, est_xy in filter_estimates.items():
        est_states = np.zeros((N, 4))
        est_states[:, :2] = est_xy
        result.filter_results[name] = FilterRunResult(
            filter_name=name,
            estimated_states=est_states,
            covariances=np.zeros((N, 4, 4)),
        )

    return result


# ─────────────────── Тесты _compute_filter_failure_metrics ───────────────────


class TestComputeFilterFailureMetrics:
    """Тесты для вычисления метрик робастности одного фильтра."""

    def test_known_errors(self):
        """Проверка на ручном примере с известными ошибками.

        Сценарий:
            - 200 шагов по 1 с (0..199)
            - Ошибка позиции = 0.5 м до сбоя
            - Ошибка растёт до 3.0 м в окне сбоя [100, 130]
            - Ошибка возвращается к 0.5 м на шаге 140
        """
        N = 200
        timestamps = np.arange(N, dtype=float)

        # Формируем ошибки позиции вручную
        pos_errors = np.full(N, 0.5)

        # В окне [100, 130] ошибка линейно растёт 0.5 → 3.0
        failure_indices = (timestamps >= 100) & (timestamps <= 130)
        n_failure = int(np.sum(failure_indices))
        pos_errors[failure_indices] = np.linspace(0.5, 3.0, n_failure)

        # После сбоя: экспоненциальное затухание обратно к 0.5
        for i in range(131, 200):
            pos_errors[i] = 0.5 + (3.0 - 0.5) * np.exp(-(i - 130) / 3.0)

        failure = FailureScenario(
            sensor_name="DVL", start_time=100.0, end_time=130.0
        )

        metrics = _compute_filter_failure_metrics(
            filter_name="EKF",
            pos_errors=pos_errors,
            timestamps=timestamps,
            failure=failure,
        )

        assert metrics.filter_name == "EKF"
        assert metrics.max_error_during == pytest.approx(3.0, abs=0.1)
        assert metrics.error_at_start == pytest.approx(0.5, abs=0.1)
        assert metrics.error_at_end == pytest.approx(3.0, abs=0.1)
        assert metrics.baseline_error == pytest.approx(0.5, abs=0.01)
        assert metrics.mean_error_during > 0.5
        assert metrics.mean_error_during < 3.0
        # Время восстановления: должно быть конечным
        assert not math.isnan(metrics.recovery_time)
        assert metrics.recovery_time > 0

    def test_no_recovery(self):
        """Фильтр не восстановился до конца симуляции → recovery_time = NaN."""
        N = 200
        timestamps = np.arange(N, dtype=float)
        pos_errors = np.full(N, 0.5)

        # В окне [100, 130] ошибка растёт, и после остаётся высокой
        for i in range(100, 200):
            pos_errors[i] = 5.0

        failure = FailureScenario(
            sensor_name="DVL", start_time=100.0, end_time=130.0
        )

        metrics = _compute_filter_failure_metrics(
            filter_name="EKF",
            pos_errors=pos_errors,
            timestamps=timestamps,
            failure=failure,
        )

        assert math.isnan(metrics.recovery_time)
        assert metrics.max_error_during == pytest.approx(5.0)

    def test_baseline_uses_median(self):
        """Baseline — медиана, а не среднее (устойчивость к выбросам).

        В начале (< 10 с) может быть переходный процесс с высокой ошибкой,
        но baseline берётся только в [10, start_time).
        """
        N = 200
        timestamps = np.arange(N, dtype=float)
        pos_errors = np.full(N, 1.0)

        # Переходный процесс в первые 10 секунд (высокая ошибка)
        pos_errors[:10] = 50.0
        # Один выброс на 50 с
        pos_errors[50] = 100.0

        failure = FailureScenario(
            sensor_name="Compass", start_time=100.0, end_time=110.0
        )

        metrics = _compute_filter_failure_metrics(
            filter_name="EKF",
            pos_errors=pos_errors,
            timestamps=timestamps,
            failure=failure,
        )

        # Медиана [10, 100) = 1.0 (выброс на 50 с не влияет)
        assert metrics.baseline_error == pytest.approx(1.0, abs=0.01)


# ─────────────────── Тесты analyze_robustness ───────────────────


class TestAnalyzeRobustness:
    """Интеграционные тесты для analyze_robustness."""

    def test_two_filters_two_failures(self):
        """Два фильтра, два сбоя — должно быть 2 отчёта, каждый с 2 фильтрами."""
        N = 300
        timestamps = np.arange(N, dtype=float)
        true_xy = np.column_stack([timestamps * 0.1, np.zeros(N)])

        # Фильтр 1 (EKF): ошибка в x = 0.5
        ekf_xy = true_xy.copy()
        ekf_xy[:, 0] += 0.5

        # Фильтр 2 (AEKF): ошибка в x = 0.3
        aekf_xy = true_xy.copy()
        aekf_xy[:, 0] += 0.3

        sim_result = _make_sim_result(
            timestamps, true_xy, {"EKF": ekf_xy, "Adaptive EKF": aekf_xy}
        )

        failures = [
            FailureScenario(sensor_name="DVL", start_time=100, end_time=130),
            FailureScenario(sensor_name="Compass", start_time=150, end_time=160),
        ]

        reports = analyze_robustness(sim_result, failures)

        assert len(reports) == 2
        assert len(reports[0].filter_metrics) == 2
        assert "EKF" in reports[0].filter_metrics
        assert "Adaptive EKF" in reports[0].filter_metrics

        # EKF имеет бо́льшую ошибку
        ekf_max = reports[0].filter_metrics["EKF"].max_error_during
        aekf_max = reports[0].filter_metrics["Adaptive EKF"].max_error_during
        assert ekf_max > aekf_max

    def test_empty_failures(self):
        """Пустой список сбоев → пустой список отчётов."""
        timestamps = np.arange(100, dtype=float)
        true_xy = np.zeros((100, 2))
        sim_result = _make_sim_result(timestamps, true_xy, {"EKF": true_xy})

        reports = analyze_robustness(sim_result, [])
        assert reports == []


# ─────────────────── Тесты вывода ───────────────────


class TestOutputFunctions:
    """Тесты для функций вывода (print и generate_text)."""

    def _make_sample_reports(self) -> list[FailureRobustnessReport]:
        """Вспомогательный метод: создаёт пример отчёта для тестов."""
        failure = FailureScenario(
            sensor_name="DVL", start_time=100, end_time=130
        )
        ekf_m = FilterFailureMetrics(
            filter_name="EKF",
            max_error_during=2.1,
            mean_error_during=1.5,
            error_at_start=0.5,
            error_at_end=2.0,
            baseline_error=0.4,
            recovery_time=8.2,
        )
        aekf_m = FilterFailureMetrics(
            filter_name="Adaptive EKF",
            max_error_during=1.8,
            mean_error_during=1.2,
            error_at_start=0.4,
            error_at_end=1.6,
            baseline_error=0.35,
            recovery_time=3.5,
        )
        return [
            FailureRobustnessReport(
                failure=failure,
                filter_metrics={"EKF": ekf_m, "Adaptive EKF": aekf_m},
            )
        ]

    def test_print_robustness_report_runs(self, capsys):
        """print_robustness_report не падает и печатает ключевые слова."""
        reports = self._make_sample_reports()
        print_robustness_report(reports)
        captured = capsys.readouterr()
        assert "DVL" in captured.out
        assert "EKF" in captured.out
        assert "2.1" in captured.out

    def test_generate_text_contains_values(self):
        """generate_robustness_text содержит числа из отчёта."""
        reports = self._make_sample_reports()
        text = generate_robustness_text(reports)
        assert "2.10" in text or "2.1" in text
        assert "1.80" in text or "1.8" in text
        assert "DVL" in text
        assert "100" in text
        assert "130" in text

    def test_generate_text_nan_recovery(self):
        """Если фильтр не восстановился — текст об этом сообщает."""
        failure = FailureScenario(
            sensor_name="USBL", start_time=50, end_time=80
        )
        m = FilterFailureMetrics(
            filter_name="EKF",
            max_error_during=5.0,
            mean_error_during=3.0,
            error_at_start=1.0,
            error_at_end=5.0,
            baseline_error=0.5,
            recovery_time=float("nan"),
        )
        reports = [
            FailureRobustnessReport(
                failure=failure,
                filter_metrics={"EKF": m},
            )
        ]
        text = generate_robustness_text(reports)
        assert "не восстановился" in text
