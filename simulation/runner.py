"""
Оркестратор симуляции.

Объединяет все компоненты: модель движения, фильтры, датчики —
и выполняет полный прогон симуляции. Это главный «дирижёр» системы,
реализующий принцип Dependency Inversion: он зависит от абстракций
(BaseFilter, BaseMotionModel), а не от конкретных реализаций.

Вход:
    SimulationData: Данные симуляции (зашумлённые управления и измерения).
    Список фильтров для сравнения.

Выход:
    SimulationResult: Оценки состояний каждого фильтра, метрики, истинная траектория.

Рекомендации по тестированию:
    - Проверить, что результат содержит данные для каждого фильтра.
    - Проверить, что длина estimated_states = длине траектории.
    - Проверить, что run() не модифицирует входные данные.
"""

from dataclasses import dataclass, field
import logging

import numpy as np

from filters.base_filter import BaseFilter
from sensors.base_sensor import SensorMeasurement
from simulation.data_generator import SimulationData

logger = logging.getLogger(__name__)


@dataclass
class FilterRunResult:
    """Результат прогона одного фильтра.

    Attributes:
        filter_name: Имя фильтра.
        estimated_states: Оценки состояний (N × 4).
        covariances: Ковариационные матрицы (N × 4 × 4).
    """
    filter_name: str
    estimated_states: np.ndarray   # (N, 4)
    covariances: np.ndarray        # (N, 4, 4)


@dataclass
class SimulationResult:
    """Полный результат симуляции.

    Attributes:
        true_states: Истинные состояния (N × 4).
        timestamps: Временные метки (N,).
        filter_results: Результаты для каждого фильтра.
    """
    true_states: np.ndarray
    timestamps: np.ndarray
    filter_results: dict[str, FilterRunResult] = field(default_factory=dict)


class SimulationRunner:
    """Оркестратор прогона симуляции.

    Прогоняет симуляцию для каждого переданного фильтра,
    выполняя predict() на каждом шаге и update() при наличии
    измерений от соответствующих датчиков.

    Args:
        filters: Словарь {имя_фильтра: экземпляр фильтра}.
    """

    def __init__(self, filters: dict[str, BaseFilter]):
        self._filters = filters

    def run(self, data: SimulationData) -> SimulationResult:
        """Выполнить полный прогон симуляции.

        Args:
            data: Данные симуляции (траектория + зашумлённые данные).

        Returns:
            SimulationResult со всеми оценками.
        """
        trajectory = data.trajectory
        N = len(trajectory.timestamps)
        dt = trajectory.timestamps[1] - trajectory.timestamps[0] if N > 1 else 0.02

        result = SimulationResult(
            true_states=trajectory.states.copy(),
            timestamps=trajectory.timestamps.copy(),
        )

        # Прогон каждого фильтра независимо
        for filter_name, nav_filter in self._filters.items():
            logger.info("Запуск фильтра: %s", filter_name)
            filter_run = self._run_single_filter(
                nav_filter, data, N, dt, filter_name
            )
            result.filter_results[filter_name] = filter_run
            logger.info(
                "Фильтр %s завершён. Финальное состояние: %s",
                filter_name,
                filter_run.estimated_states[-1],
            )

        return result

    def _run_single_filter(
        self,
        nav_filter: BaseFilter,
        data: SimulationData,
        N: int,
        dt: float,
        filter_name: str,
    ) -> FilterRunResult:
        """Прогон одного фильтра по всем данным.

        На каждом шаге:
        1. Выполняет predict() с зашумлённым управлением ИНС.
        2. Проверяет, есть ли измерения от датчиков на этом шаге.
        3. Если есть — выполняет update() для каждого измерения.

        Args:
            nav_filter: Экземпляр фильтра.
            data: Данные симуляции.
            N: Количество шагов.
            dt: Шаг времени (с).
            filter_name: Имя фильтра (для логирования).

        Returns:
            FilterRunResult с оценками состояний и ковариациями.
        """
        state_dim = 4
        estimated_states = np.zeros((N, state_dim))
        covariances = np.zeros((N, state_dim, state_dim))

        # Индексы текущего измерения для каждого датчика
        sensor_indices: dict[str, int] = {
            name: 0 for name in data.sensor_measurements
        }

        for i in range(N):
            t = data.trajectory.timestamps[i]

            # --- Шаг предсказания ---
            control = data.ins_controls[i]
            nav_filter.predict(control, dt)

            # --- Шаг обновления (для каждого доступного датчика) ---
            for sensor_name, measurements in data.sensor_measurements.items():
                idx = sensor_indices[sensor_name]
                if idx < len(measurements):
                    meas = measurements[idx]
                    # Проверяем, что время измерения ≈ текущему шагу
                    if abs(meas.timestamp - t) < dt * 0.5:
                        nav_filter.update(meas.value, meas.H, meas.R)
                        sensor_indices[sensor_name] = idx + 1

            # Сохраняем оценку
            estimated_states[i] = nav_filter.get_state()
            covariances[i] = nav_filter.get_covariance()

        return FilterRunResult(
            filter_name=filter_name,
            estimated_states=estimated_states,
            covariances=covariances,
        )
