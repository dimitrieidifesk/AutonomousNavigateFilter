"""
Абстрактный интерфейс датчика.

Определяет контракт для всех сенсоров: каждый датчик предоставляет
матрицу наблюдения H, ковариацию шума R и может генерировать
зашумлённые измерения из истинного состояния.

Вход:
    true_state (np.ndarray): Истинный вектор состояния НПА [x, y, v, ψ].

Выход:
    measure() -> np.ndarray: Зашумлённое измерение.
    H -> np.ndarray: Матрица наблюдения (связь z = H * x + шум).
    R -> np.ndarray: Ковариационная матрица шума.
    rate -> float: Частота обновления датчика (Гц).

Рекомендации по тестированию:
    - Проверить, что measure() возвращает вектор правильной размерности.
    - Проверить, что среднее множества измерений ≈ истинному значению.
    - Проверить, что дисперсия измерений ≈ заявленному шуму.
    - Проверить, что is_available(t) корректно работает по частоте.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class SensorMeasurement:
    """Результат измерения датчика.

    Attributes:
        value: Вектор измерений.
        H: Матрица наблюдения для этого измерения.
        R: Ковариационная матрица шума для этого измерения.
        sensor_name: Имя датчика (для логирования и визуализации).
        timestamp: Метка времени измерения (с).
    """
    value: np.ndarray
    H: np.ndarray
    R: np.ndarray
    sensor_name: str
    timestamp: float


class BaseSensor(ABC):
    """Абстрактный датчик НПА.

    Каждый датчик характеризуется:
    - Матрицей наблюдения H (какие компоненты состояния он измеряет).
    - Ковариацией шума R (точность датчика).
    - Частотой обновления rate (Гц).

    Args:
        noise_std: СКО шума измерений (скаляр или вектор).
        rate: Частота обновления (Гц).
        state_dim: Размерность полного вектора состояния.
        name: Имя датчика.
    """

    def __init__(
        self,
        noise_std: np.ndarray | float,
        rate: float,
        state_dim: int = 4,
        name: str = "sensor",
    ):
        self._noise_std = np.atleast_1d(np.asarray(noise_std, dtype=float))
        self._rate = rate
        self._state_dim = state_dim
        self._name = name

        # Период обновления (с)
        self._period = 1.0 / rate

        # Ковариационная матрица шума R = diag(σ²)
        self._R = np.diag(self._noise_std ** 2)

    @abstractmethod
    def get_H(self) -> np.ndarray:
        """Матрица наблюдения H (m × state_dim).

        Определяет, какие компоненты вектора состояния измеряет датчик.

        Returns:
            Матрица H размером (measurement_dim × state_dim).
        """
        ...

    def measure(self, true_state: np.ndarray, timestamp: float) -> SensorMeasurement:
        """Генерация зашумлённого измерения.

        Args:
            true_state: Истинный вектор состояния [x, y, v, ψ].
            timestamp: Текущее время (с).

        Returns:
            SensorMeasurement с зашумлённым значением.
        """
        H = self.get_H()
        true_value = H @ true_state
        noise = np.random.normal(0, self._noise_std)
        measurement = true_value + noise

        return SensorMeasurement(
            value=measurement,
            H=H,
            R=self._R.copy(),
            sensor_name=self._name,
            timestamp=timestamp,
        )

    def is_available(self, timestamp: float, dt: float) -> bool:
        """Проверяет, доступно ли измерение в данный момент времени.

        Моделирует дискретную частоту обновления датчика.
        Датчик срабатывает, если хотя бы один тик (кратный периоду)
        попадает в полуоткрытый интервал (timestamp - dt, timestamp].

        Args:
            timestamp: Текущее время симуляции (с).
            dt: Шаг интеграции (с).

        Returns:
            True, если датчик должен выдать измерение на этом шаге.
        """
        if self._period <= dt:
            # Частота датчика >= частоты симуляции — доступен каждый шаг
            return True
        # Ближайший тик, не превышающий timestamp
        eps = 1e-9
        k = int((timestamp + eps) / self._period)
        tick = k * self._period
        # Тик попадает в (timestamp - dt, timestamp] ?
        return tick > (timestamp - dt + eps) and tick <= (timestamp + eps)

    @property
    def rate(self) -> float:
        """Частота обновления датчика (Гц)."""
        return self._rate

    @property
    def name(self) -> str:
        """Имя датчика."""
        return self._name

    @property
    def measurement_dim(self) -> int:
        """Размерность измерения."""
        return len(self._noise_std)

    @property
    def R(self) -> np.ndarray:
        """Ковариационная матрица шума (копия)."""
        return self._R.copy()
