"""
Абстрактный интерфейс навигационного фильтра.

Определяет контракт для всех фильтров (EKF, Adaptive EKF, UKF и т.д.).
Каждый фильтр хранит текущую оценку состояния и ковариационную матрицу,
а также предоставляет методы predict() и update().

Вход:
    predict():
        control (np.ndarray): Вектор управления.
        dt (float): Шаг времени.
    update():
        measurement (np.ndarray): Вектор измерений.
        H (np.ndarray): Матрица наблюдения.
        R (np.ndarray): Ковариационная матрица шума измерений.

Выход:
    state (np.ndarray): Текущая оценка состояния.
    covariance (np.ndarray): Текущая ковариационная матрица.

Рекомендации по тестированию:
    - Проверить, что predict() корректно обновляет state и covariance.
    - Проверить, что update() уменьшает неопределённость (trace(P) уменьшается).
    - Проверить, что фильтр сходится к истинному значению при идеальных измерениях.
    - Проверить, что get_state() возвращает копию (не ссылку).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from models.base_motion_model import BaseMotionModel


@dataclass
class FilterResult:
    """Результат одного шага фильтрации.

    Attributes:
        state: Оценка вектора состояния после шага.
        covariance: Ковариационная матрица после шага.
        innovation: Инновация (невязка) при обновлении (None для predict).
        kalman_gain: Матрица усиления Калмана (None для predict).
    """
    state: np.ndarray
    covariance: np.ndarray
    innovation: Optional[np.ndarray] = None
    kalman_gain: Optional[np.ndarray] = None


class BaseFilter(ABC):
    """Абстрактный навигационный фильтр.

    Хранит модель движения, текущую оценку состояния и ковариацию.
    Подклассы реализуют конкретные алгоритмы predict() и update().

    Args:
        motion_model: Модель движения НПА.
        initial_state: Начальный вектор состояния.
        initial_covariance: Начальная ковариационная матрица P0.
        process_noise: Матрица шума процесса Q.
    """

    def __init__(
        self,
        motion_model: BaseMotionModel,
        initial_state: np.ndarray,
        initial_covariance: np.ndarray,
        process_noise: np.ndarray,
    ):
        self._model = motion_model
        self._state = initial_state.copy().astype(float)
        self._P = initial_covariance.copy().astype(float)
        self._Q = process_noise.copy().astype(float)

        # История для анализа
        self._history: list[FilterResult] = []

    @abstractmethod
    def predict(self, control: np.ndarray, dt: float) -> FilterResult:
        """Шаг предсказания (прогноз).

        Args:
            control: Вектор управления [ω, a].
            dt: Шаг времени (с).

        Returns:
            Результат шага предсказания.
        """
        ...

    @abstractmethod
    def update(
        self,
        measurement: np.ndarray,
        H: np.ndarray,
        R: np.ndarray,
    ) -> FilterResult:
        """Шаг обновления (коррекция по измерениям).

        Args:
            measurement: Вектор измерений z.
            H: Матрица наблюдения (связь состояния и измерений).
            R: Ковариационная матрица шума измерений.

        Returns:
            Результат шага обновления.
        """
        ...

    def get_state(self) -> np.ndarray:
        """Получить текущую оценку состояния (копию).

        Returns:
            Копия вектора состояния.
        """
        return self._state.copy()

    def get_covariance(self) -> np.ndarray:
        """Получить текущую ковариационную матрицу (копию).

        Returns:
            Копия ковариационной матрицы P.
        """
        return self._P.copy()

    def get_history(self) -> list[FilterResult]:
        """Получить историю результатов фильтрации.

        Returns:
            Список FilterResult для каждого шага.
        """
        return self._history.copy()

    def reset(self, state: np.ndarray, covariance: np.ndarray) -> None:
        """Сброс фильтра к новому начальному состоянию.

        Args:
            state: Новый начальный вектор состояния.
            covariance: Новая начальная ковариационная матрица.
        """
        self._state = state.copy().astype(float)
        self._P = covariance.copy().astype(float)
        self._history.clear()
