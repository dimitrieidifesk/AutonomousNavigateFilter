"""
Абстрактный интерфейс модели движения.

Определяет контракт, которому должна соответствовать любая модель движения НПА.
Новые модели (3D, 6-DOF и т.д.) добавляются реализацией этого интерфейса
без изменения существующего кода (принцип Open/Closed).

Вход:
    state (np.ndarray): Текущий вектор состояния.
    control (np.ndarray): Вектор управления (ω, a).
    dt (float): Шаг интеграции по времени.

Выход:
    predict() -> np.ndarray: Предсказанный вектор состояния.
    jacobian() -> np.ndarray: Матрица Якоби (∂f/∂x) для EKF.

Рекомендации по тестированию:
    - Проверить predict() на нулевом управлении (состояние не меняется).
    - Проверить predict() на прямолинейном движении (ψ=0, ω=0).
    - Проверить jacobian() численно (конечные разности vs аналитическая).
    - Проверить корректность обработки углов (нормализация ψ в [-π, π]).
"""

from abc import ABC, abstractmethod
import numpy as np


class BaseMotionModel(ABC):
    """Абстрактная модель движения НПА.

    Все модели движения наследуют этот класс и реализуют методы
    predict() и jacobian(). Это обеспечивает взаимозаменяемость
    моделей в фильтрах (принцип Liskov Substitution).
    """

    @abstractmethod
    def predict(self, state: np.ndarray, control: np.ndarray, dt: float) -> np.ndarray:
        """Предсказание следующего состояния.

        Args:
            state: Текущий вектор состояния [x, y, v, ψ].
            control: Вектор управления [ω (рад/с), a (м/с²)].
            dt: Шаг времени (с).

        Returns:
            Предсказанный вектор состояния.
        """
        ...

    @abstractmethod
    def jacobian(self, state: np.ndarray, control: np.ndarray, dt: float) -> np.ndarray:
        """Матрица Якоби модели движения (∂f/∂x).

        Используется в EKF для линеаризации нелинейной модели.

        Args:
            state: Текущий вектор состояния [x, y, v, ψ].
            control: Вектор управления [ω, a].
            dt: Шаг времени (с).

        Returns:
            Матрица Якоби размером (state_dim x state_dim).
        """
        ...

    @staticmethod
    def normalize_angle(angle: float) -> float:
        """Нормализация угла в диапазон [-π, π].

        Args:
            angle: Угол в радианах.

        Returns:
            Нормализованный угол.
        """
        return (angle + np.pi) % (2 * np.pi) - np.pi
