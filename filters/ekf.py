"""
Классический расширенный фильтр Калмана (Extended Kalman Filter, EKF).

Реализует стандартный двухэтапный алгоритм (раздел 2.2):

1. PREDICT (прогноз, раздел 2.2):
    x⁻ = f(x, u)              — нелинейная модель движения
    P⁻ = F * P * F^T + Q      — прогноз ковариации

2. UPDATE (коррекция, раздел 2.2):
    ν = z - H * x⁻             — инновация (невязка)
    S = H * P⁻ * H^T + R       — ковариация инновации
    K = P⁻ * H^T * S^{-1}      — усиление Калмана
    x⁺ = x⁻ + K * ν            — обновлённое состояние
    P⁺ = (I - K * H) * P⁻      — обновлённая ковариация

Вход predict():
    control (np.ndarray): [ω, a] — угловая скорость, ускорение.
    dt (float): Шаг времени.

Вход update():
    measurement (np.ndarray): Вектор измерений z.
    H (np.ndarray): Матрица наблюдения (m × n).
    R (np.ndarray): Ковариационная матрица шума измерений (m × m).

Выход:
    FilterResult: Обновлённые state, covariance, innovation, kalman_gain.

Рекомендации по тестированию:
    1. Предсказание без измерений: ковариация растёт (trace(P) увеличивается).
    2. Обновление с точным измерением (R→0): состояние = измерение.
    3. Обновление с высоким шумом (R→∞): состояние почти не меняется.
    4. Сходимость на прямолинейной траектории с постоянными измерениями.
    5. Проверка симметричности P после каждого шага.
"""

import logging

import numpy as np

from filters.base_filter import BaseFilter, FilterResult
from models.base_motion_model import BaseMotionModel

logger = logging.getLogger(__name__)


class EKF(BaseFilter):
    """Классический расширенный фильтр Калмана.

    Args:
        motion_model: Модель движения НПА (реализация BaseMotionModel).
        initial_state: Начальный вектор состояния [x, y, v, ψ].
        initial_covariance: Начальная ковариационная матрица P0 (n × n).
        process_noise: Матрица шума процесса Q (n × n).
    """

    def __init__(
        self,
        motion_model: BaseMotionModel,
        initial_state: np.ndarray,
        initial_covariance: np.ndarray,
        process_noise: np.ndarray,
    ):
        super().__init__(motion_model, initial_state, initial_covariance, process_noise)
        logger.info(
            "EKF инициализирован. Размерность состояния: %d", len(initial_state)
        )

    def predict(self, control: np.ndarray, dt: float) -> FilterResult:
        """Шаг предсказания EKF.

        Использует нелинейную модель движения для прогноза состояния
        и матрицу Якоби для прогноза ковариации.

        Args:
            control: Вектор управления [ω, a].
            dt: Шаг времени (с).

        Returns:
            FilterResult с предсказанными state и covariance.
        """
        # Нелинейный прогноз состояния (уравнение процесса, раздел 2.2)
        self._state = self._model.predict(self._state, control, dt)

        # Матрица Якоби F (раздел 2.2)
        F = self._model.jacobian(self._state, control, dt)

        # Прогноз ковариации: P⁻ = F * P * F^T + Q (раздел 2.2)
        self._P = F @ self._P @ F.T + self._Q

        # Гарантируем симметричность P
        self._P = 0.5 * (self._P + self._P.T)

        result = FilterResult(
            state=self._state.copy(),
            covariance=self._P.copy(),
        )
        self._history.append(result)

        logger.debug("EKF predict: state=%s, trace(P)=%.4f", self._state, np.trace(self._P))
        return result

    def update(
        self,
        measurement: np.ndarray,
        H: np.ndarray,
        R: np.ndarray,
    ) -> FilterResult:
        """Шаг обновления EKF.

        Корректирует предсказание на основе измерений датчиков.

        Args:
            measurement: Вектор измерений z (размерность m).
            H: Матрица наблюдения (m × n).
            R: Ковариационная матрица шума измерений (m × m).

        Returns:
            FilterResult с обновлёнными state, covariance, innovation, kalman_gain.
        """
        # Инновация (невязка): ν = z - H * x⁻ (раздел 2.2)
        innovation = measurement - H @ self._state

        # Нормализация угла в инновации (если измеряется курс)
        innovation = self._normalize_innovation(innovation, H)

        # Ковариация инновации: S = H * P⁻ * H^T + R (раздел 2.2)
        S = H @ self._P @ H.T + R

        # Усиление Калмана: K = P⁻ * H^T * S^{-1} (раздел 2.2)
        try:
            K = self._P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            logger.warning("Сингулярная матрица S, используем псевдообратную")
            K = self._P @ H.T @ np.linalg.pinv(S)

        # Обновление состояния: x⁺ = x⁻ + K * ν (раздел 2.2)
        self._state = self._state + K @ innovation

        # Нормализация курса
        self._state[3] = BaseMotionModel.normalize_angle(self._state[3])

        # Обновление ковариации: формула Джозефа для численной устойчивости (раздел 2.2)
        n = len(self._state)
        I_KH = np.eye(n) - K @ H
        # P⁺ = (I - K*H) * P⁻ * (I - K*H)^T + K * R * K^T
        self._P = I_KH @ self._P @ I_KH.T + K @ R @ K.T

        # Гарантируем симметричность
        self._P = 0.5 * (self._P + self._P.T)

        result = FilterResult(
            state=self._state.copy(),
            covariance=self._P.copy(),
            innovation=innovation.copy(),
            kalman_gain=K.copy(),
        )
        self._history.append(result)

        logger.debug(
            "EKF update: innovation_norm=%.4f, trace(P)=%.4f",
            np.linalg.norm(innovation),
            np.trace(self._P),
        )
        return result

    def _normalize_innovation(self, innovation: np.ndarray, H: np.ndarray) -> np.ndarray:
        """Нормализация углов в инновации.

        Если строка H соответствует курсу (индекс 3 вектора состояния),
        то нормализуем соответствующий компонент инновации.

        Args:
            innovation: Вектор инновации.
            H: Матрица наблюдения.

        Returns:
            Инновация с нормализованными углами.
        """
        innovation = innovation.copy()
        for i in range(H.shape[0]):
            # Если строка H выбирает курс (индекс 3)
            if H.shape[1] > 3 and H[i, 3] != 0:
                innovation[i] = BaseMotionModel.normalize_angle(innovation[i])
        return innovation
