"""
Наивное счисление пути (Dead Reckoning, DR).

Реализует наивную навигацию — интегрирование показаний ИНС
без коррекции по внешним датчикам (DVL, Compass, USBL).
Используется как базовый сценарий сравнения (baseline) для
оценки эффективности фильтров Калмана (раздел 4.5).

Принцип работы (раздел 4.5):
    DR выполняет только шаг предсказания (predict), накапливая
    ошибки ИНС без возможности их компенсации. Со временем
    ошибка позиции растёт неограниченно (дрейф).

    Формула эволюции состояния — та же кинематическая модель
    (раздел 2.2), но шаг обновления (update) отсутствует.

Ожидаемое поведение:
    - На коротких интервалах (~10 с) ошибка мала (< 1 м).
    - На длинных интервалах (~300 с) ошибка накапливается
      и может достигать десятков/сотен метров.
    - Это наглядно демонстрирует необходимость коррекции
      по внешним датчикам (EKF, Adaptive EKF).

Зачем нужен Dead Reckoning в дипломе:
    Показать «что будет без фильтра» — ошибка DR >> ошибка EKF.
    Это обосновывает применение расширенного фильтра Калмана
    для задачи автономной навигации НПА.

Рекомендации по тестированию:
    1. update() не изменяет состояние (игнорирует измерения).
    2. predict() корректно интегрирует кинематику.
    3. Ошибка DR растёт со временем (в отличие от EKF).
    4. get_state() и get_covariance() работают корректно.
"""

import logging

import numpy as np

from filters.base_filter import BaseFilter, FilterResult
from models.base_motion_model import BaseMotionModel

logger = logging.getLogger(__name__)


class DeadReckoning(BaseFilter):
    """Наивное счисление пути (Dead Reckoning).

    Выполняет только шаг предсказания по модели движения.
    Шаг обновления (update) игнорируется — измерения от
    внешних датчиков не используются.

    Это даёт нижнюю границу качества навигации и наглядно
    демонстрирует накопление ошибки без коррекции (раздел 4.5).

    Args:
        motion_model: Модель движения НПА (Kinematic2DModel).
        initial_state: Начальный вектор состояния [x, y, v, ψ].
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
        super().__init__(motion_model, initial_state, initial_covariance, process_noise)
        logger.info(
            "Dead Reckoning инициализирован. "
            "Внимание: измерения датчиков будут игнорироваться."
        )

    def predict(self, control: np.ndarray, dt: float) -> FilterResult:
        """Шаг предсказания (единственный рабочий шаг DR).

        Интегрирует кинематическую модель с зашумлённым управлением
        ИНС. Ковариация растёт неограниченно, т.к. коррекции нет.

        Формулы — те же, что и в EKF predict (раздел 2.2):
            x⁻ = f(x, u)
            P⁻ = F * P * F^T + Q

        Args:
            control: Вектор управления [ω, a] от ИНС (с шумом).
            dt: Шаг времени Δt (с).

        Returns:
            FilterResult с предсказанными state и covariance.
        """
        # Нелинейный прогноз состояния (раздел 2.2)
        self._state = self._model.predict(self._state, control, dt)

        # Матрица Якоби F (раздел 2.2)
        F = self._model.jacobian(self._state, control, dt)

        # Прогноз ковариации: P⁻ = F * P * F^T + Q
        self._P = F @ self._P @ F.T + self._Q
        self._P = 0.5 * (self._P + self._P.T)

        result = FilterResult(
            state=self._state.copy(),
            covariance=self._P.copy(),
        )
        self._history.append(result)
        return result

    def update(
        self,
        measurement: np.ndarray,
        H: np.ndarray,
        R: np.ndarray,
    ) -> FilterResult:
        """Шаг обновления — НАМЕРЕННО ПУСТОЙ (раздел 4.5).

        Dead Reckoning не использует измерения от внешних датчиков.
        Этот метод существует для совместимости с интерфейсом BaseFilter
        и SimulationRunner, но ничего не делает.

        Простыми словами: DR «не слышит» внешние датчики —
        он полагается только на собственные показания ИНС,
        которые дрейфуют со временем.

        Args:
            measurement: Вектор измерений z (игнорируется).
            H: Матрица наблюдения (игнорируется).
            R: Ковариация шума измерений (игнорируется).

        Returns:
            FilterResult с текущими (неизменёнными) state и covariance.
        """
        # Ничего не делаем — возвращаем текущее состояние без изменений
        result = FilterResult(
            state=self._state.copy(),
            covariance=self._P.copy(),
            innovation=None,
            kalman_gain=None,
        )
        return result
