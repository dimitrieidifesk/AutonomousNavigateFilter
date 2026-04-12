"""
Адаптивный расширенный фильтр Калмана (Adaptive EKF).

Расширяет классический EKF механизмом адаптации матрицы шума измерений R
на основе анализа инновационной последовательности (раздел 4.3).

Принцип адаптации (раздел 4.3):
    1. Накапливаем инновации в скользящем окне размера N.
    2. Оцениваем фактическую ковариацию инноваций:
       C_ν = (1/N) * Σ(ν_k * ν_k^T)
    3. Теоретическая ковариация инноваций:
       S_k = H * P⁻ * H^T + R
    4. Если C_ν >> S_k — шум измерений недооценён → увеличиваем R.
       Если C_ν << S_k — шум измерений переоценён → уменьшаем R.
    5. Обновление R:
       R_new = R_old + α * (C_ν - S_k)

    где α ∈ (0, 1] — коэффициент скорости адаптации.

Дополнительные параметры (по сравнению с EKF):
    innovation_window_size (int): Размер скользящего окна (по умолчанию 20).
    adaptation_rate (float): Коэффициент адаптации α (по умолчанию 0.1).
    r_min_scale (float): Минимальный масштаб R (защита от вырождения).
    r_max_scale (float): Максимальный масштаб R (защита от расходимости).

Выход (дополнительно):
    adapted_R: Текущая адаптированная матрица R.
    r_history: История изменения R для визуализации.

Рекомендации по тестированию:
    1. При постоянном шуме R не должна значительно меняться.
    2. При внезапном увеличении шума R должна вырасти.
    3. R_adapted всегда положительно определена.
    4. R_adapted находится в допустимых границах [r_min, r_max].
    5. Адаптивный EKF даёт меньшую RMSE, чем классический при переменном шуме.
    6. При нулевом adaptation_rate поведение = классический EKF.
"""

import logging
from collections import deque
from typing import Optional

import numpy as np

from filters.base_filter import BaseFilter, FilterResult
from models.base_motion_model import BaseMotionModel

logger = logging.getLogger(__name__)


class AdaptiveEKF(BaseFilter):
    """Адаптивный EKF с подстройкой R по инновационной последовательности.

    Наследует структуру от BaseFilter, расширяя шаг update() логикой
    адаптации матрицы R. Стратегия адаптации может быть изменена путём
    переопределения метода _adapt_R() (Open/Closed principle).

    Args:
        motion_model: Модель движения НПА.
        initial_state: Начальный вектор состояния.
        initial_covariance: Начальная ковариационная матрица P0.
        process_noise: Матрица шума процесса Q.
        innovation_window_size: Размер скользящего окна инноваций.
        adaptation_rate: Коэффициент скорости адаптации α.
        r_min_scale: Минимальный масштаб диагональных элементов R.
        r_max_scale: Максимальный масштаб диагональных элементов R.
    """

    def __init__(
        self,
        motion_model: BaseMotionModel,
        initial_state: np.ndarray,
        initial_covariance: np.ndarray,
        process_noise: np.ndarray,
        innovation_window_size: int = 20,
        adaptation_rate: float = 0.1,
        r_min_scale: float = 0.1,
        r_max_scale: float = 10.0,
    ):
        super().__init__(motion_model, initial_state, initial_covariance, process_noise)

        self._window_size = innovation_window_size
        self._alpha = adaptation_rate
        self._r_min_scale = r_min_scale
        self._r_max_scale = r_max_scale

        # Скользящее окно инноваций.
        # Ключ — размерность измерения, значение — deque инноваций.
        # Это позволяет хранить раздельные окна для датчиков разной размерности.
        self._innovation_windows: dict[int, deque] = {}

        # Адаптированные матрицы R для каждой размерности измерения.
        # Ключ — размерность, значение — текущая адаптированная R.
        self._adapted_R: dict[int, np.ndarray] = {}

        # История адаптации R для визуализации
        self._r_history: list[dict] = []

        # Начальные значения R (сохраняем для ограничений масштаба)
        self._initial_R: dict[int, np.ndarray] = {}

        logger.info(
            "Adaptive EKF инициализирован. Окно инноваций: %d, α: %.3f",
            self._window_size,
            self._alpha,
        )

    def predict(self, control: np.ndarray, dt: float) -> FilterResult:
        """Шаг предсказания (идентичен классическому EKF, раздел 2.2).

        Args:
            control: Вектор управления [ω, a].
            dt: Шаг времени Δt (с).

        Returns:
            FilterResult с предсказанными state и covariance.
        """
        # Уравнение процесса (раздел 2.2)
        self._state = self._model.predict(self._state, control, dt)
        # Матрица Якоби F (раздел 2.2)
        F = self._model.jacobian(self._state, control, dt)
        # Прогноз ковариации (раздел 2.2)
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
        """Шаг обновления с адаптацией R.

        Выполняет стандартное обновление EKF, но перед этим адаптирует
        матрицу R на основе инновационной последовательности.

        Args:
            measurement: Вектор измерений z.
            H: Матрица наблюдения (m × n).
            R: Начальная (номинальная) ковариация шума измерений (m × m).

        Returns:
            FilterResult с обновлёнными state, covariance, innovation, kalman_gain.
        """
        m = len(measurement)

        # Сохраняем начальную R при первом вызове для данной размерности
        if m not in self._initial_R:
            self._initial_R[m] = R.copy()
            self._adapted_R[m] = R.copy()
            self._innovation_windows[m] = deque(maxlen=self._window_size)

        # Вычисляем инновацию
        innovation = measurement - H @ self._state
        innovation = self._normalize_innovation(innovation, H)

        # Добавляем инновацию в окно
        self._innovation_windows[m].append(innovation.copy())

        # Адаптация R
        R_adapted = self._adapt_R(m, H)

        # --- Стандартное обновление EKF с адаптированной R ---
        S = H @ self._P @ H.T + R_adapted

        try:
            K = self._P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            logger.warning("Сингулярная матрица S, используем псевдообратную")
            K = self._P @ H.T @ np.linalg.pinv(S)

        self._state = self._state + K @ innovation
        self._state[3] = BaseMotionModel.normalize_angle(self._state[3])

        n = len(self._state)
        I_KH = np.eye(n) - K @ H
        self._P = I_KH @ self._P @ I_KH.T + K @ R_adapted @ K.T
        self._P = 0.5 * (self._P + self._P.T)

        # Сохраняем историю R
        self._r_history.append({
            "measurement_dim": m,
            "R_diag": np.diag(R_adapted).copy(),
        })

        result = FilterResult(
            state=self._state.copy(),
            covariance=self._P.copy(),
            innovation=innovation.copy(),
            kalman_gain=K.copy(),
        )
        self._history.append(result)

        logger.debug(
            "AEKF update: innov_norm=%.4f, R_diag=%s",
            np.linalg.norm(innovation),
            np.diag(R_adapted),
        )
        return result

    def _adapt_R(self, measurement_dim: int, H: np.ndarray) -> np.ndarray:
        """Адаптация матрицы R по инновационной последовательности (раздел 4.3).

        Этот метод может быть переопределён в подклассах для реализации
        других стратегий адаптации (принцип Open/Closed).

        Args:
            measurement_dim: Размерность текущего измерения.
            H: Матрица наблюдения.

        Returns:
            Адаптированная матрица R.
        """
        window = self._innovation_windows[measurement_dim]

        # Нужно накопить достаточно инноваций
        if len(window) < max(2, self._window_size // 2):
            return self._adapted_R[measurement_dim]

        # Эмпирическая ковариация инноваций: C_ν = (1/N) * Σ(ν * ν^T) (раздел 4.3)
        innovations = np.array(list(window))
        C_nu = (innovations.T @ innovations) / len(window)

        # Теоретическая ковариация инноваций: S = H * P⁻ * H^T + R (раздел 4.3)
        S_theoretical = H @ self._P @ H.T + self._adapted_R[measurement_dim]

        # Разница: если C_ν > S, шум измерений недооценён (раздел 4.3)
        delta = C_nu - S_theoretical

        # Обновление R: R_new = R_old + α * δ (раздел 4.3)
        R_new = self._adapted_R[measurement_dim] + self._alpha * delta

        # Ограничения: R должна быть положительно определена
        R_new = self._clamp_R(R_new, measurement_dim)

        self._adapted_R[measurement_dim] = R_new
        return R_new

    def _clamp_R(self, R: np.ndarray, measurement_dim: int) -> np.ndarray:
        """Ограничение диагональных элементов R допустимым диапазоном.

        Защищает от вырождения (R→0) и расходимости (R→∞).

        Args:
            R: Матрица R для ограничения.
            measurement_dim: Размерность измерения.

        Returns:
            Ограниченная матрица R.
        """
        R_init = self._initial_R[measurement_dim]
        R_clamped = R.copy()

        for i in range(R.shape[0]):
            r_min = R_init[i, i] * self._r_min_scale
            r_max = R_init[i, i] * self._r_max_scale
            R_clamped[i, i] = np.clip(R_clamped[i, i], r_min, r_max)

        # Гарантируем симметричность
        R_clamped = 0.5 * (R_clamped + R_clamped.T)
        return R_clamped

    def _normalize_innovation(self, innovation: np.ndarray, H: np.ndarray) -> np.ndarray:
        """Нормализация углов в инновации."""
        innovation = innovation.copy()
        for i in range(H.shape[0]):
            if H.shape[1] > 3 and H[i, 3] != 0:
                innovation[i] = BaseMotionModel.normalize_angle(innovation[i])
        return innovation

    def get_adapted_R(self, measurement_dim: int) -> Optional[np.ndarray]:
        """Получить текущую адаптированную R для заданной размерности.

        Args:
            measurement_dim: Размерность измерения.

        Returns:
            Адаптированная матрица R или None.
        """
        return self._adapted_R.get(measurement_dim, None)

    def get_r_history(self) -> list[dict]:
        """Получить историю адаптации R.

        Returns:
            Список словарей с measurement_dim и R_diag.
        """
        return self._r_history.copy()
