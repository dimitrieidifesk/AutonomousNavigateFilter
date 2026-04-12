"""
Модель инерциальной навигационной системы (ИНС).

ИНС измеряет:
    - ω (угловая скорость рыскания, рад/с) — гироскоп
    - a (линейное ускорение, м/с²) — акселерометр

ИНС не является датчиком с матрицей H, так как она поставляет
управляющие сигналы (control input), а не прямые измерения состояния.
Поэтому ИНС обрабатывается особым образом: её данные подаются
в шаг predict() фильтра, а не в update().

Частота: 50 Гц (основная частота работы фильтра).

Рекомендации по тестированию:
    - Проверить, что get_control() возвращает [ω, a] правильной размерности.
    - Проверить, что среднее шума ≈ 0 на большой выборке.
    - Проверить, что СКО шума ≈ заявленному значению.
"""

import numpy as np


class INS:
    """Модель ИНС (гироскоп + акселерометр).

    ИНС предоставляет зашумлённые управляющие сигналы [ω, a],
    которые используются на этапе predict() фильтра.

    Args:
        gyro_noise_std: СКО шума гироскопа (рад/с).
        accel_noise_std: СКО шума акселерометра (м/с²).
        rate: Частота обновления ИНС (Гц).
    """

    def __init__(
        self,
        gyro_noise_std: float = 0.01,
        accel_noise_std: float = 0.05,
        rate: float = 50.0,
    ):
        self._gyro_noise_std = gyro_noise_std
        self._accel_noise_std = accel_noise_std
        self._rate = rate
        self._name = "INS"

    def get_control(self, true_omega: float, true_accel: float) -> np.ndarray:
        """Генерация зашумлённого управляющего сигнала.

        Args:
            true_omega: Истинная угловая скорость (рад/с).
            true_accel: Истинное ускорение (м/с²).

        Returns:
            Зашумлённый вектор управления [ω_noisy, a_noisy].
        """
        omega_noisy = true_omega + np.random.normal(0, self._gyro_noise_std)
        accel_noisy = true_accel + np.random.normal(0, self._accel_noise_std)
        return np.array([omega_noisy, accel_noisy])

    @property
    def rate(self) -> float:
        """Частота обновления (Гц)."""
        return self._rate

    @property
    def name(self) -> str:
        """Имя датчика."""
        return self._name
