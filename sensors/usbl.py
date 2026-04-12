"""
Модель гидроакустической системы позиционирования (USBL).

USBL измеряет координаты НПА (x, y).
Измерение: [x, y] — индексы 0 и 1 вектора состояния.

Матрица наблюдения: H = [[1, 0, 0, 0],
                          [0, 1, 0, 0]]
Частота: 0.5 Гц (редкие коррекции).

Рекомендации по тестированию:
    - measure() возвращает вектор размерности 2.
    - H имеет размер (2 × 4) и выбирает x и y.
    - is_available() срабатывает каждые 2 секунды.
"""

import numpy as np
from sensors.base_sensor import BaseSensor


class USBL(BaseSensor):
    """Гидроакустическая система — измеряет координаты НПА.

    Args:
        noise_std: СКО шума координат (м). Может быть скаляром (одинаковый для x, y)
            или вектором [σ_x, σ_y].
        rate: Частота обновления (Гц).
    """

    def __init__(self, noise_std: float | np.ndarray = 1.0, rate: float = 0.5):
        # Если скаляр — одинаковый шум для x и y
        if np.isscalar(noise_std):
            noise_std = np.array([noise_std, noise_std])
        super().__init__(
            noise_std=noise_std,
            rate=rate,
            state_dim=4,
            name="USBL",
        )

    def get_H(self) -> np.ndarray:
        """Матрица наблюдения: выбирает координаты (индексы 0, 1).

        Returns:
            H = [[1, 0, 0, 0],
                 [0, 1, 0, 0]] размером (2 × 4).
        """
        H = np.zeros((2, self._state_dim))
        H[0, 0] = 1.0  # x
        H[1, 1] = 1.0  # y
        return H
