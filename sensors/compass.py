"""
Модель магнитного компаса.

Компас измеряет курс (рыскание) НПА.
Измерение: ψ (курс, рад) — индекс 3 вектора состояния.

Матрица наблюдения: H = [0, 0, 0, 1]
Частота: 2 Гц.

Рекомендации по тестированию:
    - measure() при ψ = π/4 должен возвращать значение ≈ π/4 ± шум.
    - Инновация для курса должна быть нормализована в [-π, π].
"""

import numpy as np
from sensors.base_sensor import BaseSensor


class Compass(BaseSensor):
    """Магнитный компас — измеряет курс НПА.

    Args:
        noise_std: СКО шума курса (рад).
        rate: Частота обновления (Гц).
    """

    def __init__(self, noise_std: float = 0.03, rate: float = 2.0):
        super().__init__(
            noise_std=noise_std,
            rate=rate,
            state_dim=4,
            name="Compass",
        )

    def get_H(self) -> np.ndarray:
        """Матрица наблюдения: выбирает курс (индекс 3).

        Returns:
            H = [[0, 0, 0, 1]] размером (1 × 4).
        """
        H = np.zeros((1, self._state_dim))
        H[0, 3] = 1.0  # ψ
        return H
