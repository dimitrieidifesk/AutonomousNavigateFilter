"""
Кинематическая 2D модель движения НПА (3-DOF: x, y, ψ).

Модель движения в горизонтальной плоскости (уравнение процесса, раздел 2.2):
    x_{k+1} = x_k + v_k * cos(ψ_k) * Δt
    y_{k+1} = y_k + v_k * sin(ψ_k) * Δt
    v_{k+1} = v_k + a_k * Δt
    ψ_{k+1} = ψ_k + ω_k * Δt

Вектор состояния:  [x, y, v, ψ]  (размерность 4)
Вектор управления: [ω, a]         (угловая скорость, ускорение)

Матрица Якоби F = ∂f/∂x (раздел 2.2):
    F = | 1  0  cos(ψ)*Δt  -v*sin(ψ)*Δt |
        | 0  1  sin(ψ)*Δt   v*cos(ψ)*Δt |
        | 0  0  1            0            |
        | 0  0  0            1            |

Рекомендации по тестированию:
    1. Прямолинейное движение (ψ=0, ω=0):
       x растёт, y не меняется.
    2. Движение по кругу (ω = const):
       аппарат описывает окружность.
    3. Нулевое управление (ω=0, a=0):
       координаты изменяются равномерно.
    4. Численная проверка Якобиана:
       сравнить с конечными разностями.
"""

import numpy as np
from models.base_motion_model import BaseMotionModel


class Kinematic2DModel(BaseMotionModel):
    """Кинематическая модель НПА в горизонтальной плоскости.

    Нелинейная модель движения с вектором состояния [x, y, v, ψ]
    и управлением [ω, a]. Идеально подходит для EKF благодаря
    аналитически вычисляемой матрице Якоби.
    """

    # Индексы компонент вектора состояния
    IX = 0   # координата x (север)
    IY = 1   # координата y (восток)
    IV = 2   # скорость
    IPSI = 3 # курс (рыскание)

    STATE_DIM = 4
    CONTROL_DIM = 2  # [ω, a]

    def predict(self, state: np.ndarray, control: np.ndarray, dt: float) -> np.ndarray:
        """Предсказание следующего состояния по кинематической модели.

        Уравнение процесса (раздел 2.2):
            x' = x + v * cos(ψ) * Δt
            y' = y + v * sin(ψ) * Δt
            v' = v + a * Δt
            ψ' = ψ + ω * Δt

        Args:
            state: Текущее состояние [x, y, v, ψ].
            control: Управление [ω (рад/с), a (м/с²)].
            dt: Шаг времени Δt (с).

        Returns:
            Предсказанный вектор состояния [x', y', v', ψ'].
        """
        x, y, v, psi = state
        omega, accel = control

        # Модель движения (уравнение процесса, раздел 2.2)
        x_new = x + v * np.cos(psi) * dt
        y_new = y + v * np.sin(psi) * dt
        v_new = v + accel * dt
        psi_new = self.normalize_angle(psi + omega * dt)

        return np.array([x_new, y_new, v_new, psi_new])

    def jacobian(self, state: np.ndarray, control: np.ndarray, dt: float) -> np.ndarray:
        """Аналитическая матрица Якоби F = ∂f/∂x (раздел 2.2).

        Args:
            state: Текущее состояние [x, y, v, ψ].
            control: Управление [ω, a] (не используется в Якобиане).
            dt: Шаг времени Δt (с).

        Returns:
            Матрица Якоби F размером 4×4.
        """
        _, _, v, psi = state

        # Матрица Якоби F (раздел 2.2)
        F = np.eye(self.STATE_DIM)

        # ∂x'/∂v = cos(ψ) * dt
        F[self.IX, self.IV] = np.cos(psi) * dt
        # ∂x'/∂ψ = -v * sin(ψ) * dt
        F[self.IX, self.IPSI] = -v * np.sin(psi) * dt

        # ∂y'/∂v = sin(ψ) * dt
        F[self.IY, self.IV] = np.sin(psi) * dt
        # ∂y'/∂ψ = v * cos(ψ) * dt
        F[self.IY, self.IPSI] = v * np.cos(psi) * dt

        return F

    def process_noise_matrix(self, state: np.ndarray, dt: float) -> np.ndarray:
        """Матрица влияния шума процесса G (опционально).

        Используется для более точного формирования Q = G * Qc * G^T,
        где Qc — ковариационная матрица непрерывного шума.

        Args:
            state: Текущее состояние [x, y, v, ψ].
            dt: Шаг времени (с).

        Returns:
            Матрица G размером (4 × 2).
        """
        _, _, _, psi = state

        G = np.zeros((self.STATE_DIM, self.CONTROL_DIM))

        # Влияние ускорения a на координаты через скорость
        G[self.IX, 1] = 0.5 * np.cos(psi) * dt**2
        G[self.IY, 1] = 0.5 * np.sin(psi) * dt**2
        G[self.IV, 1] = dt

        # Влияние угловой скорости ω на курс
        G[self.IPSI, 0] = dt

        return G
