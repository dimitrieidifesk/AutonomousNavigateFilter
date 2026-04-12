"""
Глобальная конфигурация проекта.

Все параметры симуляции, шумов датчиков и настроек фильтров
собраны в одном месте для удобного управления экспериментами.

Параметры фильтра (Q, P0) могут быть заданы вручную или вычислены
автоматически из физических параметров НПА (см. AUVParams).

Использование:
    from config import SimulationConfig, SensorConfig, FilterConfig
    from auv_params import AUVParams

    # Вариант 1: ручная конфигурация
    cfg = FilterConfig()

    # Вариант 2: конфигурация на основе параметров НПА (рекомендуемый)
    params = AUVParams()
    cfg = FilterConfig.from_auv_params(params)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from auv_params import AUVParams


@dataclass
class SimulationConfig:
    """Параметры симуляции.

    Attributes:
        dt: Базовый шаг интеграции Δt (с). По умолчанию 0.1 с (раздел 2.1).
        duration: Общая длительность симуляции (с).
        seed: Зерно генератора случайных чисел для воспроизводимости.
    """
    dt: float = 0.1            # 10 Гц, Δt = 0.1 с (раздел 2.1)
    duration: float = 300.0    # 5 минут
    seed: int = 42


@dataclass
class SensorConfig:
    """Параметры шумов датчиков.

    Шумы задаются как СКО (стандартное отклонение) для каждого датчика.
    Частоты обновления задаются в герцах.

    Attributes:
        ins_gyro_noise_std: СКО шума гироскопа (рад/с).
        ins_accel_noise_std: СКО шума акселерометра (м/с²).
        ins_rate: Частота обновления ИНС (Гц).
        dvl_speed_noise_std: СКО шума DVL по скорости (м/с).
            Включает температурную погрешность (раздел 2.3).
        dvl_rate: Частота обновления DVL (Гц).
        compass_heading_noise_std: СКО шума компаса (рад).
        compass_rate: Частота обновления компаса (Гц).
        usbl_position_noise_std: СКО шума гидроакустики по координатам (м).
        usbl_rate: Частота обновления гидроакустики (Гц).
    """
    # ИНС (раздел 2.3)
    ins_gyro_noise_std: float = 0.01        # рад/с
    ins_accel_noise_std: float = 0.05       # м/с²
    ins_rate: float = 10.0                  # Гц (при dt=0.1 с)

    # DVL — σ_DVL = 0.03 м/с с учётом температурной погрешности (раздел 2.3)
    dvl_speed_noise_std: float = 0.03       # м/с (базовая 0.02 + темп. 0.01)
    dvl_rate: float = 5.0                   # Гц

    # Магнитный компас (раздел 2.3)
    compass_heading_noise_std: float = 0.03  # рад (~1.7°)
    compass_rate: float = 2.0                # Гц

    # Гидроакустика USBL (раздел 2.3)
    usbl_position_noise_std: float = 1.0    # м
    usbl_rate: float = 0.5                  # Гц

    @staticmethod
    def from_auv_params(params: AUVParams) -> SensorConfig:
        """Создать конфигурацию датчиков с учётом параметров НПА.

        Вычисляет суммарный шум DVL (базовый + температурный).

        Args:
            params: Физические параметры НПА.

        Returns:
            SensorConfig с вычисленными шумами.
        """
        from auv_params import AUVParams  # noqa: F811
        return SensorConfig(
            dvl_speed_noise_std=params.dvl_total_noise_std(),
        )


@dataclass
class FilterConfig:
    """Параметры фильтров Калмана.

    Матрицы Q и P0 могут задаваться вручную или вычисляться из
    физических параметров НПА через фабричный метод from_auv_params().

    Attributes:
        state_dim: Размерность вектора состояния [x, y, v, ψ].
        initial_state: Начальное состояние [x0, y0, v0, ψ0].
            Аппарат стартует с места: v0 = 0 (раздел 2.5).
        initial_covariance_diag: Диагональ начальной ковариационной матрицы P0.
            Вычисляется из начальных неопределённостей (раздел 2.5).
        process_noise_diag: Диагональ матрицы шума процесса Q.
            Вычисляется из параметров НПА по правилу 3σ (раздел 2.1).
        innovation_window_size: Размер окна для оценки инновационной
            последовательности (только для адаптивного EKF, раздел 4.3).
        adaptation_rate: Коэффициент скорости адаптации матрицы R
            (только для адаптивного EKF). Диапазон (0, 1].
        r_min_scale: Минимальный масштаб R (защита от вырождения).
        r_max_scale: Максимальный масштаб R (защита от расходимости).
    """
    state_dim: int = 4  # [x, y, v, ψ]

    # Начальное состояние: аппарат неподвижен (раздел 2.5)
    initial_state: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 0.0, 0.0])
    )
    # P0: σ_x0 = σ_y0 = 10 м, σ_v0 = 0.5 м/с, σ_ψ0 = 0.5° (раздел 2.5)
    initial_covariance_diag: np.ndarray = field(
        default_factory=lambda: np.array([
            100.0,                          # σ²_x0 = 10² м²
            100.0,                          # σ²_y0 = 10² м²
            0.25,                           # σ²_v0 = 0.5² (м/с)²
            np.deg2rad(0.5) ** 2,           # σ²_ψ0 = (0.5°)² рад²
        ])
    )
    # Q: шум процесса, вычисляется из параметров НПА (раздел 2.1).
    # По умолчанию — значения для стандартного НПА (V_max=2, τ=3, ψ̇_max=5°/с, dt=0.1).
    # Q[0]=Q[1]=0: координаты — интегральные величины,
    # их неопределённость нарастает через модель движения.
    process_noise_diag: np.ndarray = field(
        default_factory=lambda: np.array([
            0.0,                            # σ²_wx = 0 (интегральная величина)
            0.0,                            # σ²_wy = 0 (интегральная величина)
            ((2.0 / 3.0 * 0.1) / 3.0) ** 2,  # σ²_wV = (ΔV_max/3)²
            (np.deg2rad(5.0) * 0.1 / 3.0) ** 2,  # σ²_wψ = (Δψ_max/3)²
        ])
    )

    # Параметры адаптации (Adaptive EKF, раздел 4.3)
    innovation_window_size: int = 20
    adaptation_rate: float = 0.1
    r_min_scale: float = 0.1
    r_max_scale: float = 10.0

    @staticmethod
    def from_auv_params(params: AUVParams) -> FilterConfig:
        """Создать конфигурацию фильтра на основе параметров НПА.

        Автоматически вычисляет Q и P0 из физических параметров
        аппарата и условий среды (раздел 2.1, 2.5).

        Args:
            params: Физические параметры НПА.

        Returns:
            FilterConfig с вычисленными Q и P0.
        """
        from auv_params import AUVParams  # noqa: F811
        return FilterConfig(
            initial_state=np.array([0.0, 0.0, 0.0, 0.0]),
            initial_covariance_diag=params.compute_P0_diag(),
            process_noise_diag=params.compute_Q_diag(),
        )


@dataclass
class TrajectoryConfig:
    """Параметры эталонных траекторий.

    Attributes:
        trajectory_type: Тип траектории ('circle', 'eight', 'sine', 'complex').
        radius: Радиус для круговой траектории (м).
        speed: Скорость движения НПА (м/с).
        amplitude: Амплитуда для синусоидальной траектории (м).
        frequency: Частота для синусоидальной траектории (Гц).
    """
    trajectory_type: str = "circle"
    radius: float = 50.0
    speed: float = 1.5       # м/с (~3 узла)
    amplitude: float = 30.0  # для синусоиды
    frequency: float = 0.01  # для синусоиды
