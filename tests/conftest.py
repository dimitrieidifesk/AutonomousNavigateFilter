"""
Общие фикстуры для тестов pytest.

Предоставляет переиспользуемые объекты: модели, фильтры, траектории, датчики.
Каждая фикстура изолирована и создаёт свежий экземпляр для каждого теста.
"""

import numpy as np
import pytest

from auv_params import AUVParams
from config import FilterConfig, SensorConfig, SimulationConfig
from models.kinematic_2d import Kinematic2DModel
from filters.ekf import EKF
from filters.adaptive_ekf import AdaptiveEKF
from sensors.dvl import DVL
from sensors.compass import Compass
from sensors.usbl import USBL
from sensors.ins import INS
from simulation.trajectory import TrajectoryGenerator


# ------------------------------------------------------------------
# Параметры НПА
# ------------------------------------------------------------------

@pytest.fixture
def auv_params():
    """Параметры НПА по умолчанию."""
    return AUVParams()


# ------------------------------------------------------------------
# Модель движения
# ------------------------------------------------------------------

@pytest.fixture
def motion_model():
    """Кинематическая модель 2D."""
    return Kinematic2DModel()


# ------------------------------------------------------------------
# Конфигурация фильтров
# ------------------------------------------------------------------

@pytest.fixture
def filter_config(auv_params):
    """Конфигурация фильтров из параметров НПА."""
    return FilterConfig.from_auv_params(auv_params)


@pytest.fixture
def initial_state():
    """Начальное состояние [x=0, y=0, v=0, ψ=0].
    Аппарат стартует с места (раздел 2.5).
    """
    return np.array([0.0, 0.0, 0.0, 0.0])


@pytest.fixture
def P0(auv_params):
    """Начальная ковариационная матрица (из параметров НПА)."""
    return np.diag(auv_params.compute_P0_diag())


@pytest.fixture
def Q(auv_params):
    """Матрица шума процесса (из параметров НПА, раздел 2.1)."""
    return np.diag(auv_params.compute_Q_diag())


@pytest.fixture
def ekf(motion_model, initial_state, P0, Q):
    """Экземпляр классического EKF."""
    return EKF(
        motion_model=motion_model,
        initial_state=initial_state,
        initial_covariance=P0,
        process_noise=Q,
    )


@pytest.fixture
def adaptive_ekf(motion_model, initial_state, P0, Q):
    """Экземпляр адаптивного EKF."""
    return AdaptiveEKF(
        motion_model=motion_model,
        initial_state=initial_state,
        initial_covariance=P0,
        process_noise=Q,
        innovation_window_size=10,
        adaptation_rate=0.1,
    )


@pytest.fixture
def dvl():
    """Датчик DVL (с учётом температурной погрешности, раздел 2.3)."""
    return DVL(noise_std=0.03, rate=5.0)


@pytest.fixture
def compass():
    """Магнитный компас."""
    return Compass(noise_std=0.03, rate=2.0)


@pytest.fixture
def usbl():
    """Гидроакустика USBL."""
    return USBL(noise_std=1.0, rate=0.5)


@pytest.fixture
def ins():
    """ИНС (МЭМС навигационного класса, раздел 2.3)."""
    return INS(
        gyro_noise_std=0.05,
        accel_noise_std=0.1,
        gyro_bias_drift_std=0.0001,
        rate=10.0,
    )


@pytest.fixture
def trajectory_generator():
    """Генератор траекторий (dt=0.1 с, короткий для тестов)."""
    return TrajectoryGenerator(dt=0.1, duration=10.0)


@pytest.fixture
def circle_trajectory(trajectory_generator):
    """Круговая траектория (10 секунд)."""
    return trajectory_generator.generate("circle", radius=50.0, speed=1.5)


@pytest.fixture
def straight_trajectory(trajectory_generator):
    """Прямолинейная траектория (10 секунд)."""
    return trajectory_generator.generate("straight", speed=1.5, heading=0.0)


@pytest.fixture(autouse=True)
def set_random_seed():
    """Фиксированное зерно ГСЧ для воспроизводимости тестов."""
    np.random.seed(42)
