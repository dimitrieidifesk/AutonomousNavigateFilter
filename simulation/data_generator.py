"""
Генератор синтетических данных датчиков.

Принимает эталонную траекторию и модели датчиков, генерирует
зашумлённые измерения с учётом частоты каждого датчика.

Вход:
    TrajectoryData: Эталонная траектория с истинными состояниями и управлениями.
    Список датчиков: [DVL, Compass, USBL] и INS.
    SensorConfig: Параметры шумов.

Выход:
    SimulationData: Все зашумлённые измерения и управления,
        синхронизированные по времени.

Рекомендации по тестированию:
    - Проверить, что количество измерений DVL ≈ duration * dvl_rate.
    - Проверить, что измерения USBL появляются каждые 2 секунды.
    - Проверить, что управления ИНС генерируются на каждом шаге.
"""

from dataclasses import dataclass, field

import numpy as np

from sensors.base_sensor import BaseSensor, SensorMeasurement
from sensors.ins import INS
from simulation.trajectory import TrajectoryData


@dataclass
class SimulationData:
    """Полный набор данных для прогона фильтра.

    Attributes:
        trajectory: Эталонная траектория (ground truth).
        ins_controls: Зашумлённые управления ИНС (N × 2): [ω, a].
        sensor_measurements: Словарь {имя_датчика: список SensorMeasurement}.
    """
    trajectory: TrajectoryData
    ins_controls: np.ndarray
    sensor_measurements: dict[str, list[SensorMeasurement]] = field(
        default_factory=dict
    )


class DataGenerator:
    """Генератор синтетических данных для симуляции.

    Создаёт зашумлённые данные датчиков из эталонной траектории
    с учётом различных частот обновления.

    Args:
        ins: Модель ИНС.
        sensors: Список моделей датчиков (DVL, Compass, USBL и т.д.).
        seed: Зерно ГСЧ для воспроизводимости.
    """

    def __init__(
        self,
        ins: INS,
        sensors: list[BaseSensor],
        seed: int = 42,
    ):
        self._ins = ins
        self._sensors = sensors
        self._seed = seed

    def generate(self, trajectory: TrajectoryData) -> SimulationData:
        """Генерация зашумлённых данных для всей траектории.

        Args:
            trajectory: Эталонная траектория.

        Returns:
            SimulationData со всеми зашумлёнными измерениями.
        """
        np.random.seed(self._seed)

        N = len(trajectory.timestamps)
        dt = trajectory.timestamps[1] - trajectory.timestamps[0] if N > 1 else 0.02

        # Генерация зашумлённых управлений ИНС
        ins_controls = np.zeros((N, 2))
        for i in range(N):
            true_omega = trajectory.controls[i, 0]
            true_accel = trajectory.controls[i, 1]
            ins_controls[i] = self._ins.get_control(true_omega, true_accel)

        # Генерация измерений от каждого датчика
        sensor_measurements: dict[str, list[SensorMeasurement]] = {
            sensor.name: [] for sensor in self._sensors
        }

        for sensor in self._sensors:
            for i, t in enumerate(trajectory.timestamps):
                if sensor.is_available(t, dt):
                    meas = sensor.measure(trajectory.states[i], t)
                    sensor_measurements[sensor.name].append(meas)

        return SimulationData(
            trajectory=trajectory,
            ins_controls=ins_controls,
            sensor_measurements=sensor_measurements,
        )
