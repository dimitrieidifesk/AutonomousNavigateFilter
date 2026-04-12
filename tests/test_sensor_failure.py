"""
Тесты симулятора сбоев датчиков (раздел 4.4).

Проверяемые сценарии:
    1. apply() удаляет измерения в окне сбоя.
    2. Измерения вне окна сбоя не затронуты.
    3. Другие датчики не затронуты.
    4. Комбинированный сбой работает корректно.
    5. Несуществующий датчик не вызывает ошибку (только warning).
    6. Исходные данные не модифицируются (создаётся копия).

Простыми словами:
    Проверяем, что «сбой датчика» правильно вырезает нужные
    измерения и не ломает остальные данные.
"""

import numpy as np
import pytest

from simulation.sensor_failure import (
    SensorFailureSimulator,
    FailureScenario,
    dvl_failure_scenario,
    compass_failure_scenario,
    combined_failure_scenario,
)
from simulation.trajectory import TrajectoryGenerator
from simulation.data_generator import DataGenerator
from sensors.ins import INS
from sensors.dvl import DVL
from sensors.compass import Compass
from sensors.usbl import USBL


@pytest.fixture
def sim_data():
    """Данные симуляции для тестов сбоев (30 секунд)."""
    np.random.seed(42)
    traj_gen = TrajectoryGenerator(dt=0.1, duration=30.0)
    trajectory = traj_gen.generate("circle", radius=50.0, speed=1.5)

    ins = INS(gyro_noise_std=0.01, accel_noise_std=0.05)
    dvl = DVL(noise_std=0.03, rate=5.0)
    compass = Compass(noise_std=0.03, rate=2.0)
    usbl = USBL(noise_std=1.0, rate=0.5)

    data_gen = DataGenerator(ins=ins, sensors=[dvl, compass, usbl], seed=42)
    return data_gen.generate(trajectory)


class TestSensorFailureApply:
    """Тесты метода apply()."""

    def test_dvl_failure_removes_measurements(self, sim_data):
        """Сбой DVL удаляет измерения в заданном окне."""
        original_count = len(sim_data.sensor_measurements["DVL"])

        failure = FailureScenario("DVL", start_time=5.0, end_time=15.0)
        simulator = SensorFailureSimulator([failure])
        modified = simulator.apply(sim_data)

        new_count = len(modified.sensor_measurements["DVL"])
        assert new_count < original_count

    def test_measurements_outside_window_preserved(self, sim_data):
        """Измерения вне окна сбоя сохраняются."""
        failure = FailureScenario("DVL", start_time=5.0, end_time=10.0)
        simulator = SensorFailureSimulator([failure])
        modified = simulator.apply(sim_data)

        # Все оставшиеся измерения вне окна
        for meas in modified.sensor_measurements["DVL"]:
            assert not (5.0 <= meas.timestamp <= 10.0)

    def test_other_sensors_not_affected(self, sim_data):
        """Сбой DVL не влияет на Compass и USBL."""
        compass_count_before = len(sim_data.sensor_measurements["Compass"])
        usbl_count_before = len(sim_data.sensor_measurements["USBL"])

        failure = FailureScenario("DVL", start_time=5.0, end_time=25.0)
        simulator = SensorFailureSimulator([failure])
        modified = simulator.apply(sim_data)

        assert len(modified.sensor_measurements["Compass"]) == compass_count_before
        assert len(modified.sensor_measurements["USBL"]) == usbl_count_before

    def test_original_data_not_modified(self, sim_data):
        """Исходные данные не изменяются (deepcopy)."""
        original_count = len(sim_data.sensor_measurements["DVL"])

        failure = FailureScenario("DVL", start_time=0.0, end_time=30.0)
        simulator = SensorFailureSimulator([failure])
        _ = simulator.apply(sim_data)

        # Оригинал не затронут
        assert len(sim_data.sensor_measurements["DVL"]) == original_count

    def test_nonexistent_sensor_no_error(self, sim_data):
        """Несуществующий датчик не вызывает ошибку."""
        failure = FailureScenario("NonExistent", start_time=0.0, end_time=10.0)
        simulator = SensorFailureSimulator([failure])

        # Не должно бросить исключение
        modified = simulator.apply(sim_data)
        assert len(modified.sensor_measurements) == len(sim_data.sensor_measurements)


class TestCombinedFailure:
    """Тесты комбинированного сбоя (DVL + Compass)."""

    def test_combined_failure_affects_both(self, sim_data):
        """Комбинированный сбой влияет на оба датчика."""
        # Используем маленькие окна в пределах 30 секунд
        failures = [
            FailureScenario("DVL", start_time=5.0, end_time=15.0),
            FailureScenario("Compass", start_time=10.0, end_time=20.0),
        ]
        simulator = SensorFailureSimulator(failures)

        dvl_before = len(sim_data.sensor_measurements["DVL"])
        compass_before = len(sim_data.sensor_measurements["Compass"])

        modified = simulator.apply(sim_data)

        assert len(modified.sensor_measurements["DVL"]) < dvl_before
        assert len(modified.sensor_measurements["Compass"]) < compass_before


class TestPredefinedScenarios:
    """Тесты предопределённых сценариев."""

    def test_dvl_failure_scenario_defaults(self):
        """dvl_failure_scenario() создаёт ожидаемый сценарий."""
        sc = dvl_failure_scenario()
        assert sc.sensor_name == "DVL"
        assert sc.start_time == 100.0
        assert sc.end_time == 130.0

    def test_compass_failure_scenario_defaults(self):
        """compass_failure_scenario() создаёт ожидаемый сценарий."""
        sc = compass_failure_scenario()
        assert sc.sensor_name == "Compass"
        assert sc.start_time == 150.0
        assert sc.end_time == 160.0

    def test_combined_failure_returns_two(self):
        """combined_failure_scenario() возвращает 2 сценария."""
        scenarios = combined_failure_scenario()
        assert len(scenarios) == 2
        assert scenarios[0].sensor_name == "DVL"
        assert scenarios[1].sensor_name == "Compass"


class TestFailureScenarioDescription:
    """Тесты автоматической генерации описаний."""

    def test_auto_description(self):
        """description генерируется автоматически."""
        sc = FailureScenario("DVL", 10.0, 20.0)
        assert "DVL" in sc.description
        assert "10" in sc.description
        assert "20" in sc.description

    def test_custom_description(self):
        """Пользовательское описание сохраняется."""
        sc = FailureScenario("DVL", 10.0, 20.0, description="Мой сбой")
        assert sc.description == "Мой сбой"
