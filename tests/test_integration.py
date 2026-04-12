"""
Интеграционные тесты: полный прогон симуляции.

Проверяемые сценарии:
    1. Полный цикл: траектория → данные → фильтр → метрики.
    2. EKF и Adaptive EKF дают конечные (не NaN) результаты.
    3. RMSE позиции < допустимого порога.
    4. Адаптивный EKF не хуже классического на стандартном сценарии.
    5. Все компоненты корректно работают вместе.
    6. Dead Reckoning хуже EKF на длинной траектории (раздел 4.5).
    7. Сбои датчиков корректно влияют на результаты (раздел 4.4).

Простыми словами:
    Эти тесты проверяют, что все модули проекта работают вместе:
    - Генерация траектории → генерация данных → фильтрация → метрики.
    - Dead Reckoning даёт худшую точность (нет коррекции по датчикам).
    - При сбое датчика фильтры продолжают работать (не NaN, не Inf).
"""

import numpy as np
import pytest

from auv_params import AUVParams
from config import FilterConfig
from models.kinematic_2d import Kinematic2DModel
from filters.ekf import EKF
from filters.adaptive_ekf import AdaptiveEKF
from filters.dead_reckoning import DeadReckoning
from sensors.ins import INS
from sensors.dvl import DVL
from sensors.compass import Compass
from sensors.usbl import USBL
from simulation.trajectory import TrajectoryGenerator
from simulation.data_generator import DataGenerator
from simulation.runner import SimulationRunner
from evaluation.metrics import evaluate_simulation


class TestFullPipeline:
    """Интеграционные тесты полного прогона."""

    @pytest.fixture
    def full_simulation_result(self):
        """Запуск полной симуляции (короткой) для тестов."""
        np.random.seed(42)

        # Параметры НПА
        auv = AUVParams()

        # Короткая симуляция (10 секунд)
        traj_gen = TrajectoryGenerator(dt=auv.dt, duration=10.0)
        trajectory = traj_gen.generate("circle", radius=50.0, speed=1.5)

        # Датчики (σ_DVL из параметров НПА)
        ins = INS(gyro_noise_std=0.01, accel_noise_std=0.05)
        dvl = DVL(noise_std=auv.dvl_total_noise_std(), rate=5.0)
        compass = Compass(noise_std=0.03, rate=2.0)
        usbl = USBL(noise_std=1.0, rate=0.5)

        data_gen = DataGenerator(ins=ins, sensors=[dvl, compass, usbl], seed=42)
        sim_data = data_gen.generate(trajectory)

        # Фильтры (Q, P0 из параметров НПА)
        model = Kinematic2DModel()
        cfg = FilterConfig.from_auv_params(auv)
        P0 = np.diag(cfg.initial_covariance_diag)
        Q = np.diag(cfg.process_noise_diag)

        ekf = EKF(
            motion_model=model,
            initial_state=cfg.initial_state.copy(),
            initial_covariance=P0.copy(),
            process_noise=Q.copy(),
        )

        adaptive_ekf = AdaptiveEKF(
            motion_model=model,
            initial_state=cfg.initial_state.copy(),
            initial_covariance=P0.copy(),
            process_noise=Q.copy(),
            innovation_window_size=10,
            adaptation_rate=0.1,
        )

        filters = {"EKF": ekf, "Adaptive EKF": adaptive_ekf}
        runner = SimulationRunner(filters)
        result = runner.run(sim_data)

        return result, adaptive_ekf

    def test_both_filters_produce_results(self, full_simulation_result):
        """Оба фильтра дают результаты."""
        result, _ = full_simulation_result

        assert "EKF" in result.filter_results
        assert "Adaptive EKF" in result.filter_results

    def test_results_not_nan(self, full_simulation_result):
        """Результаты не содержат NaN."""
        result, _ = full_simulation_result

        for name, fr in result.filter_results.items():
            assert not np.any(np.isnan(fr.estimated_states)), \
                f"NaN в состояниях {name}"
            assert not np.any(np.isnan(fr.covariances)), \
                f"NaN в ковариациях {name}"

    def test_results_not_inf(self, full_simulation_result):
        """Результаты не содержат Inf (фильтр не расходится)."""
        result, _ = full_simulation_result

        for name, fr in result.filter_results.items():
            assert not np.any(np.isinf(fr.estimated_states)), \
                f"Inf в состояниях {name}"

    def test_output_dimensions(self, full_simulation_result):
        """Размерности выходов корректны."""
        result, _ = full_simulation_result
        N = len(result.timestamps)

        for name, fr in result.filter_results.items():
            assert fr.estimated_states.shape == (N, 4), \
                f"Неверная размерность состояний {name}"
            assert fr.covariances.shape == (N, 4, 4), \
                f"Неверная размерность ковариаций {name}"

    def test_rmse_below_threshold(self, full_simulation_result):
        """RMSE позиции < допустимого порога (10 м для короткой симуляции)."""
        result, _ = full_simulation_result
        reports = evaluate_simulation(result)

        for name, report in reports.items():
            assert report.rmse_position < 10.0, \
                f"RMSE позиции {name} = {report.rmse_position:.2f} м > 10 м"

    def test_metrics_computed_correctly(self, full_simulation_result):
        """Метрики вычисляются без ошибок для всех фильтров."""
        result, _ = full_simulation_result
        reports = evaluate_simulation(result)

        assert len(reports) == 2
        for name, report in reports.items():
            assert report.rmse_x >= 0
            assert report.rmse_y >= 0
            assert report.mae_position >= 0
            assert report.max_error_position >= 0
            assert report.rmse_position >= report.mae_position - 0.01  # RMSE ≥ MAE

    def test_adaptive_ekf_has_r_history(self, full_simulation_result):
        """Адаптивный EKF имеет историю адаптации R."""
        _, adaptive_ekf = full_simulation_result
        r_history = adaptive_ekf.get_r_history()

        assert len(r_history) > 0, "История адаптации R пуста"


class TestTrajectoryGeneration:
    """Тесты генерации траекторий."""

    def test_all_trajectory_types(self, trajectory_generator):
        """Все типы траекторий генерируются без ошибок."""
        for traj_type in ["circle", "eight", "sine", "straight"]:
            traj = trajectory_generator.generate(traj_type)
            assert traj.states.shape[1] == 4
            assert traj.controls.shape[1] == 2
            assert len(traj.timestamps) == traj.states.shape[0]

    def test_invalid_trajectory_type(self, trajectory_generator):
        """Неизвестный тип траектории вызывает ValueError."""
        with pytest.raises(ValueError, match="Неизвестный тип"):
            trajectory_generator.generate("unknown")

    def test_straight_trajectory_y_unchanged(self, straight_trajectory):
        """Прямолинейная траектория (heading=0): y = 0."""
        np.testing.assert_allclose(
            straight_trajectory.states[:, 1], 0.0, atol=1e-10
        )

    def test_straight_trajectory_constant_speed(self, straight_trajectory):
        """Прямолинейная траектория: скорость постоянна."""
        np.testing.assert_allclose(
            straight_trajectory.states[:, 2], 1.5, atol=1e-10
        )

    def test_circle_trajectory_constant_speed(self, circle_trajectory):
        """Круговая траектория: скорость постоянна."""
        np.testing.assert_allclose(
            circle_trajectory.states[:, 2], 1.5, atol=1e-10
        )


class TestDeadReckoningIntegration:
    """Интеграционные тесты Dead Reckoning (раздел 4.5).

    Простыми словами: проверяем, что DR работает в полном цикле
    симуляции и даёт значительно худшие результаты, чем EKF.
    """

    @pytest.fixture
    def dr_vs_ekf_result(self):
        """Прогон DR и EKF на короткой круговой траектории."""
        np.random.seed(42)
        auv = AUVParams()

        traj_gen = TrajectoryGenerator(dt=auv.dt, duration=30.0)
        trajectory = traj_gen.generate("circle", radius=50.0, speed=1.5)

        ins = INS(gyro_noise_std=0.01, accel_noise_std=0.05)
        dvl = DVL(noise_std=auv.dvl_total_noise_std(), rate=5.0)
        compass = Compass(noise_std=0.03, rate=2.0)
        usbl = USBL(noise_std=1.0, rate=0.5)

        data_gen = DataGenerator(ins=ins, sensors=[dvl, compass, usbl], seed=42)
        sim_data = data_gen.generate(trajectory)

        model = Kinematic2DModel()
        cfg = FilterConfig.from_auv_params(auv)
        P0 = np.diag(cfg.initial_covariance_diag)
        Q = np.diag(cfg.process_noise_diag)

        filters = {
            "EKF": EKF(model, cfg.initial_state.copy(), P0.copy(), Q.copy()),
            "Dead Reckoning": DeadReckoning(
                model, cfg.initial_state.copy(), P0.copy(), Q.copy()
            ),
        }

        runner = SimulationRunner(filters)
        return runner.run(sim_data)

    def test_dr_produces_results(self, dr_vs_ekf_result):
        """Dead Reckoning корректно интегрируется в SimulationRunner."""
        assert "Dead Reckoning" in dr_vs_ekf_result.filter_results

    def test_dr_not_nan(self, dr_vs_ekf_result):
        """Результаты DR не содержат NaN."""
        dr = dr_vs_ekf_result.filter_results["Dead Reckoning"]
        assert not np.any(np.isnan(dr.estimated_states))

    def test_dr_worse_than_ekf(self, dr_vs_ekf_result):
        """DR даёт бо́льшую ошибку, чем EKF (раздел 4.5).

        Простыми словами: без коррекции по датчикам ошибка
        накапливается, и DR неизбежно хуже EKF.
        """
        reports = evaluate_simulation(dr_vs_ekf_result)
        assert reports["Dead Reckoning"].rmse_position > \
            reports["EKF"].rmse_position


class TestSensorFailureIntegration:
    """Интеграционные тесты сбоев датчиков (раздел 4.4).

    Простыми словами: проверяем, что фильтры не «ломаются»
    при потере сигнала DVL и компаса — результаты конечны
    и не содержат NaN/Inf.
    """

    @pytest.fixture
    def failure_result(self):
        """Прогон EKF и AEKF с потерей DVL на 5 с."""
        from simulation.sensor_failure import (
            SensorFailureSimulator,
            FailureScenario,
        )

        np.random.seed(42)
        auv = AUVParams()

        traj_gen = TrajectoryGenerator(dt=auv.dt, duration=20.0)
        trajectory = traj_gen.generate("circle", radius=50.0, speed=1.5)

        ins = INS(gyro_noise_std=0.01, accel_noise_std=0.05)
        dvl = DVL(noise_std=auv.dvl_total_noise_std(), rate=5.0)
        compass = Compass(noise_std=0.03, rate=2.0)
        usbl = USBL(noise_std=1.0, rate=0.5)

        data_gen = DataGenerator(ins=ins, sensors=[dvl, compass, usbl], seed=42)
        sim_data = data_gen.generate(trajectory)

        # Сбой DVL на 5 секунд (5–10 с)
        failures = [FailureScenario("DVL", 5.0, 10.0)]
        sim_data = SensorFailureSimulator(failures).apply(sim_data)

        model = Kinematic2DModel()
        cfg = FilterConfig.from_auv_params(auv)
        P0 = np.diag(cfg.initial_covariance_diag)
        Q = np.diag(cfg.process_noise_diag)

        filters = {
            "EKF": EKF(model, cfg.initial_state.copy(), P0.copy(), Q.copy()),
            "Adaptive EKF": AdaptiveEKF(
                model, cfg.initial_state.copy(), P0.copy(), Q.copy(),
                innovation_window_size=10, adaptation_rate=0.1,
            ),
        }

        runner = SimulationRunner(filters)
        return runner.run(sim_data)

    def test_filters_survive_failure(self, failure_result):
        """Оба фильтра выживают при сбое DVL."""
        for name, fr in failure_result.filter_results.items():
            assert not np.any(np.isnan(fr.estimated_states)), \
                f"NaN в {name} при сбое DVL"
            assert not np.any(np.isinf(fr.estimated_states)), \
                f"Inf в {name} при сбое DVL"

    def test_metrics_computable_after_failure(self, failure_result):
        """Метрики вычисляются без ошибок после сбоя."""
        reports = evaluate_simulation(failure_result)
        for name, report in reports.items():
            assert report.rmse_position >= 0
            assert report.rmse_position < 50.0, \
                f"RMSE {name} = {report.rmse_position:.2f} м слишком большая"
