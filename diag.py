"""Диагностика: почему ошибка на окружности слишком маленькая."""
import numpy as np
import logging
logging.disable(logging.CRITICAL)

from auv_params import AUVParams
from config import SimulationConfig, SensorConfig, FilterConfig, TrajectoryConfig
from models.kinematic_2d import Kinematic2DModel
from filters.ekf import EKF
from sensors.ins import INS
from sensors.dvl import DVL
from sensors.compass import Compass
from sensors.usbl import USBL
from simulation.trajectory import TrajectoryGenerator
from simulation.data_generator import DataGenerator
from simulation.runner import SimulationRunner
from evaluation.metrics import evaluate_simulation, position_error_norm

auv = AUVParams()
sim_cfg = SimulationConfig(dt=auv.dt)
sensor_cfg = SensorConfig(dvl_speed_noise_std=auv.dvl_total_noise_std())
filter_cfg = FilterConfig.from_auv_params(auv)
np.random.seed(sim_cfg.seed)

motion_model = Kinematic2DModel()
traj_gen = TrajectoryGenerator(dt=sim_cfg.dt, duration=sim_cfg.duration)
ins = INS(gyro_noise_std=sensor_cfg.ins_gyro_noise_std,
          accel_noise_std=sensor_cfg.ins_accel_noise_std,
          gyro_bias_drift_std=sensor_cfg.ins_gyro_bias_drift_std,
          rate=sensor_cfg.ins_rate)
sensors_list = [
    DVL(noise_std=sensor_cfg.dvl_speed_noise_std, rate=sensor_cfg.dvl_rate),
    Compass(noise_std=sensor_cfg.compass_heading_noise_std, rate=sensor_cfg.compass_rate),
    USBL(noise_std=sensor_cfg.usbl_position_noise_std, rate=sensor_cfg.usbl_rate),
]
data_gen = DataGenerator(ins=ins, sensors=sensors_list, seed=sim_cfg.seed)

trajectory = traj_gen.generate("circle", radius=50.0, speed=1.5)
sim_data = data_gen.generate(trajectory)

P0 = np.diag(filter_cfg.initial_covariance_diag)
Q = np.diag(filter_cfg.process_noise_diag)

ekf = EKF(motion_model=motion_model,
          initial_state=filter_cfg.initial_state.copy(),
          initial_covariance=P0.copy(),
          process_noise=Q.copy())

runner = SimulationRunner({"EKF": ekf})
result = runner.run(sim_data)
reports = evaluate_simulation(result)
r = reports["EKF"]

print("=" * 60)
print("ДИАГНОСТИКА ОШИБОК (круговая траектория)")
print("=" * 60)
print(f"RMSE position: {r.rmse_position:.6f} m")
print(f"RMSE speed:    {r.rmse_speed:.6f} m/s")
print(f"RMSE heading:  {r.rmse_heading:.6f} rad ({np.rad2deg(r.rmse_heading):.4f} deg)")
print(f"Max error pos: {r.max_error_position:.6f} m")
print(f"Mean error:    {r.mean_error_position:.6f} m")
print()

err = position_error_norm(result.filter_results["EKF"].estimated_states[:, :2],
                          result.true_states[:, :2])
print(f"Error first 10s (mean): {err[:100].mean():.4f} m")
print(f"Error 10-50s (mean):    {err[100:500].mean():.4f} m")
print(f"Error 50-300s (mean):   {err[500:].mean():.4f} m")
print()

dvl_meas = sim_data.sensor_measurements["DVL"]
compass_meas = sim_data.sensor_measurements["Compass"]
usbl_meas = sim_data.sensor_measurements["USBL"]
print(f"Num DVL meas:  {len(dvl_meas)}")
print(f"Num Compass:   {len(compass_meas)}")
print(f"Num USBL:      {len(usbl_meas)}")
print()

# Проверим: шум ИНС
true_controls = trajectory.controls
ins_controls = sim_data.ins_controls
ctrl_noise = ins_controls - true_controls
print(f"INS omega noise std (факт): {ctrl_noise[:, 0].std():.6f} (ожид: {sensor_cfg.ins_gyro_noise_std})")
print(f"INS accel noise std (факт): {ctrl_noise[:, 1].std():.6f} (ожид: {sensor_cfg.ins_accel_noise_std})")
print()

# Проверим: шум DVL
dvl_true_v = np.array([trajectory.states[int(round(m.timestamp / sim_cfg.dt)), 2] for m in dvl_meas])
dvl_meas_v = np.array([m.value[0] for m in dvl_meas])
dvl_noise = dvl_meas_v - dvl_true_v
print(f"DVL noise std (факт): {dvl_noise.std():.6f} (ожид: {sensor_cfg.dvl_speed_noise_std})")
print()

# Проверим: шум USBL
usbl_true_xy = np.array([trajectory.states[int(round(m.timestamp / sim_cfg.dt)), :2] for m in usbl_meas])
usbl_meas_xy = np.array([m.value for m in usbl_meas])
usbl_noise = usbl_meas_xy - usbl_true_xy
print(f"USBL noise std X (факт): {usbl_noise[:, 0].std():.4f} (ожид: {sensor_cfg.usbl_position_noise_std})")
print(f"USBL noise std Y (факт): {usbl_noise[:, 1].std():.4f} (ожид: {sensor_cfg.usbl_position_noise_std})")
print()

# Ковариация в стабильном режиме
P_final = result.filter_results["EKF"].covariances[-1]
print("P diagonal (final):", np.diag(P_final))
print(f"  sigma_x = {np.sqrt(P_final[0,0]):.4f} m")
print(f"  sigma_y = {np.sqrt(P_final[1,1]):.4f} m")
print(f"  sigma_v = {np.sqrt(P_final[2,2]):.6f} m/s")
print(f"  sigma_psi = {np.sqrt(P_final[3,3]):.6f} rad ({np.rad2deg(np.sqrt(P_final[3,3])):.4f} deg)")
print()

# Ключевой вопрос: сравним предсказанную sigma_x,y с фактической ошибкой
print(f"Предсказанная sigma_pos = {np.sqrt(P_final[0,0] + P_final[1,1]):.4f} m")
print(f"Фактическая средняя ошибка = {err[500:].mean():.4f} m")
print()

# Посмотрим Q
print("Q diagonal:", np.diag(Q))
print(f"  Q[0,1] = 0 (координаты, интегральные)")
print(f"  Q[2] = {Q[2,2]:.8f} (sigma_wV^2)")
print(f"  Q[3] = {Q[3,3]:.8f} (sigma_wpsi^2)")
