"""
Точка входа: запуск полной симуляции навигации НПА.

Запускает набор экспериментов для главы 4 диплома:

    Сценарий 1 (пункт 4.2): Круговая траектория — базовое сравнение
        EKF, Adaptive EKF и Dead Reckoning. Показывает, что оба
        фильтра значительно точнее наивного счисления пути.

    Сценарий 2 (пункт 4.3): Прямолинейная траектория — проверка
        на простейшем маршруте. Подтверждает, что алгоритмы работают
        не только на окружности.

    Сценарий 3 (пункт 4.3): Траектория «восьмёрка» — сложный маршрут
        с частой сменой курса. Проверяет устойчивость при интенсивном
        маневрировании.

    Сценарий 4 (пункт 4.4): Круговая траектория со сбоями датчиков —
        DVL отключается на 30 с (100–130 с), компас — на 10 с (150–160 с).
        Показывает преимущество адаптивного EKF при нестационарных условиях.

    Сценарий 5 (пункт 4.5): Dead Reckoning на полной длительности —
        ошибка растёт неограниченно (десятки/сотни метров), что наглядно
        обосновывает необходимость коррекции по внешним датчикам.

По итогам всех сценариев выводится сводная таблица метрик (пункт 4.6).

Использование:
    python main.py
"""

import logging
import os
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np

from auv_params import AUVParams
from config import SimulationConfig, SensorConfig, FilterConfig, TrajectoryConfig
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
from simulation.runner import SimulationRunner, SimulationResult
from simulation.sensor_failure import (
    SensorFailureSimulator,
    combined_failure_scenario,
)
from evaluation.metrics import (
    evaluate_simulation,
    print_metrics_table,
    MetricsReport,
)
from evaluation.robustness import (
    analyze_robustness,
    print_robustness_report,
    generate_robustness_text,
)
from visualization.plotter import NavigationPlotter


def setup_logging() -> None:
    """Настройка логирования."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


@dataclass
class ScenarioResult:
    """Результат одного экспериментального сценария.

    Простыми словами: хранит всё, что нужно для одной строки
    в сводной таблице — имя сценария, результат симуляции,
    метрики и историю адаптации R.

    Attributes:
        name: Название сценария (для таблицы и графиков).
        sim_result: Полный результат симуляции.
        reports: Метрики для каждого фильтра.
        r_history: История адаптации R (только Adaptive EKF).
    """
    name: str
    sim_result: SimulationResult
    reports: dict[str, MetricsReport]
    r_history: list[dict] = field(default_factory=list)


def create_sensors(sensor_cfg: SensorConfig) -> tuple[INS, list]:
    """Создание набора датчиков из конфигурации.

    Простыми словами: создаёт 4 виртуальных датчика
    (ИНС, DVL, компас, USBL) с параметрами шума из конфигурации.

    Args:
        sensor_cfg: Параметры шумов датчиков.

    Returns:
        Кортеж (ins, [dvl, compass, usbl]).
    """
    ins = INS(
        gyro_noise_std=sensor_cfg.ins_gyro_noise_std,
        accel_noise_std=sensor_cfg.ins_accel_noise_std,
        gyro_bias_drift_std=sensor_cfg.ins_gyro_bias_drift_std,
        rate=sensor_cfg.ins_rate,
    )
    dvl = DVL(noise_std=sensor_cfg.dvl_speed_noise_std, rate=sensor_cfg.dvl_rate)
    compass = Compass(
        noise_std=sensor_cfg.compass_heading_noise_std,
        rate=sensor_cfg.compass_rate,
    )
    usbl = USBL(
        noise_std=sensor_cfg.usbl_position_noise_std,
        rate=sensor_cfg.usbl_rate,
    )
    return ins, [dvl, compass, usbl]


def create_filters(
    filter_cfg: FilterConfig,
    motion_model: Kinematic2DModel,
    include_dr: bool = True,
) -> dict:
    """Создание набора фильтров для сравнения.

    Простыми словами: создаёт 2 или 3 навигационных фильтра —
    классический EKF, адаптивный EKF и (опционально) Dead Reckoning.
    Каждый фильтр начинает с одного и того же начального состояния.

    Args:
        filter_cfg: Параметры фильтров (Q, P0, начальное состояние).
        motion_model: Модель движения.
        include_dr: Включить Dead Reckoning в сравнение.

    Returns:
        Словарь {имя_фильтра: экземпляр}.
    """
    P0 = np.diag(filter_cfg.initial_covariance_diag)
    Q = np.diag(filter_cfg.process_noise_diag)

    filters = {
        "EKF": EKF(
            motion_model=motion_model,
            initial_state=filter_cfg.initial_state.copy(),
            initial_covariance=P0.copy(),
            process_noise=Q.copy(),
        ),
        "Adaptive EKF": AdaptiveEKF(
            motion_model=motion_model,
            initial_state=filter_cfg.initial_state.copy(),
            initial_covariance=P0.copy(),
            process_noise=Q.copy(),
            innovation_window_size=filter_cfg.innovation_window_size,
            adaptation_rate=filter_cfg.adaptation_rate,
            r_min_scale=filter_cfg.r_min_scale,
            r_max_scale=filter_cfg.r_max_scale,
        ),
    }

    if include_dr:
        filters["Dead Reckoning"] = DeadReckoning(
            motion_model=motion_model,
            initial_state=filter_cfg.initial_state.copy(),
            initial_covariance=P0.copy(),
            process_noise=Q.copy(),
        )

    return filters


def run_scenario(
    name: str,
    traj_gen: TrajectoryGenerator,
    traj_type: str,
    traj_kwargs: dict,
    data_gen: DataGenerator,
    filter_cfg: FilterConfig,
    motion_model: Kinematic2DModel,
    include_dr: bool = True,
    failure_simulator: SensorFailureSimulator | None = None,
    logger_inst: logging.Logger | None = None,
) -> ScenarioResult:
    """Запуск одного экспериментального сценария.

    Простыми словами: генерирует траекторию → создаёт данные
    датчиков → (опционально) имитирует сбои → прогоняет фильтры →
    считает метрики. Возвращает всё в одном объекте ScenarioResult.

    Args:
        name: Название сценария (для таблицы).
        traj_gen: Генератор траекторий.
        traj_type: Тип траектории ('circle', 'straight', 'eight').
        traj_kwargs: Параметры траектории (radius, speed и т.д.).
        data_gen: Генератор синтетических данных.
        filter_cfg: Параметры фильтров.
        motion_model: Модель движения.
        include_dr: Включить Dead Reckoning.
        failure_simulator: Симулятор сбоев (None = без сбоев).
        logger_inst: Логгер.

    Returns:
        ScenarioResult с результатами и метриками.
    """
    log = logger_inst or logging.getLogger(__name__)

    log.info("=" * 50)
    log.info("Сценарий: %s", name)
    log.info("=" * 50)

    # 1. Генерация траектории
    trajectory = traj_gen.generate(traj_type, **traj_kwargs)
    log.info("Траектория '%s': %d шагов", traj_type, len(trajectory.timestamps))

    # 2. Генерация данных датчиков
    sim_data = data_gen.generate(trajectory)

    for sensor_name, measurements in sim_data.sensor_measurements.items():
        log.info("  %s: %d измерений", sensor_name, len(measurements))

    # 3. Имитация сбоев (если задана)
    if failure_simulator is not None:
        log.info("Применение сбоев датчиков...")
        sim_data = failure_simulator.apply(sim_data)
        for desc in failure_simulator.descriptions:
            log.info("  %s", desc)
        for sensor_name, measurements in sim_data.sensor_measurements.items():
            log.info("  %s после сбоя: %d измерений", sensor_name, len(measurements))

    # 4. Создание фильтров
    filters = create_filters(filter_cfg, motion_model, include_dr)

    # 5. Прогон симуляции
    runner = SimulationRunner(filters)
    result = runner.run(sim_data)

    # 6. Метрики
    reports = evaluate_simulation(result)

    # 7. История адаптации R
    r_history = []
    if "Adaptive EKF" in filters:
        adaptive = filters["Adaptive EKF"]
        r_history = adaptive.get_r_history()

    return ScenarioResult(
        name=name,
        sim_result=result,
        reports=reports,
        r_history=r_history,
    )


def print_summary_table(scenarios: list[ScenarioResult]) -> None:
    """Сводная таблица метрик по всем сценариям (пункт 4.6).

    Простыми словами: выводит в консоль одну большую таблицу,
    в которой для каждого сценария показаны RMSE позиции,
    RMSE скорости и максимальная ошибка для каждого фильтра.
    Эта таблица — основа для анализа в главе 4 диплома.

    Args:
        scenarios: Список результатов сценариев.
    """
    print("\n" + "=" * 90)
    print("СВОДНАЯ ТАБЛИЦА МЕТРИК (все сценарии)")
    print("=" * 90)

    # Собираем имена фильтров
    all_filter_names = set()
    for sc in scenarios:
        all_filter_names.update(sc.reports.keys())
    filter_names = sorted(all_filter_names)

    # Заголовок
    header = f"{'Сценарий':<30}"
    for fn in filter_names:
        header += f"  {fn:>16}"
    print(header)
    print("-" * 90)

    # RMSE позиции
    print(f"\n{'RMSE позиции (м)':^90}")
    print("-" * 90)
    for sc in scenarios:
        row = f"{sc.name:<30}"
        for fn in filter_names:
            if fn in sc.reports:
                row += f"  {sc.reports[fn].rmse_position:>16.4f}"
            else:
                row += f"  {'—':>16}"
        print(row)

    # RMSE скорости
    print(f"\n{'RMSE скорости (м/с)':^90}")
    print("-" * 90)
    for sc in scenarios:
        row = f"{sc.name:<30}"
        for fn in filter_names:
            if fn in sc.reports:
                row += f"  {sc.reports[fn].rmse_speed:>16.4f}"
            else:
                row += f"  {'—':>16}"
        print(row)

    # Max ошибка позиции
    print(f"\n{'Max ошибка позиции (м)':^90}")
    print("-" * 90)
    for sc in scenarios:
        row = f"{sc.name:<30}"
        for fn in filter_names:
            if fn in sc.reports:
                row += f"  {sc.reports[fn].max_error_position:>16.4f}"
            else:
                row += f"  {'—':>16}"
        print(row)

    print("\n" + "=" * 90)


def _log_r_extremes(
    r_history: list[dict],
    scenario_name: str,
    logger_inst: logging.Logger,
) -> None:
    """Вывод экстремумов (min/max) адаптированной матрицы R в лог.

    Для каждой размерности измерения (1D — DVL/Compass, 2D — USBL)
    находит минимальное и максимальное значение каждого диагонального
    элемента R за весь прогон. Это позволяет увидеть диапазон,
    в котором «гуляла» адаптация, и использовать конкретные числа
    в тексте диплома.

    Args:
        r_history: История адаптации R из AdaptiveEKF.
        scenario_name: Имя сценария (для подписи в логе).
        logger_inst: Логгер.
    """
    dim_sensor_names = {1: "DVL/Compass (1D)", 2: "USBL (2D)"}
    component_names_by_dim = {
        1: ["v или ψ"],
        2: ["x", "y"],
    }

    dims = sorted(set(e["measurement_dim"] for e in r_history))

    logger_inst.info("=" * 60)
    logger_inst.info("Экстремумы адаптации R — сценарий: %s", scenario_name)
    logger_inst.info("=" * 60)

    for dim in dims:
        entries = [e for e in r_history if e["measurement_dim"] == dim]
        r_diags = np.array([e["R_diag"] for e in entries])
        n_components = r_diags.shape[1]
        sensor_label = dim_sensor_names.get(dim, f"dim={dim}")
        comp_names = component_names_by_dim.get(dim, [str(j) for j in range(n_components)])

        logger_inst.info("  --- %s (%d записей) ---", sensor_label, len(entries))

        for j in range(n_components):
            col = r_diags[:, j]
            r_min = col.min()
            r_max = col.max()
            r_mean = col.mean()
            r_first = col[0]
            r_last = col[-1]
            idx_min = int(col.argmin())
            idx_max = int(col.argmax())

            comp = comp_names[j] if j < len(comp_names) else str(j)
            logger_inst.info(
                "    R[%s,%s]:  начальное=%.6f  мин=%.6f (шаг %d)  "
                "макс=%.6f (шаг %d)  финальное=%.6f  среднее=%.6f",
                comp, comp,
                r_first, r_min, idx_min, r_max, idx_max, r_last, r_mean,
            )

    logger_inst.info("=" * 60)


def main() -> None:
    """Главная функция: полный набор экспериментов для главы 4.

    Простыми словами: последовательно запускает 4 сценария,
    строит графики для каждого, и в конце выводит сводную таблицу.
    """
    setup_logging()
    logger = logging.getLogger(__name__)

    # --- Параметры НПА и среды (раздел 2.1) ---
    auv = AUVParams()
    logger.info("\n%s", auv.summary())

    # --- Конфигурация ---
    sim_cfg = SimulationConfig(dt=auv.dt)
    sensor_cfg = SensorConfig(dvl_speed_noise_std=auv.dvl_total_noise_std())
    filter_cfg = FilterConfig.from_auv_params(auv)
    traj_cfg = TrajectoryConfig()

    np.random.seed(sim_cfg.seed)

    logger.info("Δt = %.3f с, длительность = %.0f с", sim_cfg.dt, sim_cfg.duration)
    logger.info("Q_diag = %s", filter_cfg.process_noise_diag)
    logger.info("P0_diag = %s", filter_cfg.initial_covariance_diag)
    logger.info("σ_DVL = %.4f м/с", sensor_cfg.dvl_speed_noise_std)

    # --- Общие компоненты ---
    motion_model = Kinematic2DModel()
    traj_gen = TrajectoryGenerator(dt=sim_cfg.dt, duration=sim_cfg.duration)
    ins, sensors_list = create_sensors(sensor_cfg)
    data_gen = DataGenerator(ins=ins, sensors=sensors_list, seed=sim_cfg.seed)

    plotter = NavigationPlotter(output_dir="output")
    scenarios: list[ScenarioResult] = []

    # =========================================================
    # Сценарий 1: Круговая траектория (пункт 4.2)
    # =========================================================
    sc1 = run_scenario(
        name="Круг (базовый)",
        traj_gen=traj_gen,
        traj_type="circle",
        traj_kwargs={"radius": traj_cfg.radius, "speed": traj_cfg.speed},
        data_gen=data_gen,
        filter_cfg=filter_cfg,
        motion_model=motion_model,
        include_dr=True,
        logger_inst=logger,
    )
    scenarios.append(sc1)
    print_metrics_table(sc1.reports)

    # =========================================================
    # Сценарий 2: Прямолинейная траектория (пункт 4.3)
    # =========================================================
    sc2 = run_scenario(
        name="Прямая",
        traj_gen=traj_gen,
        traj_type="straight",
        traj_kwargs={"speed": traj_cfg.speed, "heading": 0.0},
        data_gen=data_gen,
        filter_cfg=filter_cfg,
        motion_model=motion_model,
        include_dr=True,
        logger_inst=logger,
    )
    scenarios.append(sc2)
    print_metrics_table(sc2.reports)

    # =========================================================
    # Сценарий 3: Восьмёрка (пункт 4.3)
    # =========================================================
    sc3 = run_scenario(
        name="Восьмёрка",
        traj_gen=traj_gen,
        traj_type="eight",
        traj_kwargs={"radius": traj_cfg.radius, "speed": traj_cfg.speed},
        data_gen=data_gen,
        filter_cfg=filter_cfg,
        motion_model=motion_model,
        include_dr=True,
        logger_inst=logger,
    )
    scenarios.append(sc3)
    print_metrics_table(sc3.reports)

    # --- Экстремумы адаптации R для восьмёрки (для диплома) ---
    if sc3.r_history:
        _log_r_extremes(sc3.r_history, "Восьмёрка", logger)

    # =========================================================
    # Сценарий 4: Круг со сбоями датчиков (пункт 4.4)
    # =========================================================
    failure_sim = SensorFailureSimulator(combined_failure_scenario())
    sc4 = run_scenario(
        name="Круг + сбои датчиков",
        traj_gen=traj_gen,
        traj_type="circle",
        traj_kwargs={"radius": traj_cfg.radius, "speed": traj_cfg.speed},
        data_gen=data_gen,
        filter_cfg=filter_cfg,
        motion_model=motion_model,
        include_dr=False,  # DR и так плох, нет смысла показывать со сбоями
        failure_simulator=failure_sim,
        logger_inst=logger,
    )
    scenarios.append(sc4)
    print_metrics_table(sc4.reports)

    # --- Количественный анализ робастности (раздел 4.4) ---
    failure_scenarios = combined_failure_scenario()
    robustness_reports = analyze_robustness(sc4.sim_result, failure_scenarios)
    print_robustness_report(robustness_reports)

    diploma_text = generate_robustness_text(robustness_reports)
    logger.info("Текст для диплома (раздел 4.4):\n%s", diploma_text)

    # Сохраняем текст в файл для удобства
    os.makedirs("output", exist_ok=True)
    with open("output/robustness_analysis.txt", "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("Количественный анализ робастности (раздел 4.4)\n")
        f.write("=" * 60 + "\n\n")
        f.write(diploma_text)
        f.write("\n")
    logger.info("Текст робастного анализа сохранён в output/robustness_analysis.txt")

    # =========================================================
    # Сводная таблица (пункт 4.6)
    # =========================================================
    print_summary_table(scenarios)

    # =========================================================
    # Визуализация
    # =========================================================
    logger.info("Построение графиков...")
    os.makedirs("output", exist_ok=True)

    # Графики для каждого сценария
    for i, sc in enumerate(scenarios, 1):
        prefix = f"scenario{i}"
        fig_traj = plotter.plot_trajectory(sc.sim_result)
        fig_traj.suptitle(f"Сценарий: {sc.name}", fontsize=14, y=1.02)
        fig_traj.savefig(
            f"output/{prefix}_trajectory.png",
            dpi=plotter._dpi,
            bbox_inches="tight",
        )

        fig_err = plotter.plot_position_error_norm(sc.sim_result)
        fig_err.suptitle(f"Ошибка позиции: {sc.name}", fontsize=14, y=1.02)
        fig_err.savefig(
            f"output/{prefix}_error_norm.png",
            dpi=plotter._dpi,
            bbox_inches="tight",
        )

        # График адаптации R (только для сценариев с AEKF)
        if sc.r_history:
            fig_r = plotter.plot_r_adaptation(sc.r_history)
            fig_r.suptitle(
                f"Адаптация R: {sc.name}", fontsize=14, y=1.02
            )
            fig_r.savefig(
                f"output/{prefix}_r_adaptation.png",
                dpi=plotter._dpi,
                bbox_inches="tight",
            )

    logger.info("Все графики сохранены в output/")
    logger.info("Готово!")

    plt.show()


if __name__ == "__main__":
    main()
