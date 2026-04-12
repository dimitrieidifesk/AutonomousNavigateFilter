"""
Количественный анализ робастности фильтров при сбоях датчиков (раздел 4.4).

Этот модуль вычисляет метрики, которые показывают, насколько хорошо
фильтр справляется с отказом датчика:

    1. Максимальная ошибка во время сбоя — показывает «худший момент»,
       когда аппарат навигируется без одного из датчиков.

    2. Средняя ошибка во время сбоя — характеризует качество навигации
       в целом на протяжении всего отказа.

    3. Время восстановления — сколько секунд после окончания сбоя
       нужно фильтру, чтобы вернуться к нормальной точности.
       «Нормальная точность» определяется как медиана ошибки
       позиции до начала сбоя (в стабильном режиме).

    4. Ошибка на момент начала сбоя — позволяет оценить, с какого
       уровня ошибка начинает расти.

    5. Ошибка на момент окончания сбоя — максимальный «ущерб»,
       нанесённый отказом.

Простыми словами:
    Мы берём временной ряд ошибок позиции (||x_true - x_est||),
    «вырезаем» из него участок, соответствующий окну сбоя,
    и считаем на этом участке max/mean. Затем смотрим, как быстро
    ошибка после сбоя возвращается к «досбойному» уровню.

    Это даёт конкретные числа для фразы в дипломе:
    «Во время отключения DVL (100–130 с) ошибка EKF достигла 2.1 м,
    а адаптивного EKF — 1.8 м; после восстановления адаптивный
    EKF вернулся к норме на 5 с быстрее.»

Использование:
    from evaluation.robustness import analyze_robustness, print_robustness_report

    failure_scenarios = combined_failure_scenario()
    report = analyze_robustness(sim_result, failure_scenarios)
    print_robustness_report(report)

Зависимости:
    - SimulationResult из simulation.runner
    - FailureScenario из simulation.sensor_failure
    - position_error_norm из evaluation.metrics
"""

from dataclasses import dataclass

import numpy as np

from evaluation.metrics import position_error_norm
from simulation.runner import SimulationResult
from simulation.sensor_failure import FailureScenario


# ───────────────────────── Датаклассы ─────────────────────────


@dataclass
class FilterFailureMetrics:
    """Метрики робастности одного фильтра для одного сбоя.

    Простыми словами: ответ на вопрос «насколько плохо пришлось
    этому фильтру во время данного сбоя и как быстро он оправился».

    Attributes:
        filter_name: Имя фильтра (EKF / Adaptive EKF).
        max_error_during: Максимальная ошибка позиции (м) внутри
            окна сбоя [start_time, end_time].
        mean_error_during: Средняя ошибка позиции (м) внутри
            окна сбоя.
        error_at_start: Ошибка позиции (м) в момент начала сбоя.
        error_at_end: Ошибка позиции (м) в момент окончания сбоя.
        baseline_error: Медиана ошибки до сбоя (м) — «нормальный»
            уровень точности фильтра.
        recovery_time: Время (с) от окончания сбоя до момента,
            когда ошибка вернулась к baseline_error.
            NaN если не восстановился до конца симуляции.
    """

    filter_name: str
    max_error_during: float
    mean_error_during: float
    error_at_start: float
    error_at_end: float
    baseline_error: float
    recovery_time: float


@dataclass
class FailureRobustnessReport:
    """Полный отчёт о робастности для одного сценария сбоя.

    Attributes:
        failure: Описание сбоя (датчик, временное окно).
        filter_metrics: Словарь {имя_фильтра: FilterFailureMetrics}.
    """

    failure: FailureScenario
    filter_metrics: dict[str, FilterFailureMetrics]


# ───────────────────────── Анализ ─────────────────────────


def _compute_filter_failure_metrics(
    filter_name: str,
    pos_errors: np.ndarray,
    timestamps: np.ndarray,
    failure: FailureScenario,
) -> FilterFailureMetrics:
    """Вычислить метрики робастности для одного фильтра и одного сбоя.

    Алгоритм:
        1. Определяем baseline — медиана ошибки до начала сбоя.
           Используем медиану (а не среднее), чтобы исключить
           влияние начального переходного процесса фильтра
           (первые секунды ошибка может быть высокой).

        2. Вырезаем ошибки в окне [start_time, end_time] и
           считаем max / mean.

        3. Ищем recovery_time: первый момент после end_time,
           когда ошибка опустилась ≤ baseline. Если ошибка
           колеблется, берём первый такой момент (допущение:
           после первого касания baseline фильтр считается
           «восстановившимся»).

    Args:
        filter_name: Имя фильтра.
        pos_errors: Нормы ошибок позиции (N,).
        timestamps: Временные метки (N,).
        failure: Сценарий сбоя.

    Returns:
        FilterFailureMetrics с заполненными полями.
    """
    dt = timestamps[1] - timestamps[0] if len(timestamps) > 1 else 0.1

    # --- Индексы ---
    # Стабильная зона до сбоя: от 10 с (пропускаем переходный процесс) до start_time
    stable_start = 10.0
    pre_mask = (timestamps >= stable_start) & (timestamps < failure.start_time)
    during_mask = (timestamps >= failure.start_time) & (timestamps <= failure.end_time)
    post_mask = timestamps > failure.end_time

    # --- Baseline (до сбоя) ---
    if np.any(pre_mask):
        baseline_error = float(np.median(pos_errors[pre_mask]))
    else:
        # Если сбой начинается очень рано — baseline = первая ошибка
        baseline_error = float(pos_errors[0])

    # --- Метрики в окне сбоя ---
    if np.any(during_mask):
        errors_during = pos_errors[during_mask]
        max_error_during = float(np.max(errors_during))
        mean_error_during = float(np.mean(errors_during))
        error_at_start = float(errors_during[0])
        error_at_end = float(errors_during[-1])
    else:
        max_error_during = 0.0
        mean_error_during = 0.0
        error_at_start = 0.0
        error_at_end = 0.0

    # --- Время восстановления ---
    # Ищем первый момент после конца сбоя, когда ошибка ≤ baseline
    # Используем порог 1.5 × baseline чтобы учесть стохастические
    # колебания (фильтр может «осциллировать» около baseline).
    recovery_threshold = 1.5 * baseline_error
    recovery_time = float("nan")

    if np.any(post_mask):
        post_timestamps = timestamps[post_mask]
        post_errors = pos_errors[post_mask]
        recovered_indices = np.where(post_errors <= recovery_threshold)[0]
        if len(recovered_indices) > 0:
            first_recovered_time = post_timestamps[recovered_indices[0]]
            recovery_time = first_recovered_time - failure.end_time

    return FilterFailureMetrics(
        filter_name=filter_name,
        max_error_during=max_error_during,
        mean_error_during=mean_error_during,
        error_at_start=error_at_start,
        error_at_end=error_at_end,
        baseline_error=baseline_error,
        recovery_time=recovery_time,
    )


def analyze_robustness(
    sim_result: SimulationResult,
    failure_scenarios: list[FailureScenario],
) -> list[FailureRobustnessReport]:
    """Количественный анализ робастности всех фильтров при сбоях.

    Простыми словами: для каждого сбоя и каждого фильтра считаем,
    насколько выросла ошибка и как быстро она вернулась к норме.

    Args:
        sim_result: Результат симуляции (содержит true_states и
            оценки каждого фильтра).
        failure_scenarios: Список сценариев сбоев (из sensor_failure.py).

    Returns:
        Список FailureRobustnessReport — по одному на каждый сбой.
    """
    timestamps = sim_result.timestamps
    true_xy = sim_result.true_states[:, :2]

    reports: list[FailureRobustnessReport] = []

    for failure in failure_scenarios:
        filter_metrics: dict[str, FilterFailureMetrics] = {}

        for name, fr in sim_result.filter_results.items():
            est_xy = fr.estimated_states[:, :2]
            pos_errors = position_error_norm(est_xy, true_xy)

            metrics = _compute_filter_failure_metrics(
                filter_name=name,
                pos_errors=pos_errors,
                timestamps=timestamps,
                failure=failure,
            )
            filter_metrics[name] = metrics

        reports.append(
            FailureRobustnessReport(failure=failure, filter_metrics=filter_metrics)
        )

    return reports


# ───────────────────────── Вывод ─────────────────────────


def print_robustness_report(reports: list[FailureRobustnessReport]) -> None:
    """Вывод отчёта о робастности в консоль.

    Простыми словами: печатает таблицу, из которой сразу видно,
    какой фильтр лучше справился с каждым сбоем.

    Args:
        reports: Список отчётов (от analyze_robustness).
    """
    print("\n" + "=" * 80)
    print("АНАЛИЗ РОБАСТНОСТИ ПРИ СБОЯХ ДАТЧИКОВ (раздел 4.4)")
    print("=" * 80)

    for report in reports:
        f = report.failure
        print(f"\n--- {f.description} ---")
        print(f"    Датчик: {f.sensor_name}, "
              f"окно: [{f.start_time:.0f} – {f.end_time:.0f}] с")

        filter_names = sorted(report.filter_metrics.keys())

        # Заголовок
        header = f"  {'Метрика':<30}"
        for fn in filter_names:
            header += f"  {fn:>16}"
        print(header)
        print("  " + "-" * (30 + 18 * len(filter_names)))

        # Строки метрик
        rows = [
            ("Baseline ошибка (м)", "baseline_error"),
            ("Ошибка в начале сбоя (м)", "error_at_start"),
            ("Max ошибка во время сбоя (м)", "max_error_during"),
            ("Средняя ошибка в сбое (м)", "mean_error_during"),
            ("Ошибка в конце сбоя (м)", "error_at_end"),
            ("Время восстановления (с)", "recovery_time"),
        ]

        for label, attr in rows:
            line = f"  {label:<30}"
            for fn in filter_names:
                value = getattr(report.filter_metrics[fn], attr)
                if np.isnan(value):
                    line += f"  {'не восст.':>16}"
                else:
                    line += f"  {value:>16.4f}"
            print(line)

    print("\n" + "=" * 80)


def generate_robustness_text(reports: list[FailureRobustnessReport]) -> str:
    """Генерация текста для диплома (раздел 4.4).

    Простыми словами: автоматически формирует абзацы текста,
    которые можно вставить в диплом. Например:

        «Во время отключения DVL (100–130 с) максимальная ошибка
        EKF достигла 2.10 м, а адаптивного EKF — 1.80 м.
        После восстановления датчика адаптивный EKF вернулся
        к нормальной точности за 3.5 с, тогда как классическому
        EKF потребовалось 8.2 с.»

    Args:
        reports: Список отчётов (от analyze_robustness).

    Returns:
        Готовый текст для вставки в диплом (str).
    """
    paragraphs: list[str] = []

    for report in reports:
        f = report.failure
        metrics = report.filter_metrics

        filter_names = sorted(metrics.keys())
        if len(filter_names) < 2:
            # Для одного фильтра — просто описание
            fn = filter_names[0]
            m = metrics[fn]
            text = (
                f"Во время отключения {f.sensor_name} "
                f"({f.start_time:.0f}–{f.end_time:.0f} с) "
                f"максимальная ошибка позиции {fn} достигла "
                f"{m.max_error_during:.2f} м "
                f"(средняя: {m.mean_error_during:.2f} м)."
            )
            if not np.isnan(m.recovery_time):
                text += (
                    f" После восстановления датчика фильтр вернулся "
                    f"к нормальной точности за {m.recovery_time:.1f} с."
                )
            else:
                text += (
                    " Фильтр не восстановился до нормальной точности "
                    "к концу симуляции."
                )
            paragraphs.append(text)
            continue

        # Сравнение двух фильтров
        # Определяем «лучший» по max ошибке во время сбоя
        sorted_by_max = sorted(
            filter_names, key=lambda n: metrics[n].max_error_during
        )
        best_name = sorted_by_max[0]
        worst_name = sorted_by_max[-1]

        parts: list[str] = []

        # Часть 1: максимальные ошибки
        err_parts = []
        for fn in filter_names:
            err_parts.append(
                f"{fn} — {metrics[fn].max_error_during:.2f} м"
            )
        parts.append(
            f"Во время отключения {f.sensor_name} "
            f"({f.start_time:.0f}–{f.end_time:.0f} с) "
            f"максимальная ошибка позиции составила: "
            + ", ".join(err_parts)
            + "."
        )

        # Часть 2: средние ошибки
        mean_parts = []
        for fn in filter_names:
            mean_parts.append(
                f"{fn} — {metrics[fn].mean_error_during:.2f} м"
            )
        parts.append(
            f"Средняя ошибка в окне сбоя: " + ", ".join(mean_parts) + "."
        )

        # Часть 3: восстановление
        recovery_parts = []
        for fn in filter_names:
            rt = metrics[fn].recovery_time
            if np.isnan(rt):
                recovery_parts.append(f"{fn} не восстановился")
            else:
                recovery_parts.append(f"{fn} — {rt:.1f} с")

        parts.append(
            "Время восстановления после окончания сбоя: "
            + ", ".join(recovery_parts)
            + "."
        )

        # Часть 4: вывод о преимуществе
        best_max = metrics[best_name].max_error_during
        worst_max = metrics[worst_name].max_error_during
        if worst_max > 0:
            advantage_pct = (worst_max - best_max) / worst_max * 100
            parts.append(
                f"Таким образом, {best_name} показал максимальную ошибку "
                f"на {advantage_pct:.0f}% ниже, чем {worst_name}."
            )

        # Часть 5: сравнение восстановления
        rt_best = metrics[best_name].recovery_time
        rt_worst = metrics[worst_name].recovery_time
        if not np.isnan(rt_best) and not np.isnan(rt_worst):
            diff = rt_worst - rt_best
            if diff > 0:
                parts.append(
                    f"После восстановления датчика {best_name} вернулся "
                    f"к нормальной точности на {diff:.1f} с быстрее."
                )

        paragraphs.append(" ".join(parts))

    return "\n\n".join(paragraphs)
