"""
Модуль вычисления метрик точности навигации.

Предоставляет функции для оценки качества работы фильтров
путём сравнения оценённых состояний с эталонными (ground truth).

Метрики:
    - RMSE (Root Mean Square Error): Среднеквадратичная ошибка.
    - MAE (Mean Absolute Error): Средняя абсолютная ошибка.
    - Max Error: Максимальное отклонение.
    - Mean Error: Средняя ошибка по времени.
    - Position RMSE: RMSE по координатам (x, y).

Вход:
    estimated (np.ndarray): Массив оценённых состояний (N × m).
    true (np.ndarray): Массив истинных состояний (N × m).

Выход:
    Скалярное значение метрики или словарь метрик.

Рекомендации по тестированию:
    - RMSE(x, x) = 0 (нулевая ошибка).
    - RMSE ≥ MAE для любых данных.
    - Max Error ≥ RMSE для любых данных.
    - Метрики >= 0 для любых данных.
    - Известный сценарий: RMSE([0,0], [3,4]) = 5/√2... проверить вычисление.
"""

from dataclasses import dataclass

import numpy as np

from simulation.runner import SimulationResult


@dataclass
class MetricsReport:
    """Отчёт о метриках для одного фильтра.

    Attributes:
        filter_name: Имя фильтра.
        rmse_x: RMSE по координате X (м).
        rmse_y: RMSE по координате Y (м).
        rmse_position: RMSE позиции (м).
        rmse_speed: RMSE скорости (м/с).
        rmse_heading: RMSE курса (рад).
        mae_x: MAE по X (м).
        mae_y: MAE по Y (м).
        mae_position: MAE позиции (м).
        max_error_x: Максимальная ошибка по X (м).
        max_error_y: Максимальная ошибка по Y (м).
        max_error_position: Максимальная ошибка позиции (м).
        mean_error_position: Средняя ошибка позиции (м).
    """
    filter_name: str
    rmse_x: float
    rmse_y: float
    rmse_position: float
    rmse_speed: float
    rmse_heading: float
    mae_x: float
    mae_y: float
    mae_position: float
    max_error_x: float
    max_error_y: float
    max_error_position: float
    mean_error_position: float


def rmse(estimated: np.ndarray, true: np.ndarray) -> float:
    """Среднеквадратичная ошибка (RMSE).

    RMSE = √(mean((estimated - true)²))

    Args:
        estimated: Оценённые значения (N,) или (N, m).
        true: Истинные значения (N,) или (N, m).

    Returns:
        Скалярное значение RMSE.
    """
    return float(np.sqrt(np.mean((estimated - true) ** 2)))


def mae(estimated: np.ndarray, true: np.ndarray) -> float:
    """Средняя абсолютная ошибка (MAE).

    MAE = mean(|estimated - true|)

    Args:
        estimated: Оценённые значения.
        true: Истинные значения.

    Returns:
        Скалярное значение MAE.
    """
    return float(np.mean(np.abs(estimated - true)))


def max_error(estimated: np.ndarray, true: np.ndarray) -> float:
    """Максимальное отклонение.

    Args:
        estimated: Оценённые значения.
        true: Истинные значения.

    Returns:
        Скалярное значение максимальной ошибки.
    """
    return float(np.max(np.abs(estimated - true)))


def position_error_norm(estimated: np.ndarray, true: np.ndarray) -> np.ndarray:
    """Норма ошибки позиции ||Δx, Δy|| для каждого шага.

    Args:
        estimated: Оценённые координаты (N × 2): [x, y].
        true: Истинные координаты (N × 2): [x, y].

    Returns:
        Массив норм ошибок (N,).
    """
    return np.linalg.norm(estimated - true, axis=1)


def compute_metrics(
    estimated_states: np.ndarray,
    true_states: np.ndarray,
    filter_name: str,
) -> MetricsReport:
    """Вычисление полного набора метрик для одного фильтра.

    Args:
        estimated_states: Оценённые состояния (N × 4).
        true_states: Истинные состояния (N × 4).
        filter_name: Имя фильтра.

    Returns:
        MetricsReport с полным набором метрик.
    """
    # Ошибки по координатам
    error_x = estimated_states[:, 0] - true_states[:, 0]
    error_y = estimated_states[:, 1] - true_states[:, 1]
    pos_errors = position_error_norm(
        estimated_states[:, :2], true_states[:, :2]
    )

    # Ошибка по курсу (с нормализацией)
    heading_error = estimated_states[:, 3] - true_states[:, 3]
    heading_error = (heading_error + np.pi) % (2 * np.pi) - np.pi

    return MetricsReport(
        filter_name=filter_name,
        rmse_x=rmse(estimated_states[:, 0], true_states[:, 0]),
        rmse_y=rmse(estimated_states[:, 1], true_states[:, 1]),
        rmse_position=float(np.sqrt(np.mean(pos_errors ** 2))),
        rmse_speed=rmse(estimated_states[:, 2], true_states[:, 2]),
        rmse_heading=float(np.sqrt(np.mean(heading_error ** 2))),
        mae_x=mae(estimated_states[:, 0], true_states[:, 0]),
        mae_y=mae(estimated_states[:, 1], true_states[:, 1]),
        mae_position=float(np.mean(pos_errors)),
        max_error_x=max_error(estimated_states[:, 0], true_states[:, 0]),
        max_error_y=max_error(estimated_states[:, 1], true_states[:, 1]),
        max_error_position=float(np.max(pos_errors)),
        mean_error_position=float(np.mean(pos_errors)),
    )


def evaluate_simulation(result: SimulationResult) -> dict[str, MetricsReport]:
    """Вычисление метрик для всех фильтров в результате симуляции.

    Args:
        result: Результат симуляции.

    Returns:
        Словарь {имя_фильтра: MetricsReport}.
    """
    reports = {}
    for name, fr in result.filter_results.items():
        reports[name] = compute_metrics(
            fr.estimated_states, result.true_states, name
        )
    return reports


def print_metrics_table(reports: dict[str, MetricsReport]) -> None:
    """Вывод таблицы метрик в консоль.

    Args:
        reports: Словарь отчётов метрик.
    """
    print("\n" + "=" * 70)
    print(f"{'Метрика':<30} ", end="")
    for name in reports:
        print(f"{name:>18} ", end="")
    print()
    print("-" * 70)

    fields = [
        ("RMSE X (м)", "rmse_x"),
        ("RMSE Y (м)", "rmse_y"),
        ("RMSE позиции (м)", "rmse_position"),
        ("RMSE скорости (м/с)", "rmse_speed"),
        ("RMSE курса (рад)", "rmse_heading"),
        ("MAE X (м)", "mae_x"),
        ("MAE Y (м)", "mae_y"),
        ("MAE позиции (м)", "mae_position"),
        ("Max ошибка X (м)", "max_error_x"),
        ("Max ошибка Y (м)", "max_error_y"),
        ("Max ошибка позиции (м)", "max_error_position"),
        ("Средняя ошибка (м)", "mean_error_position"),
    ]

    for label, attr in fields:
        print(f"{label:<30} ", end="")
        for report in reports.values():
            value = getattr(report, attr)
            print(f"{value:>18.6f} ", end="")
        print()

    print("=" * 70)
