"""
Модуль визуализации результатов навигации.

Строит графики для анализа работы фильтров:
    1. Траектории: истинная vs. оценённые (EKF, Adaptive EKF, DR).
    2. Ошибки по X и Y во времени.
    3. Норма ошибки позиции во времени.
    4. Динамика адаптации матрицы R (только для Adaptive EKF).

Простыми словами:
    Этот модуль рисует «картинки» для главы 4 диплома.
    Каждый график показывает один аспект работы фильтров:
    - Насколько точно фильтр «следит» за истинной траекторией.
    - Как ошибка меняется во времени (растёт? стабилизируется?).
    - Как адаптируется матрица R при изменении условий.

Для графика адаптации R (раздел 4.3):
    - Ось X — шаг обновления (номер вызова update()).
    - Ось Y — диагональные элементы R (дисперсия шума).
    - Отдельные линии для каждого типа датчика:
      dim=1 → DVL (скорость) или Compass (курс),
      dim=2 → USBL (координаты x, y).
    - Если R растёт — фильтр «меньше доверяет» датчику.
    - Если R стабильна — шум стационарный, адаптация не нужна.

Рекомендации по тестированию:
    - Проверить, что plot_trajectory() не падает на пустых данных.
    - Проверить, что save_all() создаёт файлы.
    - Визуальная проверка: оценённая траектория «следит» за истинной.
"""

import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from simulation.runner import SimulationResult


class NavigationPlotter:
    """Визуализатор результатов навигации.

    Создаёт набор стандартных графиков для анализа и сравнения фильтров.

    Args:
        figsize: Размер фигуры (ширина, высота) в дюймах.
        dpi: Разрешение графиков.
        output_dir: Директория для сохранения графиков.
    """

    def __init__(
        self,
        figsize: tuple[float, float] = (12, 8),
        dpi: int = 150,
        output_dir: str = "output",
    ):
        self._figsize = figsize
        self._dpi = dpi
        self._output_dir = output_dir

        # Цвета для разных фильтров
        self._colors = {
            "true": "black",
            "EKF": "blue",
            "Adaptive EKF": "red",
            "Dead Reckoning": "green",
        }
        # Стили линий
        self._linestyles = {
            "true": "-",
            "EKF": "--",
            "Adaptive EKF": "-.",
            "Dead Reckoning": ":",
        }

        # Имена датчиков по размерности измерения (для подписей на графике R)
        self._dim_sensor_names = {
            1: "DVL/Compass (1D)",
            2: "USBL (2D)",
        }

    def plot_trajectory(self, result: SimulationResult) -> plt.Figure:
        """График траекторий: истинная vs. оценённые.

        Args:
            result: Результат симуляции.

        Returns:
            Фигура matplotlib.
        """
        fig, ax = plt.subplots(1, 1, figsize=self._figsize, dpi=self._dpi)

        # Истинная траектория
        ax.plot(
            result.true_states[:, 1],  # y (восток)
            result.true_states[:, 0],  # x (север)
            color=self._colors["true"],
            linestyle=self._linestyles["true"],
            linewidth=2,
            label="Истинная траектория",
        )

        # Оценённые траектории
        for name, fr in result.filter_results.items():
            color = self._colors.get(name, "green")
            ls = self._linestyles.get(name, ":")
            ax.plot(
                fr.estimated_states[:, 1],
                fr.estimated_states[:, 0],
                color=color,
                linestyle=ls,
                linewidth=1.5,
                label=f"{name}",
            )

        ax.set_xlabel("Y (Восток), м")
        ax.set_ylabel("X (Север), м")
        ax.set_title("Сравнение траекторий")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal")

        plt.tight_layout()
        return fig

    def plot_position_errors(self, result: SimulationResult) -> plt.Figure:
        """Графики ошибок по координатам X и Y во времени.

        Args:
            result: Результат симуляции.

        Returns:
            Фигура matplotlib.
        """
        fig, axes = plt.subplots(2, 1, figsize=self._figsize, dpi=self._dpi, sharex=True)
        t = result.timestamps

        for name, fr in result.filter_results.items():
            error_x = fr.estimated_states[:, 0] - result.true_states[:, 0]
            error_y = fr.estimated_states[:, 1] - result.true_states[:, 1]

            color = self._colors.get(name, "green")

            axes[0].plot(t, error_x, color=color, linewidth=1, label=name)
            axes[1].plot(t, error_y, color=color, linewidth=1, label=name)

        axes[0].set_ylabel("Ошибка X (Север), м")
        axes[0].set_title("Ошибки позиционирования")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].set_ylabel("Ошибка Y (Восток), м")
        axes[1].set_xlabel("Время, с")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_position_error_norm(self, result: SimulationResult) -> plt.Figure:
        """График нормы ошибки позиции ||Δx, Δy|| во времени.

        Args:
            result: Результат симуляции.

        Returns:
            Фигура matplotlib.
        """
        fig, ax = plt.subplots(1, 1, figsize=self._figsize, dpi=self._dpi)
        t = result.timestamps

        for name, fr in result.filter_results.items():
            error = fr.estimated_states[:, :2] - result.true_states[:, :2]
            error_norm = np.linalg.norm(error, axis=1)

            color = self._colors.get(name, "green")
            ax.plot(t, error_norm, color=color, linewidth=1, label=name)

        ax.set_xlabel("Время, с")
        ax.set_ylabel("Норма ошибки позиции, м")
        ax.set_title("Норма ошибки позиционирования")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_r_adaptation(self, r_history: list[dict]) -> plt.Figure:
        """График динамики адаптации матрицы R (раздел 4.3).

        Простыми словами:
            Этот график показывает, как адаптивный EKF изменяет
            свою «оценку шума» для каждого датчика. Ось X — номер
            обновления, ось Y — значение дисперсии шума R[i,i].

            Для 1D-датчиков (DVL — скорость, Compass — курс)
            рисуется одна линия на каждый тип.
            Для 2D-датчика (USBL — координаты x, y) — две линии
            (R[0,0] для x и R[1,1] для y).

            Если линия R растёт — фильтр стал «меньше доверять»
            датчику (например, при увеличении реального шума).
            Если R стабильна — шум стационарный.

        Args:
            r_history: История адаптации R из AdaptiveEKF.get_r_history().

        Returns:
            Фигура matplotlib.
        """
        fig, ax = plt.subplots(1, 1, figsize=self._figsize, dpi=self._dpi)

        if not r_history:
            ax.text(0.5, 0.5, "Нет данных адаптации R",
                    ha="center", va="center", transform=ax.transAxes)
            return fig

        # Группируем по размерности измерения
        dims = set(entry["measurement_dim"] for entry in r_history)

        # Цвета для компонент
        component_colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

        for dim in sorted(dims):
            entries = [e for e in r_history if e["measurement_dim"] == dim]
            steps = range(len(entries))
            r_diags = np.array([e["R_diag"] for e in entries])

            sensor_label = self._dim_sensor_names.get(dim, f"dim={dim}")

            for j in range(r_diags.shape[1]):
                color_idx = (dim - 1) * 2 + j
                color = component_colors[color_idx % len(component_colors)]

                if r_diags.shape[1] == 1:
                    label = f"R ({sensor_label})"
                else:
                    component_names = ["x", "y", "v", "ψ"]
                    comp = component_names[j] if j < len(component_names) else str(j)
                    label = f"R[{comp},{comp}] ({sensor_label})"

                ax.plot(
                    steps,
                    r_diags[:, j],
                    linewidth=1,
                    color=color,
                    label=label,
                )

        ax.set_xlabel("Шаг обновления (номер вызова update())")
        ax.set_ylabel("Диагональные элементы R (дисперсия шума, м² или рад²)")
        ax.set_title(
            "Адаптация матрицы шума измерений R\n"
            "(рост R → фильтр меньше доверяет датчику)"
        )
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def save_all(
        self,
        result: SimulationResult,
        r_history: Optional[list[dict]] = None,
    ) -> list[str]:
        """Сохранить все графики в файлы.

        Args:
            result: Результат симуляции.
            r_history: История адаптации R (опционально).

        Returns:
            Список путей к сохранённым файлам.
        """
        os.makedirs(self._output_dir, exist_ok=True)
        saved_files = []

        plots = [
            ("trajectory.png", self.plot_trajectory(result)),
            ("position_errors.png", self.plot_position_errors(result)),
            ("error_norm.png", self.plot_position_error_norm(result)),
        ]

        if r_history:
            plots.append(("r_adaptation.png", self.plot_r_adaptation(r_history)))

        for filename, fig in plots:
            path = os.path.join(self._output_dir, filename)
            fig.savefig(path, dpi=self._dpi, bbox_inches="tight")
            plt.close(fig)
            saved_files.append(path)

        return saved_files

    def show_all(
        self,
        result: SimulationResult,
        r_history: Optional[list[dict]] = None,
    ) -> None:
        """Показать все графики интерактивно.

        Args:
            result: Результат симуляции.
            r_history: История адаптации R (опционально).
        """
        self.plot_trajectory(result)
        self.plot_position_errors(result)
        self.plot_position_error_norm(result)

        if r_history:
            self.plot_r_adaptation(r_history)

        plt.show()
