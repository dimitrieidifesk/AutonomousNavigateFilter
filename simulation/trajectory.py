"""
Генерация эталонных траекторий НПА.

Предоставляет набор аналитических траекторий для тестирования фильтров.
Каждая траектория возвращает истинное состояние [x, y, v, ψ]
и управляющие сигналы [ω, a] в каждый момент времени.

Поддерживаемые траектории:
    - circle: Движение по окружности.
    - eight: Движение по фигуре «восьмёрка» (лемниската).
    - sine: Синусоидальная траектория.
    - straight: Прямолинейное движение (для калибровки).

Вход:
    trajectory_type (str): Тип траектории.
    duration (float): Длительность (с).
    dt (float): Шаг времени (с).
    Параметры траектории (radius, speed, amplitude, frequency).

Выход:
    TrajectoryData: Временные ряды состояний и управлений.

Рекомендации по тестированию:
    - Проверить, что straight-траектория не меняет y и ψ.
    - Проверить, что circle-траектория замыкается (конец ≈ начало).
    - Проверить, что скорость v постоянна для circle.
    - Проверить, что длина массивов = duration / dt + 1.
"""

from dataclasses import dataclass

import numpy as np

from models.base_motion_model import BaseMotionModel


@dataclass
class TrajectoryData:
    """Данные эталонной траектории.

    Attributes:
        timestamps: Массив временных меток (с). Размер (N,).
        states: Массив истинных состояний (N × 4): [x, y, v, ψ].
        controls: Массив управляющих сигналов (N × 2): [ω, a].
    """
    timestamps: np.ndarray   # (N,)
    states: np.ndarray       # (N, 4)
    controls: np.ndarray     # (N, 2)


class TrajectoryGenerator:
    """Генератор эталонных траекторий.

    Создаёт аналитические траектории НПА с известными состояниями
    и управляющими сигналами для тестирования фильтров.

    Args:
        dt: Шаг времени (с).
        duration: Длительность траектории (с).
    """

    def __init__(self, dt: float = 0.02, duration: float = 300.0):
        self._dt = dt
        self._duration = duration

    def generate(self, trajectory_type: str, **kwargs) -> TrajectoryData:
        """Генерация траектории заданного типа.

        Args:
            trajectory_type: Тип ('circle', 'eight', 'sine', 'straight').
            **kwargs: Параметры конкретной траектории.

        Returns:
            TrajectoryData с истинными состояниями и управлениями.

        Raises:
            ValueError: Неизвестный тип траектории.
        """
        generators = {
            "circle": self._generate_circle,
            "eight": self._generate_eight,
            "sine": self._generate_sine,
            "straight": self._generate_straight,
        }

        if trajectory_type not in generators:
            raise ValueError(
                f"Неизвестный тип траектории: '{trajectory_type}'. "
                f"Доступные: {list(generators.keys())}"
            )

        return generators[trajectory_type](**kwargs)

    def _generate_circle(
        self,
        radius: float = 50.0,
        speed: float = 1.5,
        **_kwargs,
    ) -> TrajectoryData:
        """Движение по окружности.

        Аппарат движется с постоянной скоростью по окружности заданного радиуса.

        Args:
            radius: Радиус окружности (м).
            speed: Скорость движения (м/с).

        Returns:
            TrajectoryData.
        """
        omega = speed / radius  # угловая скорость (рад/с)
        N = int(self._duration / self._dt) + 1
        timestamps = np.linspace(0, self._duration, N)

        states = np.zeros((N, 4))
        controls = np.zeros((N, 2))

        for i, t in enumerate(timestamps):
            psi = omega * t
            states[i] = [
                radius * np.sin(psi),       # x (север)
                radius * (1 - np.cos(psi)),  # y (восток)
                speed,                        # v
                BaseMotionModel.normalize_angle(psi),  # ψ
            ]
            controls[i] = [omega, 0.0]  # постоянная ω, нулевое ускорение

        return TrajectoryData(timestamps=timestamps, states=states, controls=controls)

    def _generate_eight(
        self,
        radius: float = 50.0,
        speed: float = 1.5,
        **_kwargs,
    ) -> TrajectoryData:
        """Движение по фигуре «восьмёрка» (лемниската Бернулли).

        Args:
            radius: Масштаб фигуры (м).
            speed: Средняя скорость (м/с).

        Returns:
            TrajectoryData.
        """
        N = int(self._duration / self._dt) + 1
        timestamps = np.linspace(0, self._duration, N)
        period = self._duration / 2  # два цикла за время симуляции

        states = np.zeros((N, 4))
        controls = np.zeros((N, 2))

        for i, t in enumerate(timestamps):
            phase = 2 * np.pi * t / period
            x = radius * np.sin(phase)
            y = radius * np.sin(2 * phase) / 2

            # Производные для скорости и курса
            dx = radius * (2 * np.pi / period) * np.cos(phase)
            dy = radius * (4 * np.pi / period) * np.cos(2 * phase) / 2

            v = np.sqrt(dx**2 + dy**2)
            psi = np.arctan2(dy, dx)

            states[i] = [x, y, v, BaseMotionModel.normalize_angle(psi)]

            # Управление (приблизительное)
            if i > 0:
                dpsi = BaseMotionModel.normalize_angle(states[i, 3] - states[i - 1, 3])
                omega = dpsi / self._dt
                dv = states[i, 2] - states[i - 1, 2]
                accel = dv / self._dt
                controls[i] = [omega, accel]
            else:
                controls[i] = [0.0, 0.0]

        return TrajectoryData(timestamps=timestamps, states=states, controls=controls)

    def _generate_sine(
        self,
        amplitude: float = 30.0,
        frequency: float = 0.01,
        speed: float = 1.5,
        **_kwargs,
    ) -> TrajectoryData:
        """Синусоидальная траектория.

        Аппарат движется вдоль оси X с синусоидальным отклонением по Y.

        Args:
            amplitude: Амплитуда отклонения (м).
            frequency: Частота (Гц).
            speed: Средняя скорость вдоль X (м/с).

        Returns:
            TrajectoryData.
        """
        N = int(self._duration / self._dt) + 1
        timestamps = np.linspace(0, self._duration, N)

        states = np.zeros((N, 4))
        controls = np.zeros((N, 2))

        for i, t in enumerate(timestamps):
            x = speed * t
            y = amplitude * np.sin(2 * np.pi * frequency * t)

            dx = speed
            dy = amplitude * 2 * np.pi * frequency * np.cos(2 * np.pi * frequency * t)

            v = np.sqrt(dx**2 + dy**2)
            psi = np.arctan2(dy, dx)

            states[i] = [x, y, v, BaseMotionModel.normalize_angle(psi)]

            if i > 0:
                dpsi = BaseMotionModel.normalize_angle(states[i, 3] - states[i - 1, 3])
                omega = dpsi / self._dt
                dv = states[i, 2] - states[i - 1, 2]
                accel = dv / self._dt
                controls[i] = [omega, accel]

        return TrajectoryData(timestamps=timestamps, states=states, controls=controls)

    def _generate_straight(
        self,
        speed: float = 1.5,
        heading: float = 0.0,
        **_kwargs,
    ) -> TrajectoryData:
        """Прямолинейное движение.

        Простейшая траектория для базовой проверки фильтра.

        Args:
            speed: Скорость (м/с).
            heading: Курс (рад).

        Returns:
            TrajectoryData.
        """
        N = int(self._duration / self._dt) + 1
        timestamps = np.linspace(0, self._duration, N)

        states = np.zeros((N, 4))
        controls = np.zeros((N, 2))

        for i, t in enumerate(timestamps):
            states[i] = [
                speed * np.cos(heading) * t,
                speed * np.sin(heading) * t,
                speed,
                heading,
            ]
            controls[i] = [0.0, 0.0]

        return TrajectoryData(timestamps=timestamps, states=states, controls=controls)
