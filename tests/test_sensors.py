"""
Тесты датчиков.

Проверяемые сценарии:
    1. Размерность измерений каждого датчика.
    2. Размерность матрицы H для каждого датчика.
    3. Среднее зашумлённых измерений ≈ истинному значению.
    4. Дисперсия шума ≈ заявленной.
    5. is_available() корректно работает по частоте.
    6. ИНС: get_control() возвращает [ω, a] правильной размерности.
"""

import numpy as np
import pytest

from sensors.dvl import DVL
from sensors.compass import Compass
from sensors.usbl import USBL
from sensors.ins import INS


class TestDVL:
    """Тесты доплеровского лага."""

    def test_measurement_dimension(self, dvl):
        """DVL возвращает 1D измерение (скорость)."""
        assert dvl.measurement_dim == 1

    def test_H_matrix_shape(self, dvl):
        """H матрица DVL: (1 × 4)."""
        H = dvl.get_H()
        assert H.shape == (1, 4)

    def test_H_selects_speed(self, dvl):
        """H выбирает скорость (индекс 2)."""
        H = dvl.get_H()
        expected = np.array([[0, 0, 1, 0]])
        np.testing.assert_array_equal(H, expected)

    def test_measurement_mean_close_to_true(self, dvl):
        """Среднее N измерений ≈ истинному значению."""
        true_state = np.array([10.0, 20.0, 1.5, 0.3])
        measurements = []

        for _ in range(1000):
            meas = dvl.measure(true_state, timestamp=0.0)
            measurements.append(meas.value[0])

        mean_meas = np.mean(measurements)
        assert mean_meas == pytest.approx(1.5, abs=0.01)

    def test_measurement_std_close_to_declared(self, dvl):
        """СКО измерений ≈ заявленному шуму (0.03 м/с с учётом температуры)."""
        true_state = np.array([0.0, 0.0, 1.5, 0.0])
        measurements = []

        for _ in range(5000):
            meas = dvl.measure(true_state, timestamp=0.0)
            measurements.append(meas.value[0])

        std_meas = np.std(measurements)
        assert std_meas == pytest.approx(0.03, abs=0.005)


class TestCompass:
    """Тесты магнитного компаса."""

    def test_measurement_dimension(self, compass):
        """Compass возвращает 1D измерение (курс)."""
        assert compass.measurement_dim == 1

    def test_H_selects_heading(self, compass):
        """H выбирает курс (индекс 3)."""
        H = compass.get_H()
        expected = np.array([[0, 0, 0, 1]])
        np.testing.assert_array_equal(H, expected)

    def test_measurement_mean(self, compass):
        """Среднее измерений курса ≈ истинному."""
        true_state = np.array([0.0, 0.0, 1.5, np.pi / 4])
        measurements = []

        for _ in range(1000):
            meas = compass.measure(true_state, timestamp=0.0)
            measurements.append(meas.value[0])

        assert np.mean(measurements) == pytest.approx(np.pi / 4, abs=0.01)


class TestUSBL:
    """Тесты гидроакустики."""

    def test_measurement_dimension(self, usbl):
        """USBL возвращает 2D измерение (x, y)."""
        assert usbl.measurement_dim == 2

    def test_H_selects_position(self, usbl):
        """H выбирает координаты (индексы 0, 1)."""
        H = usbl.get_H()
        expected = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        np.testing.assert_array_equal(H, expected)

    def test_measurement_shape(self, usbl):
        """Измерение имеет размерность 2."""
        true_state = np.array([10.0, 20.0, 1.5, 0.3])
        meas = usbl.measure(true_state, timestamp=0.0)
        assert meas.value.shape == (2,)

    def test_R_shape(self, usbl):
        """R имеет размерность (2 × 2)."""
        assert usbl.R.shape == (2, 2)


class TestINS:
    """Тесты ИНС."""

    def test_control_dimension(self, ins):
        """get_control() возвращает [ω, a] размерности 2."""
        control = ins.get_control(0.1, 0.05)
        assert control.shape == (2,)

    def test_control_mean_close_to_true(self, ins):
        """Среднее N управлений ≈ истинному."""
        controls = []
        for _ in range(1000):
            c = ins.get_control(0.1, 0.05)
            controls.append(c)
        controls = np.array(controls)

        assert np.mean(controls[:, 0]) == pytest.approx(0.1, abs=0.005)
        assert np.mean(controls[:, 1]) == pytest.approx(0.05, abs=0.01)


class TestSensorAvailability:
    """Тесты доступности датчиков по частоте (dt=0.1 с, раздел 2.1)."""

    def test_dvl_available_at_5hz(self, dvl):
        """DVL (5 Гц) доступен каждые 0.2 с при dt=0.1."""
        dt = 0.1
        available_count = sum(
            1 for i in range(100)
            if dvl.is_available(i * dt, dt)
        )
        # Ожидаем ~50 измерений за 10 секунд (5 Гц * 10 с)
        assert 40 <= available_count <= 60

    def test_usbl_available_at_05hz(self, usbl):
        """USBL (0.5 Гц) доступен каждые 2 с при dt=0.1."""
        dt = 0.1
        available_count = sum(
            1 for i in range(100)
            if usbl.is_available(i * dt, dt)
        )
        # Ожидаем ~5 измерений за 10 секунд (0.5 Гц * 10 с)
        assert 3 <= available_count <= 8
