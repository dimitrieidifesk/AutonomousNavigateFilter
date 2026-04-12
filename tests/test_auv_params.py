"""
Тесты параметров НПА (AUVParams).

Проверяемые сценарии:
    1. delta_v_max() — ручной расчёт: (2.0 / 3.0) * 0.1 = 0.06667 м/с.
    2. delta_psi_max() — ручной расчёт: deg2rad(5) * 0.1 ≈ 0.008727 рад.
    3. process_noise_std() — правило трёх сигм: σ = Δ_max / 3.
    4. compute_Q_diag() — Q[0]=Q[1]=0, Q[2]=σ²_wV, Q[3]=σ²_wψ.
    5. compute_P0_diag() — из начальных неопределённостей.
    6. dvl_total_noise_std() — sqrt(0.02² + 0.01²) ≈ 0.02236 м/с.
    7. from_auv_params() — FilterConfig создаётся корректно.
    8. Изменение параметров НПА влияет на Q.
"""

import numpy as np
import pytest

from auv_params import AUVParams
from config import FilterConfig


class TestDeltaMax:
    """Тесты предельных изменений за шаг Δt (раздел 2.1)."""

    def test_delta_v_max_default(self, auv_params):
        """ΔV_max = (V_max / τ) * Δt = (2.0 / 3.0) * 0.1."""
        expected = (2.0 / 3.0) * 0.1
        assert auv_params.delta_v_max() == pytest.approx(expected, rel=1e-10)

    def test_delta_psi_max_default(self, auv_params):
        """Δψ_max = ψ̇_max * Δt = deg2rad(5) * 0.1."""
        expected = np.deg2rad(5.0) * 0.1
        assert auv_params.delta_psi_max() == pytest.approx(expected, rel=1e-10)

    def test_delta_v_max_custom_params(self):
        """ΔV_max для нестандартных параметров."""
        params = AUVParams(max_speed=3.0, time_constant=5.0, dt=0.2)
        expected = (3.0 / 5.0) * 0.2
        assert params.delta_v_max() == pytest.approx(expected, rel=1e-10)

    def test_delta_psi_max_custom_params(self):
        """Δψ_max для нестандартных параметров."""
        params = AUVParams(max_yaw_rate_deg=10.0, dt=0.05)
        expected = np.deg2rad(10.0) * 0.05
        assert params.delta_psi_max() == pytest.approx(expected, rel=1e-10)


class TestProcessNoiseStd:
    """Тесты правила трёх сигм (раздел 2.1)."""

    def test_sigma_v(self, auv_params):
        """σ_wV = ΔV_max / 3."""
        sigma_v, _ = auv_params.process_noise_std()
        expected = auv_params.delta_v_max() / 3.0
        assert sigma_v == pytest.approx(expected, rel=1e-10)

    def test_sigma_psi(self, auv_params):
        """σ_wψ = Δψ_max / 3."""
        _, sigma_psi = auv_params.process_noise_std()
        expected = auv_params.delta_psi_max() / 3.0
        assert sigma_psi == pytest.approx(expected, rel=1e-10)

    def test_sigma_positive(self, auv_params):
        """Оба σ > 0."""
        sigma_v, sigma_psi = auv_params.process_noise_std()
        assert sigma_v > 0
        assert sigma_psi > 0


class TestComputeQDiag:
    """Тесты вычисления диагонали матрицы Q."""

    def test_q_shape(self, auv_params):
        """Q_diag имеет размерность 4."""
        Q_diag = auv_params.compute_Q_diag()
        assert Q_diag.shape == (4,)

    def test_q_xy_zero(self, auv_params):
        """Q[0] = Q[1] = 0 (координаты — интегральные величины)."""
        Q_diag = auv_params.compute_Q_diag()
        assert Q_diag[0] == 0.0
        assert Q_diag[1] == 0.0

    def test_q_v_component(self, auv_params):
        """Q[2] = σ²_wV."""
        sigma_v, _ = auv_params.process_noise_std()
        Q_diag = auv_params.compute_Q_diag()
        assert Q_diag[2] == pytest.approx(sigma_v ** 2, rel=1e-10)

    def test_q_psi_component(self, auv_params):
        """Q[3] = σ²_wψ."""
        _, sigma_psi = auv_params.process_noise_std()
        Q_diag = auv_params.compute_Q_diag()
        assert Q_diag[3] == pytest.approx(sigma_psi ** 2, rel=1e-10)

    def test_q_all_non_negative(self, auv_params):
        """Все элементы Q_diag ≥ 0."""
        Q_diag = auv_params.compute_Q_diag()
        assert np.all(Q_diag >= 0)


class TestComputeP0Diag:
    """Тесты вычисления начальной ковариационной матрицы P0."""

    def test_p0_shape(self, auv_params):
        """P0_diag имеет размерность 4."""
        P0_diag = auv_params.compute_P0_diag()
        assert P0_diag.shape == (4,)

    def test_p0_position_uncertainty(self, auv_params):
        """P0[0] = P0[1] = σ²_pos = 10² = 100."""
        P0_diag = auv_params.compute_P0_diag()
        expected = auv_params.initial_position_uncertainty ** 2
        assert P0_diag[0] == pytest.approx(expected)
        assert P0_diag[1] == pytest.approx(expected)

    def test_p0_speed_uncertainty(self, auv_params):
        """P0[2] = σ²_v0 = 0.5² = 0.25."""
        P0_diag = auv_params.compute_P0_diag()
        expected = auv_params.initial_speed_uncertainty ** 2
        assert P0_diag[2] == pytest.approx(expected)

    def test_p0_heading_uncertainty(self, auv_params):
        """P0[3] = σ²_ψ0 = (0.5°)² в рад²."""
        P0_diag = auv_params.compute_P0_diag()
        expected = np.deg2rad(auv_params.initial_heading_uncertainty_deg) ** 2
        assert P0_diag[3] == pytest.approx(expected)

    def test_p0_all_positive(self, auv_params):
        """Все элементы P0_diag > 0."""
        P0_diag = auv_params.compute_P0_diag()
        assert np.all(P0_diag > 0)


class TestDVLTotalNoise:
    """Тесты суммарного шума DVL (раздел 2.3)."""

    def test_dvl_total_noise_default(self, auv_params):
        """σ_DVL = sqrt(0.02² + 0.01²) ≈ 0.02236."""
        expected = np.sqrt(0.02**2 + 0.01**2)
        assert auv_params.dvl_total_noise_std() == pytest.approx(expected, rel=1e-10)

    def test_dvl_total_noise_no_temp(self):
        """Без температурной погрешности: σ_DVL = σ_base."""
        params = AUVParams(dvl_base_noise_std=0.02, dvl_temp_noise_std=0.0)
        assert params.dvl_total_noise_std() == pytest.approx(0.02, rel=1e-10)

    def test_dvl_total_noise_positive(self, auv_params):
        """σ_DVL > 0."""
        assert auv_params.dvl_total_noise_std() > 0


class TestFilterConfigFromAUVParams:
    """Тесты фабричного метода FilterConfig.from_auv_params()."""

    def test_creates_filter_config(self, auv_params):
        """from_auv_params() возвращает FilterConfig."""
        cfg = FilterConfig.from_auv_params(auv_params)
        assert isinstance(cfg, FilterConfig)

    def test_initial_state_zeros(self, auv_params):
        """Начальное состояние [0, 0, 0, 0] (аппарат неподвижен)."""
        cfg = FilterConfig.from_auv_params(auv_params)
        np.testing.assert_array_equal(cfg.initial_state, [0.0, 0.0, 0.0, 0.0])

    def test_q_from_auv_params(self, auv_params):
        """Q_diag совпадает с compute_Q_diag()."""
        cfg = FilterConfig.from_auv_params(auv_params)
        np.testing.assert_allclose(
            cfg.process_noise_diag, auv_params.compute_Q_diag(), rtol=1e-10
        )

    def test_p0_from_auv_params(self, auv_params):
        """P0_diag совпадает с compute_P0_diag()."""
        cfg = FilterConfig.from_auv_params(auv_params)
        np.testing.assert_allclose(
            cfg.initial_covariance_diag, auv_params.compute_P0_diag(), rtol=1e-10
        )

    def test_changed_params_affect_q(self):
        """Изменение параметров НПА меняет Q."""
        params_default = AUVParams()
        params_fast = AUVParams(max_speed=5.0, time_constant=1.0)

        q_default = params_default.compute_Q_diag()
        q_fast = params_fast.compute_Q_diag()

        # Быстрый аппарат → бо́льший шум по скорости
        assert q_fast[2] > q_default[2]


class TestAUVParamsProperties:
    """Тесты свойств AUVParams."""

    def test_max_yaw_rate_rad(self, auv_params):
        """Конвертация °/с → рад/с."""
        expected = np.deg2rad(5.0)
        assert auv_params.max_yaw_rate_rad == pytest.approx(expected)

    def test_initial_heading_uncertainty_rad(self, auv_params):
        """Конвертация ° → рад для начальной неопределённости курса."""
        expected = np.deg2rad(0.5)
        assert auv_params.initial_heading_uncertainty_rad == pytest.approx(expected)

    def test_summary_not_empty(self, auv_params):
        """summary() возвращает непустую строку."""
        s = auv_params.summary()
        assert len(s) > 100
        assert "ΔV_max" in s
        assert "σ_wV" in s
