"""
Физические параметры НПА и среды.

Содержит датакласс AUVParams, описывающий характеристики необитаемого
подводного аппарата и условия внешней среды. На основе этих параметров
вычисляются ковариационные матрицы шума процесса Q и начальная P0
для фильтров Калмана (раздел 2.1 диплома).

Ключевые формулы (раздел 2.1):
    ΔV_max  = (V_max / τ) * Δt      — макс. изменение скорости за шаг
    Δψ_max  = ψ̇_max * Δt            — макс. изменение курса за шаг
    σ_wV    = ΔV_max / 3             — СКО шума скорости (правило 3σ)
    σ_wψ    = Δψ_max / 3             — СКО шума курса (правило 3σ)

Использование:
    from auv_params import AUVParams

    params = AUVParams()
    sigma_v, sigma_psi = params.process_noise_std()
    Q_diag = params.compute_Q_diag()

Рекомендации по тестированию:
    - Проверить delta_v_max() вручную: (2.0 / 3.0) * 0.1 ≈ 0.0667.
    - Проверить delta_psi_max() вручную: deg2rad(5) * 0.1 ≈ 0.00873.
    - Проверить правило трёх сигм: sigma_v = delta_v_max / 3.
    - Проверить compute_Q_diag(): Q[0]=Q[1]=0, Q[2]=σ²_wV, Q[3]=σ²_wψ.
    - Проверить dvl_total_noise_std(): sqrt(0.02² + 0.01²) ≈ 0.022.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class AUVParams:
    """Физические параметры НПА и условия среды.

    Attributes:
        mass: Масса аппарата (кг).
        max_speed: Максимальная скорость (м/с).
        time_constant: Характерное время разгона τ (с).
        max_yaw_rate_deg: Максимальная угловая скорость рыскания (°/с).
        water_density: Плотность воды (кг/м³).
        temperature: Температура воды (°C).
        sound_speed: Скорость звука в воде (м/с).
        dt: Шаг дискретизации (с). Δt = 0.1 с (раздел 2.1).
        dvl_base_noise_std: Базовое СКО шума DVL (м/с).
        dvl_temp_noise_std: Дополнительное СКО DVL из-за температуры (м/с).
            При отклонении ±5°C от номинальных 20°C — доп. 0.01 м/с.
        initial_position_uncertainty: Начальная неопределённость позиции (м).
        initial_speed_uncertainty: Начальная неопределённость скорости (м/с).
        initial_heading_uncertainty_deg: Начальная неопределённость курса (°).
    """

    # --- Параметры аппарата ---
    mass: float = 50.0                    # кг
    max_speed: float = 2.0                # м/с
    time_constant: float = 3.0            # с (характерное время разгона)
    max_yaw_rate_deg: float = 5.0         # °/с

    # --- Параметры среды ---
    water_density: float = 1025.0         # кг/м³
    temperature: float = 20.0             # °C
    sound_speed: float = 1480.0           # м/с

    # --- Дискретизация ---
    dt: float = 0.1                       # с (раздел 2.1)

    # --- Погрешности DVL ---
    dvl_base_noise_std: float = 0.02      # м/с (базовая)
    dvl_temp_noise_std: float = 0.01      # м/с (температурная добавка)

    # --- Погрешности ИНС (раздел 2.3) ---
    ins_gyro_noise_std: float = 0.05      # рад/с (белый шум МЭМС гироскопа)
    ins_accel_noise_std: float = 0.1      # м/с² (белый шум МЭМС акселерометра)

    # --- Начальная неопределённость ---
    initial_position_uncertainty: float = 10.0   # м
    initial_speed_uncertainty: float = 0.5       # м/с
    initial_heading_uncertainty_deg: float = 0.5 # °

    # ------------------------------------------------------------------
    # Вычисляемые свойства
    # ------------------------------------------------------------------

    @property
    def max_yaw_rate_rad(self) -> float:
        """Максимальная угловая скорость рыскания (рад/с)."""
        return np.deg2rad(self.max_yaw_rate_deg)

    @property
    def initial_heading_uncertainty_rad(self) -> float:
        """Начальная неопределённость курса (рад)."""
        return np.deg2rad(self.initial_heading_uncertainty_deg)

    # ------------------------------------------------------------------
    # Методы вычисления параметров фильтра (раздел 2.1)
    # ------------------------------------------------------------------

    def delta_v_max(self) -> float:
        """Максимальное изменение скорости за один шаг Δt.

        ΔV_max = (V_max / τ) * Δt   (раздел 2.1)

        Returns:
            ΔV_max (м/с).
        """
        return (self.max_speed / self.time_constant) * self.dt

    def delta_psi_max(self) -> float:
        """Максимальное изменение курса за один шаг Δt.

        Δψ_max = ψ̇_max * Δt   (раздел 2.1)

        Returns:
            Δψ_max (рад).
        """
        return self.max_yaw_rate_rad * self.dt

    def process_noise_std(self) -> tuple[float, float]:
        """СКО шума процесса по правилу трёх сигм (раздел 2.1).

        σ_wV  = ΔV_max / 3
        σ_wψ  = Δψ_max / 3

        Returns:
            (σ_wV, σ_wψ) — СКО шума скорости и курса.
        """
        sigma_v = self.delta_v_max() / 3.0
        sigma_psi = self.delta_psi_max() / 3.0
        return sigma_v, sigma_psi

    def compute_Q_diag(self) -> np.ndarray:
        """Диагональ ковариационной матрицы шума процесса Q.

        Q = diag(0, 0, σ²_wV, σ²_wψ)

        Шум процесса складывается из двух источников:
        1. Неопределённость модели (из динамических пределов аппарата,
           правило 3σ).
        2. Шум ИНС, который подаётся на вход модели через predict().
           За один шаг Δt ИНС вносит ошибку:
             σ_ins_v = σ_accel × Δt  (ошибка скорости от шума акселерометра)
             σ_ins_ψ = σ_gyro × Δt   (ошибка курса от шума гироскопа)

        Суммарная дисперсия: σ² = σ²_model + σ²_ins (раздел 2.1).

        Шум по координатам (x, y) не вводится, так как координаты
        являются интегральными величинами — их неопределённость
        нарастает через модель движения (раздел 2.1).

        Returns:
            Вектор [0, 0, σ²_wV, σ²_wψ] размерности 4.
        """
        sigma_v_model, sigma_psi_model = self.process_noise_std()

        # Вклад шума ИНС за один шаг
        sigma_v_ins = self.ins_accel_noise_std * self.dt
        sigma_psi_ins = self.ins_gyro_noise_std * self.dt

        # Суммарная дисперсия (независимые источники складываются)
        sigma_v_total_sq = sigma_v_model ** 2 + sigma_v_ins ** 2
        sigma_psi_total_sq = sigma_psi_model ** 2 + sigma_psi_ins ** 2

        return np.array([0.0, 0.0, sigma_v_total_sq, sigma_psi_total_sq])

    def compute_P0_diag(self) -> np.ndarray:
        """Диагональ начальной ковариационной матрицы P0.

        P0 = diag(σ²_x0, σ²_y0, σ²_v0, σ²_ψ0)

        Значения определяются начальной неопределённостью
        положения, скорости и курса (раздел 2.5).

        Returns:
            Вектор [σ²_x0, σ²_y0, σ²_v0, σ²_ψ0] размерности 4.
        """
        return np.array([
            self.initial_position_uncertainty ** 2,
            self.initial_position_uncertainty ** 2,
            self.initial_speed_uncertainty ** 2,
            self.initial_heading_uncertainty_rad ** 2,
        ])

    def dvl_total_noise_std(self) -> float:
        """Суммарное СКО шума DVL с учётом температурной погрешности.

        σ_DVL = sqrt(σ²_base + σ²_temp)

        При отклонении температуры от номинальных 20°C на ±5°C
        дополнительная погрешность 0.01 м/с (раздел 2.3).

        Returns:
            Суммарное СКО шума DVL (м/с).
        """
        return float(np.sqrt(
            self.dvl_base_noise_std ** 2 + self.dvl_temp_noise_std ** 2
        ))

    def summary(self) -> str:
        """Текстовое описание параметров и вычисленных величин."""
        sigma_v, sigma_psi = self.process_noise_std()
        Q = self.compute_Q_diag()
        lines = [
            "=== Параметры НПА ===",
            f"  Масса:               {self.mass} кг",
            f"  Макс. скорость:      {self.max_speed} м/с",
            f"  Время разгона τ:     {self.time_constant} с",
            f"  Макс. угл. скорость: {self.max_yaw_rate_deg}°/с "
            f"({self.max_yaw_rate_rad:.4f} рад/с)",
            f"  Шаг Δt:              {self.dt} с",
            "",
            "=== Параметры среды ===",
            f"  Плотность воды:      {self.water_density} кг/м³",
            f"  Температура:         {self.temperature}°C",
            f"  Скорость звука:      {self.sound_speed} м/с",
            "",
            "=== Шумы ИНС (МЭМС, раздел 2.3) ===",
            f"  σ_gyro:              {self.ins_gyro_noise_std} рад/с "
            f"({np.rad2deg(self.ins_gyro_noise_std):.2f}°/с)",
            f"  σ_accel:             {self.ins_accel_noise_std} м/с²",
            f"  σ_ins_v (за шаг):    {self.ins_accel_noise_std * self.dt:.6f} м/с",
            f"  σ_ins_ψ (за шаг):    {self.ins_gyro_noise_std * self.dt:.6f} рад",
            "",
            "=== Вычисленные величины ===",
            f"  ΔV_max:              {self.delta_v_max():.6f} м/с",
            f"  Δψ_max:              {self.delta_psi_max():.6f} рад "
            f"({np.rad2deg(self.delta_psi_max()):.4f}°)",
            f"  σ_wV (модель):       {sigma_v:.6f} м/с",
            f"  σ_wψ (модель):       {sigma_psi:.6f} рад",
            f"  σ_wV (суммарный):    {np.sqrt(Q[2]):.6f} м/с",
            f"  σ_wψ (суммарный):    {np.sqrt(Q[3]):.6f} рад",
            f"  Q_diag:              {Q}",
            f"  P0_diag:             {self.compute_P0_diag()}",
            f"  σ_DVL (с учётом T):  {self.dvl_total_noise_std():.4f} м/с",
        ]
        return "\n".join(lines)
