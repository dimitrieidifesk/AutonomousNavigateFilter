"""
Симулятор сбоев датчиков (раздел 4.4).

Модифицирует данные SimulationData, удаляя измерения датчиков
в заданных временных интервалах. Это позволяет исследовать
устойчивость фильтров при потере сигнала.

Простыми словами:
    В реальной подводной навигации датчики могут отказать.
    Например, DVL теряет «дно» при проходе над обрывом,
    USBL может быть экранирован конструкцией, компас даёт
    ошибки вблизи магнитных аномалий.

    Этот модуль моделирует такие ситуации: «вырезает» измерения
    конкретного датчика на заданном отрезке времени, оставляя
    остальные данные без изменений.

Сценарии сбоев (раздел 4.4):
    - Потеря DVL на 30 секунд (100–130 с): аппарат теряет
      информацию о скорости, навигация опирается только на
      ИНС + USBL + Compass. Ожидается рост ошибки скорости.
    - Отключение компаса на 10 секунд (150–160 с): курс
      оценивается только по ИНС. Ожидается дрейф курса.
    - Комбинированный сбой: DVL (100–130 с) + Compass (150–160 с).

Как это связано с адаптивным EKF:
    При сбое датчика адаптивный EKF должен автоматически
    увеличить соответствующий элемент R (шум измерений),
    что снижает влияние «пропущенных» или «плохих» данных.
    Классический EKF не адаптируется и может сильнее ошибаться.

Использование:
    from simulation.sensor_failure import SensorFailureSimulator, FailureScenario

    failures = [
        FailureScenario(sensor_name="DVL", start_time=100.0, end_time=130.0),
    ]
    simulator = SensorFailureSimulator(failures)
    modified_data = simulator.apply(original_sim_data)

Рекомендации по тестированию:
    - Проверить, что количество измерений уменьшается после apply().
    - Проверить, что измерения вне окна сбоя не затронуты.
    - Проверить, что другие датчики не затронуты.
"""

from copy import deepcopy
from dataclasses import dataclass
import logging

from simulation.data_generator import SimulationData

logger = logging.getLogger(__name__)


@dataclass
class FailureScenario:
    """Описание одного сбоя датчика.

    Attributes:
        sensor_name: Имя датчика (должно совпадать с sensor.name,
            например 'DVL', 'Compass', 'USBL').
        start_time: Начало сбоя (с). Измерения в [start_time, end_time]
            будут удалены.
        end_time: Конец сбоя (с).
        description: Человекочитаемое описание сбоя (для логов и графиков).
    """
    sensor_name: str
    start_time: float
    end_time: float
    description: str = ""

    def __post_init__(self):
        if not self.description:
            self.description = (
                f"Сбой {self.sensor_name}: "
                f"[{self.start_time:.0f}–{self.end_time:.0f}] с"
            )


class SensorFailureSimulator:
    """Симулятор сбоев датчиков.

    Принимает список сценариев сбоев и применяет их к данным
    симуляции, удаляя измерения попавшие в окно сбоя.

    Простыми словами:
        Это «фильтр данных». Он берёт полные данные симуляции
        (без сбоев) и вырезает из них измерения тех датчиков,
        которые «отказали» в заданные моменты времени.

    Args:
        failures: Список сценариев сбоев.
    """

    def __init__(self, failures: list[FailureScenario]):
        self._failures = failures

    def apply(self, data: SimulationData) -> SimulationData:
        """Применить сбои к данным симуляции.

        Создаёт копию данных и удаляет измерения датчиков,
        попавшие в окна сбоев. Исходные данные не изменяются.

        Алгоритм (простыми словами):
            1. Копируем все данные.
            2. Для каждого сбоя находим все измерения
               соответствующего датчика.
            3. Оставляем только те, чьё время (timestamp)
               НЕ попадает в интервал [start_time, end_time].

        Args:
            data: Исходные данные симуляции (не модифицируются).

        Returns:
            Новый SimulationData с удалёнными измерениями.
        """
        modified = deepcopy(data)

        for failure in self._failures:
            sensor_name = failure.sensor_name
            if sensor_name not in modified.sensor_measurements:
                logger.warning(
                    "Датчик '%s' не найден в данных. "
                    "Доступные: %s",
                    sensor_name,
                    list(modified.sensor_measurements.keys()),
                )
                continue

            original_count = len(modified.sensor_measurements[sensor_name])

            # Фильтруем: оставляем измерения ВНЕ окна сбоя
            modified.sensor_measurements[sensor_name] = [
                meas
                for meas in modified.sensor_measurements[sensor_name]
                if not (failure.start_time <= meas.timestamp <= failure.end_time)
            ]

            removed_count = original_count - len(
                modified.sensor_measurements[sensor_name]
            )
            logger.info(
                "%s → удалено %d из %d измерений",
                failure.description,
                removed_count,
                original_count,
            )

        return modified

    @property
    def descriptions(self) -> list[str]:
        """Описания всех сбоев (для логов и подписей графиков)."""
        return [f.description for f in self._failures]


# --- Предопределённые сценарии для диплома (раздел 4.4) ---

def dvl_failure_scenario(
    start: float = 100.0,
    end: float = 130.0,
) -> FailureScenario:
    """Потеря DVL на 30 секунд.

    Моделирует ситуацию, когда DVL теряет «дно» — например,
    при проходе над подводным обрывом или каньоном.
    Без DVL аппарат не получает измерений скорости.

    Ожидаемый эффект:
        - Ошибка оценки скорости растёт.
        - Ошибка позиции увеличивается (скорость интегрируется).
        - Адаптивный EKF должен увеличить R для DVL
          после возобновления сигнала (раздел 4.4).
    """
    return FailureScenario(
        sensor_name="DVL",
        start_time=start,
        end_time=end,
        description=f"Потеря DVL [{start:.0f}–{end:.0f}] с",
    )


def compass_failure_scenario(
    start: float = 150.0,
    end: float = 160.0,
) -> FailureScenario:
    """Отключение компаса на 10 секунд.

    Моделирует сбой компаса вблизи магнитной аномалии.
    Без компаса курс оценивается только по гироскопу ИНС.

    Ожидаемый эффект:
        - Ошибка курса возрастает (дрейф гироскопа).
        - Как следствие, ухудшается оценка позиции.
    """
    return FailureScenario(
        sensor_name="Compass",
        start_time=start,
        end_time=end,
        description=f"Сбой компаса [{start:.0f}–{end:.0f}] с",
    )


def combined_failure_scenario() -> list[FailureScenario]:
    """Комбинированный сбой: DVL (100–130 с) + Compass (150–160 с).

    Сценарий «наихудшего случая» для главы 4 диплома.
    Позволяет показать робастность адаптивного EKF
    при последовательных отказах разных датчиков.
    """
    return [
        dvl_failure_scenario(100.0, 130.0),
        compass_failure_scenario(150.0, 160.0),
    ]
