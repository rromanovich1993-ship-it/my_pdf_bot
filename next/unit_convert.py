"""
Конвертация единиц измерения объёмов.

Правило (вариант А):
    приводим единицу ВЕДОМОСТИ к единице БАЗЫ.
    например, если ведомость в 'т', а норма в базе 'ч/кг' → volume' = volume * 1000, unit' = 'кг'

Что НЕ конвертируем (падает в ручной выбор):
    м2 ↔ м3, м2 ↔ кг, м3 ↔ кг — физически требует плотности/толщины,
    которых в базе норм нет.

Публичный API:
    normalize_unit(u) -> str
    convert_volume(vol, from_unit, to_unit) -> ConversionResult
    can_convert(from_unit, to_unit) -> bool
"""
from dataclasses import dataclass
from typing import Optional


# ---------- единая нормализация ----------

_UNIT_NORMALIZE = {
    "м³": "м3", "m3": "м3", "куб": "м3", "куб.м": "м3", "куб. м": "м3",
    "м²": "м2", "m2": "м2", "кв": "м2", "кв.м": "м2", "кв. м": "м2",
    "кг": "кг", "kg": "кг",
    "т": "т", "тн": "т", "ton": "т", "тонн": "т", "тонна": "т",
    "шт": "шт", "шт.": "шт", "pcs": "шт", "ед": "шт", "ед.": "шт",
    "м.п.": "м.п.", "мп": "м.п.", "пог.м": "м.п.", "пог.м.": "м.п.",
    "м.п": "м.п.", "м": "м.п.", "п.м.": "м.п.",
    "компл.": "компл.", "компл": "компл.", "комплект": "компл.",
    "стык": "стык", "стыки": "стык", "ст": "стык",
}


def normalize_unit(u) -> str:
    if u is None:
        return ""
    try:
        # NaN из pandas
        if isinstance(u, float) and u != u:
            return ""
    except Exception:
        pass
    s = str(u).strip().lower()
    if not s:
        return ""
    return _UNIT_NORMALIZE.get(s, s)


# ---------- таблица коэффициентов ----------
# Ключ: (from, to), значение: множитель, который применяется к объёму.
# Т.е. volume_in_to = volume_in_from * factor

_CONVERSIONS = {
    # масса
    ("т", "кг"): 1000.0,
    ("кг", "т"): 0.001,
    # длина (у нас всё в м.п., но на всякий)
    ("м.п.", "м"): 1.0,
    ("м", "м.п."): 1.0,
    # штучные
    ("компл.", "шт"): 1.0,
    ("шт", "компл."): 1.0,
    # идентичные
    ("м3", "м3"): 1.0,
    ("м2", "м2"): 1.0,
    ("кг", "кг"): 1.0,
    ("т", "т"): 1.0,
    ("шт", "шт"): 1.0,
    ("м.п.", "м.п."): 1.0,
    ("компл.", "компл."): 1.0,
    ("стык", "стык"): 1.0,
}

# Явно запрещённые пары — конвертация физически невозможна без доп. данных.
_FORBIDDEN = {
    frozenset({"м2", "м3"}),
    frozenset({"м2", "кг"}),
    frozenset({"м2", "т"}),
    frozenset({"м3", "кг"}),
    frozenset({"м3", "т"}),
    frozenset({"кг", "м.п."}),
    frozenset({"т", "м.п."}),
    frozenset({"м2", "м.п."}),
    frozenset({"м3", "м.п."}),
    frozenset({"кг", "шт"}),
    frozenset({"т", "шт"}),
    frozenset({"м3", "шт"}),
    frozenset({"м2", "шт"}),
    frozenset({"м.п.", "шт"}),
    frozenset({"стык", "шт"}),
}


@dataclass
class ConversionResult:
    ok: bool
    volume: Optional[float]     # сконвертированный объём (None если ok=False)
    unit: str                   # итоговая единица (to_unit, нормализованная)
    factor: Optional[float]
    reason: str                 # пустая строка если ok=True

    def as_dict(self) -> dict:
        return {
            "ok": self.ok,
            "volume": self.volume,
            "unit": self.unit,
            "factor": self.factor,
            "reason": self.reason,
        }


def can_convert(from_unit, to_unit) -> bool:
    f = normalize_unit(from_unit)
    t = normalize_unit(to_unit)
    if not f or not t:
        return False
    if f == t:
        return True
    return (f, t) in _CONVERSIONS


def convert_volume(vol, from_unit, to_unit) -> ConversionResult:
    """
    vol * factor -> в единицах to_unit.
    Если конвертация невозможна — ok=False, volume=None, reason объясняет почему.
    """
    f = normalize_unit(from_unit)
    t = normalize_unit(to_unit)

    # нет данных — не конвертируем
    if not f:
        return ConversionResult(False, None, t, None,
                                f"не указана исходная единица (ведомость)")
    if not t:
        return ConversionResult(False, None, "", None,
                                f"не указана целевая единица (база)")

    # объём отсутствует — вернуть как есть, но без значения
    if vol is None:
        return ConversionResult(False, None, t, None, "объём не задан")
    try:
        v = float(vol)
    except (TypeError, ValueError):
        return ConversionResult(False, None, t, None,
                                f"объём не число: {vol!r}")

    # единицы совпали
    if f == t:
        return ConversionResult(True, v, t, 1.0, "")

    # запрещённая пара
    if frozenset({f, t}) in _FORBIDDEN:
        return ConversionResult(
            False, None, t, None,
            f"конвертация '{f}' ↔ '{t}' физически невозможна без доп. данных"
        )

    # известный множитель
    factor = _CONVERSIONS.get((f, t))
    if factor is None:
        return ConversionResult(
            False, None, t, None,
            f"нет правила конвертации '{f}' → '{t}'"
        )

    return ConversionResult(True, v * factor, t, factor, "")