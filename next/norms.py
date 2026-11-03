"""
Загрузка и поиск норм-часов из базы «База для нормы часам и единицам измерениям.xlsx».
Публичный API:
    load_norms(path_or_bytes) -> NormsDB
    NormsDB.find(element, unit) -> dict | None
"""
import io
import re
import pandas as pd
from dataclasses import dataclass
from typing import Optional


# ---------- парсинг чисел с запятой и «мусором» ----------
_NUM_RE = re.compile(r"-?\d+(?:[.,]\d+)?")

def _parse_hours(raw) -> Optional[float]:
    """
    '3.8'          -> 3.8
    '0,7/1,4'      -> 0.7          (берём первое число)
    '0,334-м2'     -> 0.334
    'входит в ...' -> None
    ''             -> None
    """
    if raw is None:
        return None
    try:
        if pd.isna(raw):
            return None
    except (TypeError, ValueError):
        pass
    s = str(raw).strip()
    if not s:
        return None
    # мусорные формулировки
    low = s.lower()
    if any(k in low for k in ("входит", "в насыпь", "из вкладки", "нет данных", "—", " - ")):
        # «входит в ...» — нормы нет
        if "входит" in low or "из вкладки" in low:
            return None
    m = _NUM_RE.search(s)
    if not m:
        return None
    try:
        return float(m.group().replace(",", "."))
    except ValueError:
        return None


# ---------- нормализация единиц (совместимо с app.py) ----------
_UNIT_NORMALIZE = {
    "м³": "м3", "m3": "м3", "куб": "м3", "куб.м": "м3",
    "м²": "м2", "m2": "м2", "кв": "м2", "кв.м": "м2",
    "кг": "кг", "kg": "кг",
    "т": "т", "тн": "т", "ton": "т", "тонн": "т",
    "шт": "шт", "шт.": "шт", "pcs": "шт", "ед": "шт", "ед.": "шт",
    "м.п.": "м.п.", "мп": "м.п.", "пог.м": "м.п.", "пог.м.": "м.п.",
    "м.п": "м.п.", "м": "м.п.",          # «м» в базе = погонный метр
    "компл.": "компл.", "компл": "компл.",
    "стык": "стык", "стыки": "стык", "ст": "стык",
}

def _norm_unit(u) -> str:
    if u is None:
        return ""
    try:
        if pd.isna(u):
            return ""
    except (TypeError, ValueError):
        pass
    s = str(u).strip().lower()
    return _UNIT_NORMALIZE.get(s, s)


# ---------- токенизация для нечёткого матча ----------
_WORD_RE = re.compile(r"[a-zа-яё0-9]+")

def _tokens(text: str) -> set:
    return {w for w in _WORD_RE.findall((text or "").lower()) if len(w) > 2}


@dataclass
class NormRecord:
    code: str
    kind: str            # B: «д», «Р», «р», «м»
    name: str            # C
    unit: str            # D (нормализованная)
    type_short: str      # E
    hours: Optional[float]  # F
    key: str             # G — «Ключевые»
    gsp: Optional[float] = None
    velesstroy: Optional[float] = None
    interinj: Optional[float] = None
    ours: Optional[float] = None

    def as_dict(self) -> dict:
        return {
            "code": self.code,
            "name": self.name,
            "unit": self.unit,
            "hours": self.hours,
            "key": self.key,
        }


class NormsDB:
    def __init__(self, records: list[NormRecord]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def find(self, element: str, unit: str) -> Optional[NormRecord]:
        """
        Ищет лучшую запись по наименованию + единице.
        Приоритеты:
          1) точное совпадение нормализованной единицы + максимум общих токенов
          2) если нормы по точной единице нет — берём лучший по токенам
             (в этом случае hours будет относиться к другой единице —
              вызывающий код сам решит, использовать или нет)
        """
        if not element:
            return None
        want_unit = _norm_unit(unit)
        el_tokens = _tokens(element)
        if not el_tokens:
            return None

        best_same_unit = None
        best_same_score = 0
        best_any = None
        best_any_score = 0

        for rec in self.records:
            if rec.hours is None:
                continue
            rec_tokens = _tokens(rec.name)
            if not rec_tokens:
                continue
            common = el_tokens & rec_tokens
            score = len(common)

            # усиление за вхождение полной подстроки
            if rec.name.lower() in (element or "").lower():
                score += 3
            if (element or "").lower() in rec.name.lower():
                score += 2

            if score <= 0:
                continue

            if rec.unit and rec.unit == want_unit:
                if score > best_same_score:
                    best_same_score = score
                    best_same_unit = rec
            if score > best_any_score:
                best_any_score = score
                best_any = rec

        return best_same_unit or best_any


def load_norms(source) -> NormsDB:
    """
    source: str | bytes | file-like.
    Читает первый лист, пропускает «Раздел ...» строки.
    """
    if isinstance(source, bytes):
        source = io.BytesIO(source)
    df = pd.read_excel(source, dtype=str, header=0)

    # Ожидаемые имена колонок (A..K)
    # На случай если шапка сдвинется — ищем по названиям.
    col_map = {}
    for c in df.columns:
        low = str(c).strip().lower()
        if low == "п":                  col_map["code"] = c
        elif low == "вид":              col_map["kind"] = c
        elif "наименован" in low:       col_map["name"] = c
        elif "ед" in low and "изм" in low: col_map["unit"] = c
        elif low == "тип":              col_map["type"] = c
        elif "норм" in low and "час" in low: col_map["hours"] = c
        elif "ключев" in low:           col_map["key"] = c
        elif "гсп" in low:              col_map["gsp"] = c
        elif "велесстрой" in low:       col_map["veles"] = c
        elif "интер" in low:            col_map["inter"] = c
        elif low == "наш":              col_map["ours"] = c

    required = {"code", "name", "unit", "hours"}
    missing = required - col_map.keys()
    if missing:
        raise ValueError(f"В файле норм не найдены обязательные колонки: {missing}")

    records: list[NormRecord] = []
    for _, row in df.iterrows():
        name = (row.get(col_map["name"]) or "")
        name_s = str(name).strip()
        if not name_s:
            continue
        # строки-разделы: «Раздел ...» — в колонке «Наименование»
        if name_s.lower().startswith("раздел"):
            continue

        records.append(NormRecord(
            code=str(row.get(col_map["code"]) or "").strip(),
            kind=str(row.get(col_map.get("kind", ""), "") or "").strip(),
            name=name_s,
            unit=_norm_unit(row.get(col_map["unit"])),
            type_short=str(row.get(col_map.get("type", ""), "") or "").strip(),
            hours=_parse_hours(row.get(col_map["hours"])),
            key=str(row.get(col_map.get("key", ""), "") or "").strip(),
            gsp=_parse_hours(row.get(col_map.get("gsp", ""), None)) if "gsp" in col_map else None,
            velesstroy=_parse_hours(row.get(col_map.get("veles", ""), None)) if "veles" in col_map else None,
            interinj=_parse_hours(row.get(col_map.get("inter", ""), None)) if "inter" in col_map else None,
            ours=_parse_hours(row.get(col_map.get("ours", ""), None)) if "ours" in col_map else None,
        ))

    return NormsDB(records)