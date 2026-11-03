"""
Агент #3: проверка логики + конвертация единиц + нормы часов.

Порядок работы (run_agent3):
    1. Автофиксы механики (Вариант В):
       - перестановка фаз внутри WBS
       - перевязка predecessor
       - удаление битых ссылок
    2. Повторные проверки; оставшиеся issues — только семантические
    3. Конвертация единиц volume -> unit базы (вариант А)
    4. Матчинг норм:
       - точный (unit совпал) -> high
       - похожий по ключу/типу -> medium
       - не найден -> None (в ручной выбор)
    5. Обогащение activities колонками:
       Norm Hours, Total Hours, Norm Source, Norm Confidence, Norm Reason
    6. Финальный отчёт: issues + сводка

Публичный API:
    run_agent3(activities, edited_df, norms_db) -> Agent3Result
    build_llm_fix_prompt(issues, activities, edited_df) -> str
"""
from dataclasses import dataclass, field
from typing import Optional
import re

import pandas as pd

from unit_convert import convert_volume, normalize_unit, can_convert
from validator import (
    Issue,
    PHASE_ORDER,
    run_all_checks,
)
from norms import NormsDB, NormRecord


# ============================================================
# КОНФИГ
# ============================================================

# Соответствие нашей категории (phase) типу из базы норм (колонка "Тип").
# Используется для второго прохода матчинга, когда точное наименование не совпало.
CATEGORY_TO_TYPE = {
    "demolition":    "д",
    "earthwork":     "з",
    "backfill":      "з",
    "rebar":         "б",
    "concrete":      "б",
    "formwork":      "б",
    "waterproofing": "от",
    "insulation":    "ти",
    "pipeline":      "тр",
    "installation":  "тр",
    "masonry":       "от",
    "finishing":     "от",
    "roofing":       "от",
    "electrical":    "эл",
    "hvac":          "ов",
    "general":       "",
}

# Порог уверенности для "похожего" матча.
# Чем больше общих токенов, тем выше. Если ниже — ставим confidence="low".
CONFIDENCE_MEDIUM_MIN_TOKENS = 2
CONFIDENCE_LOW_MIN_TOKENS = 1


# ============================================================
# РЕЗУЛЬТАТ
# ============================================================

@dataclass
class Agent3Result:
    activities: list                    # обогащённые activities
    issues: list                        # оставшиеся issues (после автофикса)
    autofixed_count: int                # сколько issues закрыто автофиксом
    autofix_log: list                   # что именно автофикс сделал
    needs_llm_retry: bool               # есть ли семантические issues для повтора
    manual_selection_rows: list         # строки activities, где норма = None

    def summary(self) -> dict:
        errors = sum(1 for i in self.issues if i.severity == "error")
        warnings = sum(1 for i in self.issues if i.severity == "warning")
        return {
            "errors": errors,
            "warnings": warnings,
            "autofixed": self.autofixed_count,
            "manual_selection": len(self.manual_selection_rows),
            "needs_llm_retry": self.needs_llm_retry,
        }


# ============================================================
# 1) АВТОФИКСЫ (Вариант В — только механика)
# ============================================================

def _norm_wbs(w) -> str:
    return str(w or "").strip()


def _norm_phase(p) -> str:
    return str(p or "").strip().lower()


def _order(p) -> int:
    return PHASE_ORDER.get(_norm_phase(p), 99)


def _autofix_sequence(activities: list) -> tuple:
    """
    Переставляет задачи внутри WBS по PHASE_ORDER.
    Возвращает (activities', log).
    Activity ID НЕ меняем (чтобы не ломать ссылки), меняем только порядок в списке.
    """
    log = []
    by_wbs = {}
    for a in activities:
        by_wbs.setdefault(_norm_wbs(a.get("WBS")), []).append(a)

    result = []
    for wbs, group in by_wbs.items():
        original = list(group)
        reordered = sorted(group, key=lambda x: _order(x.get("Category")))
        if [a["Activity ID"] for a in original] != [a["Activity ID"] for a in reordered]:
            log.append(f"[{wbs}] задачи переупорядочены по фазам")
        result.extend(reordered)

    return result, log


def _autofix_predecessors(activities: list) -> tuple:
    """
    Перевязывает predecessor внутри каждого WBS:
      - битые ссылки обнуляются
      - цепочка строится по порядку фаз
    Возвращает (activities', log).
    """
    log = []
    ids = {str(a.get("Activity ID", "")) for a in activities}

    # сначала чистим битые ссылки
    for a in activities:
        pred = str(a.get("Predecessor", "") or "").strip()
        if pred and pred not in ids:
            a["Predecessor"] = ""
            log.append(f"{a.get('Activity ID')}: сброшена битая ссылка '{pred}'")

    # теперь перевязываем по порядку внутри WBS
    by_wbs = {}
    for a in activities:
        by_wbs.setdefault(_norm_wbs(a.get("WBS")), []).append(a)

    for wbs, group in by_wbs.items():
        prev_id = ""
        for a in group:   # group уже отсортирован по phase (см. _autofix_sequence)
            a["Predecessor"] = prev_id
            prev_id = str(a.get("Activity ID", ""))

    return activities, log


def _autofix_semantics(activities: list) -> tuple:
    """
    Переставляет задачи в WBS так, чтобы убрать FORBIDDEN_BEFORE-нарушения.
    По сути — то же, что _autofix_sequence, но с логом «семантический конфликт».
    Здесь оставляем мягкий вариант: применяем _autofix_sequence повторно
    и логируем, если были ошибки SEM_FORBIDDEN / SEM_MISSING_DEP.
    """
    log = []
    activities, sub_log = _autofix_sequence(activities)
    for s in sub_log:
        log.append(f"semantics: {s}")
    # после перестановки — перевязать предков
    activities, pred_log = _autofix_predecessors(activities)
    log.extend(pred_log)
    return activities, log


def apply_autofix(activities: list, issues: list) -> tuple:
    """
    Применяет автофиксы по типам issues.
    Возвращает (activities', remaining_issues, fixed_count, log).
    """
    if not issues:
        return activities, [], 0, []

    # копируем, чтобы не мутировать вход
    acts = [dict(a) for a in activities]
    codes = {i.code for i in issues}
    log = []
    fixed_codes = set()

    # 1) порядок фаз
    if "SEQ_ORDER" in codes:
        acts, sub = _autofix_sequence(acts)
        log.extend(sub)
        fixed_codes.add("SEQ_ORDER")

    # 2) битые ссылки и циклы
    if codes & {"PRED_BROKEN", "PRED_CYCLE", "PRED_MISSING"}:
        acts, sub = _autofix_predecessors(acts)
        log.extend(sub)
        fixed_codes |= {"PRED_BROKEN", "PRED_CYCLE", "PRED_MISSING"}

    # 3) семантика — после перестановки должна уйти сама
    if codes & {"SEM_FORBIDDEN", "SEM_MISSING_DEP"}:
        acts, sub = _autofix_semantics(acts)
        log.extend(sub)
        fixed_codes |= {"SEM_FORBIDDEN", "SEM_MISSING_DEP"}

    # 4) coverage не чиним кодом — оставляем как есть
    remaining = [i for i in issues if i.code not in fixed_codes]
    fixed_count = len(issues) - len(remaining)

    return acts, remaining, fixed_count, log


# ============================================================
# 2) МАТЧИНГ НОРМ
# ============================================================

def _tokens(text: str) -> set:
    return {w for w in re.findall(r"[a-zа-яё0-9]+", (text or "").lower()) if len(w) > 2}


def match_norm_with_fallback(
    element: str,
    unit: str,
    category: str,
    norms_db: Optional[NormsDB],
) -> tuple:
    """
    Возвращает (record, confidence, reason).

    confidence:
        "high"   — точное совпадение по единице + ≥ 2 общих токена
        "medium" — совпадение по единице + 1 токен, или похожий тип/ключ
        "low"    — матч только по типу или по 1 общему токену без единицы
        "none"   — ничего не нашли

    Если confidence == "none", record = None.
    """
    if norms_db is None or not element:
        return None, "none", "база норм не загружена"

    # --- проход 1: точное совпадение по единице ---
    el_tokens = _tokens(element)
    want_unit = normalize_unit(unit)
    type_hint = CATEGORY_TO_TYPE.get(category, "")

    best = None
    best_score = 0
    best_rec = None

    for rec in norms_db.records:
        if rec.hours is None:
            continue
        rec_tokens = _tokens(rec.name)
        common = el_tokens & rec_tokens
        score = len(common)
        if rec.name.lower() in (element or "").lower():
            score += 3
        if (element or "").lower() in rec.name.lower():
            score += 2
        if type_hint and rec.type_short and rec.type_short == type_hint:
            score += 1
        if score <= 0:
            continue

        # точное совпадение единицы
        if rec.unit and rec.unit == want_unit:
            if score > best_score:
                best_score = score
                best_rec = rec
        if score > best:
            best = score

    if best_rec is not None:
        if best_score >= 2:
            return best_rec, "high", ""
        if best_score >= 1:
            return best_rec, "medium", "мало общих слов в наименовании"

    # --- проход 2: по типу/ключу, без совпадения единицы ---
    type_rec = None
    for rec in norms_db.records:
        if rec.hours is None:
            continue
        if type_hint and rec.type_short == type_hint:
            rec_tokens = _tokens(rec.name)
            common = el_tokens & rec_tokens
            if common:
                type_rec = rec
                break

    if type_rec is not None:
        return type_rec, "low", (
            f"найдено по типу '{type_hint}', но единица не совпала: "
            f"норма в '{type_rec.unit}', ведомость в '{want_unit}'"
        )

    return None, "none", "нет подходящей записи в базе"


# ============================================================
# 3) ОБОГАЩЕНИЕ СТРОК
# ============================================================

def enrich_norms(activities: list, edited_df: pd.DataFrame,
                 norms_db: Optional[NormsDB]) -> tuple:
    """
    Для каждой activity:
      - подбирает запись нормы (по element из ведомости, у которой
        совпадает WBS и phase=category)
      - конвертирует volume ведомости в единицу базы (вариант А)
      - считает Total Hours = volume * norm_hours

    Возвращает (activities', manual_selection_rows).
    """
    manual_selection_rows = []

    # индекс ведомости: (wbs, category) -> список строк
    est_index = {}
    if edited_df is not None and not edited_df.empty:
        for _, row in edited_df.iterrows():
            wbs = _norm_wbs(row.get("wbs_parent"))
            cat = _norm_phase(row.get("category"))
            est_index.setdefault((wbs, cat), []).append(row)

    for a in activities:
        wbs = _norm_wbs(a.get("WBS"))
        cat = _norm_phase(a.get("Category"))
        vol = a.get("Volume", 0)
        unit = a.get("Unit", "")

        # ищем соответствующую строку ведомости
        candidates = est_index.get((wbs, cat), [])
        source_row = None
        if candidates:
            # если одна — берём её; если несколько — по наибольшему совпадению слов
            if len(candidates) == 1:
                source_row = candidates[0]
            else:
                task_tokens = _tokens(a.get("Activity Name", ""))
                best_row = None
                best_score = 0
                for row in candidates:
                    row_tokens = _tokens(str(row.get("element", "")))
                    score = len(task_tokens & row_tokens)
                    if score > best_score:
                        best_score = score
                        best_row = row
                source_row = best_row or candidates[0]

        if source_row is None:
            a["Norm Hours"] = ""
            a["Total Hours"] = ""
            a["Norm Source"] = ""
            a["Norm Confidence"] = "none"
            a["Norm Reason"] = "нет строки в ведомости"
            manual_selection_rows.append({
                "Activity ID": a.get("Activity ID"),
                "WBS": wbs,
                "Activity Name": a.get("Activity Name"),
                "Volume": vol,
                "Unit": unit,
                "reason": "нет строки в ведомости",
            })
            continue

        element = str(source_row.get("element", "")).strip()

        # 3.1 матчинг
        rec, confidence, reason = match_norm_with_fallback(
            element, unit, cat, norms_db
        )

        # 3.2 конвертация объёма под единицу базы
        converted_vol = None
        converted_unit = unit
        conv_reason = ""
        if rec is not None:
            cr = convert_volume(vol, unit, rec.unit)
            if cr.ok:
                converted_vol = cr.volume
                converted_unit = cr.unit
            else:
                conv_reason = cr.reason
                # если конвертация невозможна — confidence понижаем до low
                if confidence == "high":
                    confidence = "medium"
                reason = (reason + "; " if reason else "") + conv_reason

        # 3.3 итоговые значения
        # 3.3 итоговые значения
        if rec is not None and converted_vol is not None and rec.hours is not None:
            norm_hours = rec.hours
            try:
                # Если volume это None или NaN, total_hours не считаем
                if vol is None or pd.isna(vol):
                    total_hours = None
                else:
                    total_hours = float(converted_vol) * float(norm_hours)
            except (TypeError, ValueError):
                total_hours = None

        a["Norm Hours"] = norm_hours if norm_hours is not None else ""
        a["Total Hours"] = total_hours if total_hours is not None else ""
        a["Norm Source"] = rec.code if rec else ""
        a["Norm Confidence"] = confidence
        a["Norm Reason"] = reason or ""

        # 3.4 в ручной выбор — если norm_hours = None
        if norm_hours is None:
            manual_selection_rows.append({
                "Activity ID": a.get("Activity ID"),
                "WBS": wbs,
                "Activity Name": a.get("Activity Name"),
                "Volume": vol,
                "Unit": unit,
                "reason": reason or "норма не найдена",
            })

    return activities, manual_selection_rows


# ============================================================
# 4) ТОЧКА ВХОДА
# ============================================================

def run_agent3(
    activities: list,
    edited_df: pd.DataFrame,
    norms_db: Optional[NormsDB],
) -> Agent3Result:
    """
    Полный проход Агента #3:
        1. Первичные проверки
        2. Автофикс механики
        3. Повторные проверки
        4. Конвертация + нормы
        5. Отчёт
    """
    # 1) первичные проверки
    issues_before = run_all_checks(activities, edited_df)

    # 2) автофикс
    acts_after, remaining, fixed_count, log = apply_autofix(
        activities, issues_before
    )

    # 3) повторные проверки (после автофикса)
    issues_after = run_all_checks(acts_after, edited_df)

    # 4) обогащение
    acts_after, manual = enrich_norms(acts_after, edited_df, norms_db)

    # 5) нужен ли LLM-повтор (есть семантические errors, которые автофикс не убрал)
    semantic_errors = [
        i for i in issues_after
        if i.code in ("SEM_FORBIDDEN", "SEM_MISSING_DEP", "COV_MISSING")
        and i.severity == "error"
    ]
    # COV_MISSING — warning, но если хотите блокировать — уберите фильтр по severity
    coverage_warnings = [i for i in issues_after if i.code == "COV_MISSING"]

    needs_llm_retry = bool(semantic_errors or coverage_warnings)

    return Agent3Result(
        activities=acts_after,
        issues=issues_after,
        autofixed_count=fixed_count,
        autofix_log=log,
        needs_llm_retry=needs_llm_retry,
        manual_selection_rows=manual,
    )


# ============================================================
# 5) ПРОМПТ ДЛЯ ПОВТОРА АГЕНТА #2
# ============================================================

def build_llm_fix_prompt(issues: list, activities: list,
                         edited_df: pd.DataFrame) -> str:
    """
    Собирает промпт для повторного запуска Агента #2 с учётом замечаний Агента #3.
    """
    # компактная ведомость
    cols = ["wbs_parent", "element", "volume", "unit", "category"]
    if edited_df is not None and not edited_df.empty:
        exist = [c for c in cols if c in edited_df.columns]
        est_table = edited_df[exist].to_string(index=False)
    else:
        est_table = "(ведомость пуста)"

    # текущие задачи
    task_lines = []
    for a in activities:
        task_lines.append(
            f"- {a.get('Activity ID')} | {a.get('WBS')} | "
            f"{a.get('Activity Name')} | phase={a.get('Category')} | "
            f"pred={a.get('Predecessor') or '—'}"
        )
    tasks_block = "\n".join(task_lines) if task_lines else "(задач нет)"

    # замечания Агента #3
    issue_lines = []
    for i in issues:
        issue_lines.append(
            f"- [{i.severity.upper()}] {i.code}: {i.message}"
        )
    issues_block = "\n".join(issue_lines) if issue_lines else "(замечаний нет)"

    return f"""Ты — Агент-Планировщик Primavera P6. Ты уже строил структуру работ,
но Агент-Верификатор нашёл замечания. Перестрой структуру с их учётом.

ВЕДОМОСТЬ ОБЪЁМОВ:
{est_table}

ТЕКУЩАЯ СТРУКТУРА (её нужно исправить):
{tasks_block}

ЗАМЕЧАНИЯ ВЕРИФИКАТОРА:
{issues_block}

ПРАВИЛА ПЕРЕСТРОЙКИ:
1. Устрани ВСЕ замечания из списка выше.
2. Сохрани wbs_parent БУКВАЛЬНО из ведомости.
3. Порядок фаз внутри WBS: earthwork → formwork → rebar → concrete →
   waterproofing/insulation → backfill → installation → finishing.
4. У каждой задачи, кроме первой в WBS, должен быть predecessor = "AUTO".
5. Не добавляй задачи, которых нет в ведомости.
6. Арматура — ОДНА задача на WBS. Бетон — ДВЕ (подготовка + конструктив).
7. Если позиция ведомости была не покрыта задачей — добавь её.

Верни строго JSON-массив (без markdown, без пояснений):
[{{"wbs_parent": "...", "task_name": "...", "phase": "...", "predecessor": null|"AUTO"}}, ...]
"""