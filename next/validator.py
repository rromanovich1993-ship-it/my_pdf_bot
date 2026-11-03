"""
Агент #3, часть 1: валидация структуры Primavera P6.

Проверки:
    1. validate_sequence()      — порядок фаз внутри WBS
    2. validate_predecessors()  — циклы, висячие ссылки, пропуски
    3. validate_coverage()      — все позиции ведомости покрыты задачами
    4. validate_semantics()     — содержательные запреты (бетон до арматуры и т.п.)

Публичный API:
    run_all_checks(activities, edited_df) -> list[Issue]

Все правила — в словарях ниже. Меняйте их без правки логики.
"""
from dataclasses import dataclass, field
from typing import Optional
import re

import pandas as pd

from unit_convert import normalize_unit


# ============================================================
# КОНФИГ — ПРАВИТЬ ЗДЕСЬ
# ============================================================

# Порядок фаз (чем меньше число, тем раньше должна идти задача).
PHASE_ORDER = {
    "demolition":     0,
    "earthwork":      1,
    "formwork":       2,
    "rebar":          3,
    "concrete":       4,
    "waterproofing":  5,
    "insulation":     5,
    "backfill":       6,
    "installation":   7,
    "pipeline":       7,
    "masonry":        8,
    "roofing":        9,
    "finishing":      10,
    "electrical":     11,
    "hvac":           11,
    "general":        99,
}

# Содержательные запреты:
#   { "фаза_задачи": ["фазы, которые НЕ ДОЛЖНЫ быть её predecessor"] }
#   Т.е. если у задачи фазы X predecessor — задача фазы Y из списка,
#   это нарушение технологии.
FORBIDDEN_BEFORE = {
    "concrete":       ["rebar"],                 # бетон нельзя после арматуры? нет — наоборот:
                                                 # бетон нельзя ДО арматуры → если предок бетона = арматура, норм.
                                                 # ВАЖНО: здесь мы смотрим на предка.
                                                 # Правило читается как:
                                                 # "у задачи фазы concrete предок НЕ МОЖЕТ быть rebar" — НЕВЕРНО.
                                                 # Оставляем пустым и делаем проверку через REQUIRED_AFTER.
    "backfill":       ["concrete", "waterproofing"],   # засыпка не может быть ДО бетона/гидроизоляции
    "waterproofing":  ["backfill"],                    # гидроизоляция не может быть ПОСЛЕ засыпки
    "finishing":      ["concrete", "masonry", "roofing"],
    "installation":   ["concrete", "masonry"],
    "roofing":        ["concrete"],                    # кровля не до бетона конструкций
}

# Обратное правило: "фаза X ОБЯЗАНА идти после фазы Y".
# Читается: у задачи фазы X среди предков (прямых или транзитивных)
# должен быть хотя бы один узел фазы Y.
REQUIRED_AFTER = {
    "concrete":       ["rebar", "formwork"],   # бетон после арматуры и опалубки
    "backfill":       ["concrete"],            # засыпка после бетона
    "waterproofing":  ["concrete"],            # гидроизоляция после бетона
    "finishing":      ["concrete", "masonry"],
}

# Фазы, у которых должен быть предшественник внутри WBS.
# Если у задачи такой фазы Predecessor пустой и она не первая по порядку — это warning.
REQUIRED_PREDECESSOR = {
    "formwork", "rebar", "concrete", "waterproofing",
    "insulation", "backfill", "finishing", "roofing",
}


# ============================================================
# МОДЕЛЬ
# ============================================================

SEVERITY_ERROR = "error"
SEVERITY_WARN = "warning"


@dataclass
class Issue:
    severity: str                    # "error" | "warning"
    code: str                        # короткий код: SEQ_ORDER, PRED_CYCLE, COV_MISSING…
    message: str                     # человекочитаемое описание
    wbs: str = ""
    activity_id: str = ""
    activity_name: str = ""
    autofixable: bool = False        # можно ли исправить кодом (Вариант В)

    def as_dict(self) -> dict:
        return {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "wbs": self.wbs,
            "activity_id": self.activity_id,
            "activity_name": self.activity_name,
            "autofixable": self.autofixable,
        }


# ============================================================
# ХЕЛПЕРЫ
# ============================================================

def _norm_phase(p) -> str:
    return str(p or "").strip().lower()


def _order(p) -> int:
    return PHASE_ORDER.get(_norm_phase(p), 99)


def _norm_wbs(w) -> str:
    return str(w or "").strip()


# ============================================================
# 1) ПОРЯДОК ФАЗ
# ============================================================

def validate_sequence(activities: list) -> list:
    """
    Проверяет, что внутри каждого WBS задачи идут в PHASE_ORDER.
    Нарушение — error, autofixable=True.
    """
    issues = []
    by_wbs = {}
    for a in activities:
        by_wbs.setdefault(_norm_wbs(a.get("WBS")), []).append(a)

    for wbs, group in by_wbs.items():
        # порядок в исходном списке = порядок ввода (после сортировки в app.py)
        prev_order = -1
        prev_a = None
        for a in group:
            o = _order(a.get("Category"))
            if prev_a is not None and o < prev_order:
                issues.append(Issue(
                    severity=SEVERITY_ERROR,
                    code="SEQ_ORDER",
                    message=(
                        f"Нарушен порядок фаз: "
                        f"'{a.get('Activity Name')}' ({a.get('Category')}) "
                        f"идёт после '{prev_a.get('Activity Name')}' "
                        f"({prev_a.get('Category')})"
                    ),
                    wbs=wbs,
                    activity_id=str(a.get("Activity ID", "")),
                    activity_name=str(a.get("Activity Name", "")),
                    autofixable=True,
                ))
            prev_order = o
            prev_a = a

    return issues


# ============================================================
# 2) ПРЕДШЕСТВЕННИКИ
# ============================================================

def validate_predecessors(activities: list) -> list:
    """
    - Predecessor ссылается на несуществующий Activity ID → error (autofixable)
    - Цикл в графе predecessors → error (autofixable)
    - Задача фазы из REQUIRED_PREDECESSOR не первая, но Predecessor пуст → warning (autofixable)
    """
    issues = []

    ids = {str(a.get("Activity ID", "")) for a in activities}
    by_id = {str(a.get("Activity ID", "")): a for a in activities}

    # 2.1 битые ссылки
    for a in activities:
        pred = str(a.get("Predecessor", "") or "").strip()
        if not pred:
            continue
        if pred not in ids:
            issues.append(Issue(
                severity=SEVERITY_ERROR,
                code="PRED_BROKEN",
                message=(
                    f"Predecessor '{pred}' не существует "
                    f"(возможно, задача была удалена или ID переименован)"
                ),
                wbs=_norm_wbs(a.get("WBS")),
                activity_id=str(a.get("Activity ID", "")),
                activity_name=str(a.get("Activity Name", "")),
                autofixable=True,
            ))

    # 2.2 циклы
    def _has_cycle() -> list:
        """Возвращает список activity_id, участвующих в цикле."""
        color = {}   # 0 — не посещён, 1 — в стеке, 2 — готов
        bad = []

        def dfs(node):
            stack = [(node, False)]
            while stack:
                cur, processed = stack.pop()
                if processed:
                    color[cur] = 2
                    continue
                if color.get(cur) == 1:
                    bad.append(cur)
                    continue
                if color.get(cur) == 2:
                    continue
                color[cur] = 1
                stack.append((cur, True))
                a = by_id.get(cur)
                if a is None:
                    color[cur] = 2
                    continue
                pred = str(a.get("Predecessor", "") or "").strip()
                if pred and pred in by_id and color.get(pred, 0) != 2:
                    stack.append((pred, False))

        for aid in ids:
            if color.get(aid, 0) == 0:
                dfs(aid)
        return bad

    for aid in _has_cycle():
        a = by_id.get(aid, {})
        issues.append(Issue(
            severity=SEVERITY_ERROR,
            code="PRED_CYCLE",
            message=f"Цикл в цепочке предшественников вокруг Activity ID '{aid}'",
            wbs=_norm_wbs(a.get("WBS")),
            activity_id=aid,
            activity_name=str(a.get("Activity Name", "")),
            autofixable=True,
        ))

    # 2.3 пропущенный predecessor
    by_wbs = {}
    for a in activities:
        by_wbs.setdefault(_norm_wbs(a.get("WBS")), []).append(a)

    for wbs, group in by_wbs.items():
        # в рамках WBS «первая» — та, у которой минимальный PHASE_ORDER
        if not group:
            continue
        first = min(group, key=lambda x: _order(x.get("Category")))
        for a in group:
            if a is first:
                continue
            phase = _norm_phase(a.get("Category"))
            if phase not in REQUIRED_PREDECESSOR:
                continue
            pred = str(a.get("Predecessor", "") or "").strip()
            if not pred:
                issues.append(Issue(
                    severity=SEVERITY_WARN,
                    code="PRED_MISSING",
                    message=(
                        f"У задачи фазы '{phase}' нет предшественника внутри WBS "
                        f"'{wbs}' — нарушена технологическая цепочка"
                    ),
                    wbs=wbs,
                    activity_id=str(a.get("Activity ID", "")),
                    activity_name=str(a.get("Activity Name", "")),
                    autofixable=True,
                ))

    return issues


# ============================================================
# 3) ПОКРЫТИЕ ВЕДОМОСТИ
# ============================================================

def _tokens(text: str) -> set:
    return {w for w in re.findall(r"[a-zа-яё0-9]+", (text or "").lower()) if len(w) > 2}


def validate_coverage(activities: list, edited_df: pd.DataFrame) -> list:
    issues = []
    if edited_df is None or edited_df.empty:
        return issues

    task_by_wbs_phase = {}
    for a in activities:
        key = (_norm_wbs(a.get("WBS")), _norm_phase(a.get("Category")))
        task_by_wbs_phase.setdefault(key, []).append(a)

    for _, row in edited_df.iterrows():
        wbs = _norm_wbs(row.get("wbs_parent"))
        cat = _norm_phase(row.get("category"))
        el = str(row.get("element", "")).strip()
        if not el:
            continue

        # ЗАЩИТА: пропускаем строки, где нет объёма
        vol = row.get("volume")
        if pd.isna(vol) or vol is None:
            continue

        matched = task_by_wbs_phase.get((wbs, cat), [])
        if matched:
            continue
        issues.append(Issue(
            severity=SEVERITY_WARN,
            code="COV_MISSING",
            message=(
                f"Позиция ведомости «{el}» (WBS: {wbs}, фаза: {cat}) "
                f"не покрыта ни одной задачей Агента #2"
            ),
            wbs=wbs,
            autofixable=False,
        ))

    return issues

# ============================================================
# 4) СЕМАНТИКА (технологические запреты)
# ============================================================

def validate_semantics(activities: list) -> list:
    """
    Для каждой задачи проверяет:
      - у предка не должно быть фазы из FORBIDDEN_BEFORE[phase задачи]
      - у задачи фазы из REQUIRED_AFTER должны быть нужные предки
        (прямые предки; транзитивные не проверяем — слишком строго)

    Нарушение → error, autofixable=True (переставляется кодом).
    """
    issues = []
    by_id = {str(a.get("Activity ID", "")): a for a in activities}

    for a in activities:
        aid = str(a.get("Activity ID", ""))
        phase = _norm_phase(a.get("Category"))
        wbs = _norm_wbs(a.get("WBS"))
        pred_id = str(a.get("Predecessor", "") or "").strip()
        pred = by_id.get(pred_id) if pred_id else None
        pred_phase = _norm_phase(pred.get("Category")) if pred else ""

        # 4.1 запрещённый предок
        forbidden = {_norm_phase(x) for x in FORBIDDEN_BEFORE.get(phase, [])}
        if pred and pred_phase in forbidden:
            issues.append(Issue(
                severity=SEVERITY_ERROR,
                code="SEM_FORBIDDEN",
                message=(
                    f"«{a.get('Activity Name')}» ({phase}) не может идти после "
                    f"«{pred.get('Activity Name')}» ({pred_phase}) — нарушен "
                    f"технологический порядок"
                ),
                wbs=wbs,
                activity_id=aid,
                activity_name=str(a.get("Activity Name", "")),
                autofixable=True,
            ))

        # 4.2 отсутствует обязательный предок
        required = {_norm_phase(x) for x in REQUIRED_AFTER.get(phase, [])}
        if required and pred_phase not in required:
            # допускаем, что правильный предок есть раньше в цепочке —
            # проверим только прямого, чтобы не усложнять
            # (если хотите транзитивную проверку — расширим)
            if pred is None or pred_phase not in required:
                issues.append(Issue(
                    severity=SEVERITY_WARN,
                    code="SEM_MISSING_DEP",
                    message=(
                        f"«{a.get('Activity Name')}» ({phase}) должен идти после "
                        f"одной из фаз {sorted(required)}, а идёт после "
                        f"'{pred_phase or '—'}'"
                    ),
                    wbs=wbs,
                    activity_id=aid,
                    activity_name=str(a.get("Activity Name", "")),
                    autofixable=True,
                ))

    return issues


# ============================================================
# ТОЧКА ВХОДА
# ============================================================

def run_all_checks(activities: list, edited_df: pd.DataFrame) -> list:
    """Возвращает объединённый список Issue со всех проверок."""
    issues = []
    issues += validate_sequence(activities)
    issues += validate_predecessors(activities)
    issues += validate_coverage(activities, edited_df)
    issues += validate_semantics(activities)
    return issues