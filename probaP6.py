"""
Мультиагентная система: Чертеж ➔ Смета ➔ Primavera P6
Агентная версия с tool-calling loop (ReAct-style).

Часть 1: импорты, конфиг, утилиты, инструменты.
Часть 2 (ниже в этом же файле): агентный цикл + Streamlit UI.
"""

import streamlit as st
import io
import base64
import json
import os
import hashlib
import re
from collections import Counter

from dotenv import load_dotenv
import pandas as pd
from openai import OpenAI
from pdf2image import convert_from_bytes
from PIL import Image

# ============================================================
# КОНФИГ
# ============================================================

load_dotenv()

st.set_page_config(page_title="Двухэтапный ИИ-Сметчик (агентный)", layout="wide")
st.title("🏗️ Мультиагентная система: Чертеж ➔ Смета ➔ Primavera P6")

POPPLER_PATH = os.getenv("POPPLER_PATH")

# ============================================================
# СПРАВОЧНИКИ
# ============================================================

WORK_RATES = {
    "concrete":      {"rate": 14.0,  "unit": "м3",  "name": "Укладка бетона"},
    "formwork":      {"rate": None,  "unit": "",    "name": "Монтаж опалубки"},
    "rebar":         {"rate": 500.0, "unit": "кг",  "name": "Армирование"},
    "insulation":    {"rate": 50.0,  "unit": "м2",  "name": "Гидроизоляция"},
    "waterproofing": {"rate": 50.0,  "unit": "м2",  "name": "Гидроизоляция"},
    "backfill":      {"rate": 20.0,  "unit": "м3",  "name": "Обратная засыпка/уплотнение"},
    "earthwork":     {"rate": 60.0,  "unit": "м3",  "name": "Земляные работы"},
    "demolition":    {"rate": 10.0,  "unit": "м3",  "name": "Демонтажные работы"},
    "pipeline":      {"rate": 30.0,  "unit": "м.п.", "name": "Монтаж трубопровода"},
    "installation":  {"rate": 3.0,   "unit": "шт",  "name": "Монтаж оборудования"},
    "finishing":     {"rate": 40.0,  "unit": "м2",  "name": "Отделочные работы"},
    "masonry":       {"rate": 5.0,   "unit": "м3",  "name": "Кладка"},
    "roofing":       {"rate": 30.0,  "unit": "м2",  "name": "Кровельные работы"},
    "electrical":    {"rate": 30.0,  "unit": "м.п.", "name": "Электромонтаж"},
    "hvac":          {"rate": 10.0,  "unit": "м.п.", "name": "ОВиК"},
    "general":       {"rate": 10.0,  "unit": "",    "name": "Работа"},
}

EXPECTED_UNIT = {
    "concrete": "м3", "formwork": "м2", "rebar": "кг", "insulation": "м2",
    "waterproofing": "м2", "backfill": "м3", "earthwork": "м3", "demolition": "м3",
    "pipeline": "м.п.", "installation": "шт", "finishing": "м2", "masonry": "м3",
    "roofing": "м2", "electrical": "м.п.", "hvac": "м.п.", "general": "",
}

UNIT_NORMALIZE = {
    "м³": "м3", "m3": "м3", "куб": "м3", "куб.м": "м3",
    "м²": "м2", "m2": "м2", "кв": "м2", "кв.м": "м2",
    "кг": "кг", "kg": "кг", "т": "т", "тн": "т", "ton": "т", "тонн": "т",
    "шт": "шт", "шт.": "шт", "pcs": "шт", "ед": "шт", "ед.": "шт",
    "м.п.": "м.п.", "мп": "м.п.", "пог.м": "м.п.", "пог.м.": "м.п.",
    "м.п": "м.п.", "м": "м.п.", "компл.": "компл.", "компл": "компл.",
}

PHASE_ORDER = {
    "demolition": 0, "earthwork": 1, "formwork": 2, "rebar": 3,
    "concrete": 4, "waterproofing": 5, "insulation": 5, "backfill": 6,
    "installation": 7, "pipeline": 7, "masonry": 8, "roofing": 9,
    "finishing": 10, "electrical": 11, "hvac": 11, "general": 99,
}

# ============================================================
# БАЗОВЫЕ УТИЛИТЫ (не тулы — просто хелперы)
# ============================================================

def encode_image(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("utf-8")


def shrink_image_to_limit(pil_img, max_side: int = 3500):
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    w, h = pil_img.size
    if w > max_side or h > max_side:
        scale = min(max_side / w, max_side / h)
        new_size = (int(w * scale), int(h * scale))
        return pil_img.resize(new_size, Image.Resampling.LANCZOS)
    return pil_img


def extract_json_array(text: str):
    if not text:
        raise ValueError("Пустой ответ модели")

    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)

    start = text.find("[")
    if start == -1:
        raise ValueError("В ответе модели нет JSON-массива")

    end = text.rfind("]") + 1
    if end > start:
        candidate = text[start:end]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    body = text[start:]
    objects = []
    depth = 0
    buf = []
    in_string = False
    escape = False

    for ch in body:
        if escape:
            buf.append(ch)
            escape = False
            continue
        if ch == "\\":
            buf.append(ch)
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            buf.append(ch)
            continue
        if in_string:
            buf.append(ch)
            continue

        if ch == "{":
            depth += 1
            buf.append(ch)
        elif ch == "}":
            depth -= 1
            buf.append(ch)
            if depth == 0 and buf:
                try:
                    objects.append(json.loads("".join(buf)))
                except json.JSONDecodeError:
                    pass
                buf = []
        elif depth > 0:
            buf.append(ch)

    if not objects:
        raise ValueError("Не удалось извлечь ни одного корректного объекта.")
    return objects


def normalize_unit(u) -> str:
    if u is None:
        return "м3"
    try:
        if pd.isna(u):
            return "м3"
    except (TypeError, ValueError):
        pass
    u = str(u).strip().lower()
    return UNIT_NORMALIZE.get(u, u or "м3")


def to_kg(vol: float, unit: str) -> float:
    return vol * 1000.0 if unit == "т" else vol


def clean_wbs(w) -> str:
    s = str(w or "").strip()
    if not s:
        return "Общие работы"
    parts = re.split(r"[/|—–:]", s)
    first = parts[0].strip()
    return first or s


def classify_work(element_name: str, unit: str):
    name = (element_name or "").lower()
    u = normalize_unit(unit)

    RULES = [
        (["демонтаж", "разборка", "снос"], ["м3", "м2", "шт", "т"], "demolition"),
        (["арматур", "сетк", "каркас", "стержн"], ["кг", "т"], "rebar"),
        (["землян", "котлован", "выемк", "разработк", "выторфовк"], ["м3"], "earthwork"),
        (["засыпк", "отсев", "уплотнен", "послойн"], ["м3"], "backfill"),
        (["бетон", "бетонир", "стяжк"], ["м3"], "concrete"),
        (["опалубк"], ["м2"], "formwork"),
        (["гидроизоляц", "обмазк", "мастик", "рубероид", "мембран",
          "праймер", "битум"], ["м2"], "waterproofing"),
        (["труб", "трубопровод", "воздуховод"], ["м.п.", "м", "мп"], "pipeline"),
        (["монтаж", "установк", "резервуар", "насос", "люк", "колодец",
          "дверь", "окно", "фиксатор", "болт"], ["шт", "компл."], "installation"),
        (["штукатур", "окраск", "покраск", "отделк"], ["м2"], "finishing"),
        (["кладк", "кирпич", "газобетон", "пеноблок"], ["м3"], "masonry"),
    ]

    for keywords, units, category in RULES:
        if any(k in name for k in keywords):
            if not units or u in units:
                return category
    return "general"


# ============================================================
# РЕАЛИЗАЦИЯ ТУЛОВ (для обоих агентов)
# ============================================================

def tool_render_pdf_pages(file_bytes: bytes, max_pages: int = 3,
                          dpi: int = 150) -> dict:
    """PDF -> список base64 JPEG."""
    try:
        last_page = None if max_pages == 0 else max_pages
        images = convert_from_bytes(
            file_bytes, dpi=dpi, first_page=1,
            last_page=last_page, poppler_path=POPPLER_PATH,
        )
        if not images:
            return {"ok": False, "error": "PDF не содержит страниц", "pages": []}

        out = []
        for img in images:
            img = shrink_image_to_limit(img)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=90)
            out.append(base64.b64encode(buf.getvalue()).decode("utf-8"))
        return {"ok": True, "pages_count": len(out), "pages": out}
    except Exception as e:
        return {"ok": False, "error": str(e), "pages": []}


def tool_validate_estimate(json_text: str) -> dict:
    """Проверяет JSON-массив ведомости."""
    try:
        data = extract_json_array(json_text)
    except Exception as e:
        return {"ok": False, "errors": [f"JSON parse error: {e}"], "rows_count": 0}

    errors = []
    allowed_units = {"м3", "м2", "кг", "т", "шт", "м.п.", "компл."}
    allowed_cats = set(WORK_RATES.keys())

    for i, row in enumerate(data):
        if not isinstance(row, dict):
            errors.append(f"row {i}: not an object")
            continue
        for f in ("wbs_parent", "element", "unit", "category"):
            if f not in row:
                errors.append(f"row {i}: missing field '{f}'")
        if row.get("unit") not in allowed_units:
            errors.append(f"row {i}: bad unit '{row.get('unit')}'")
        if row.get("category") not in allowed_cats:
            errors.append(f"row {i}: bad category '{row.get('category')}'")
        v = row.get("volume")
        if v is not None and not isinstance(v, (int, float)):
            errors.append(f"row {i}: volume not numeric ({v!r})")
        name = str(row.get("element", "")).lower()
        if re.search(r"\bитого\b|\bвсего\b", name):
            errors.append(f"row {i}: summary row not allowed")

    return {
        "ok": not errors,
        "errors": errors,
        "rows_count": len(data),
    }


def tool_classify_work(element_name: str, unit: str) -> dict:
    return {"category": classify_work(element_name, unit)}


def tool_compute_duration(phase: str, volume: float) -> dict:
    cfg = WORK_RATES.get(phase, WORK_RATES["general"])
    rate = cfg.get("rate")
    if volume is None or volume <= 0:
        return {"duration_days": 1}
    if rate is None:
        conc_rate = WORK_RATES["concrete"]["rate"]
        conc_dur = max(1, int(-(-volume // conc_rate)))
        return {"duration_days": max(1, int(round(conc_dur * 0.7)))}
    return {"duration_days": max(1, int(-(-volume // rate)))}


def tool_build_p6_schedule(estimate_json: str, planner_rows_json: str) -> dict:
    """
    Собирает таблицу Primavera P6 из:
      - estimate_json: JSON-массив ведомости объёмов
      - planner_rows_json: JSON-массив от Агента-Планировщика
    """
    try:
        est_rows = extract_json_array(estimate_json)
    except Exception as e:
        return {"ok": False, "errors": [f"estimate_json: {e}"], "activities": []}

    try:
        rows = extract_json_array(planner_rows_json)
    except Exception as e:
        return {"ok": False, "errors": [f"planner_rows_json: {e}"], "activities": []}

    if not est_rows:
        return {"ok": False, "errors": ["Пустая ведомость объёмов"], "activities": []}
    if not rows:
        return {"ok": False, "errors": ["Планировщик вернул пустой список работ"], "activities": []}

    edited_df = pd.DataFrame(est_rows)
    for col in ("wbs_parent", "element", "count", "volume",
                "unit", "category", "note", "source_page"):
        if col not in edited_df.columns:
            edited_df[col] = ""

    edited_df["volume"] = pd.to_numeric(edited_df["volume"], errors="coerce")
    edited_df["unit"] = edited_df["unit"].apply(normalize_unit)
    edited_df["category"] = (
        edited_df["category"].fillna("").astype(str).str.strip().str.lower()
    )
    mask_empty = edited_df["category"] == ""
    if mask_empty.any():
        edited_df.loc[mask_empty, "category"] = [
            classify_work(r["element"], r["unit"])
            for _, r in edited_df[mask_empty].iterrows()
        ]
    edited_df["wbs_parent"] = edited_df["wbs_parent"].apply(clean_wbs)

    known_wbs = set(str(x).strip() for x in edited_df["wbs_parent"].dropna().unique())

    def match_wbs(raw_wbs: str) -> str:
        raw = (raw_wbs or "").strip()
        if raw in known_wbs:
            return raw
        raw_l = raw.lower()
        candidates = [k for k in known_wbs if raw_l in k.lower() or k.lower() in raw_l]
        if candidates:
            return min(candidates, key=len)
        first_word = raw.split()[0] if raw.split() else ""
        if first_word:
            candidates = [k for k in known_wbs if first_word.lower() in k.lower()]
            if candidates:
                return min(candidates, key=len)
        return raw

    task_count_by_group = Counter(
        (
            match_wbs(str(x.get("wbs_parent", "")).strip()),
            str(x.get("phase", "general")).strip().lower(),
        )
        for x in rows
    )

    CATEGORY_ALIAS = {"insulation": "waterproofing", "waterproofing": "insulation"}

    volume_map = {}
    for _, r in edited_df.iterrows():
        wbs_key = str(r["wbs_parent"]).strip()
        cat_key = str(r["category"]).strip().lower()
        vol = float(r["volume"]) if pd.notna(r["volume"]) else 0.0
        unit = normalize_unit(str(r.get("unit", "")))

        if cat_key == "rebar" and unit == "т":
            vol = vol * 1000.0
            unit = "кг"

        item = {
            "volume": vol,
            "unit": unit,
            "element": str(r.get("element", "")).strip(),
        }
        volume_map.setdefault((wbs_key, cat_key), []).append(item)
        alias = CATEGORY_ALIAS.get(cat_key)
        if alias:
            volume_map.setdefault((wbs_key, alias), []).append(dict(item))

    activities = []
    aid_counter = 2000

    rows_sorted = sorted(
        rows,
        key=lambda x: (
            str(x.get("wbs_parent", "")),
            PHASE_ORDER.get(str(x.get("phase", "general")).lower(), 99),
        ),
    )

    prev_aid_by_wbs = {}

    for r in rows_sorted:
        raw_wbs = str(r.get("wbs_parent", "Общие работы")).strip()
        wbs = match_wbs(raw_wbs)
        task_name = str(r.get("task_name", "")).strip()
        phase = str(r.get("phase", "general")).strip().lower()

        expected_unit = EXPECTED_UNIT.get(phase, "")
        task_lower = task_name.lower()

        exact_matches = []
        for _, er in edited_df.iterrows():
            if str(er["wbs_parent"]).strip() != wbs:
                continue
            if str(er["category"]).strip().lower() != phase:
                continue

            el_name = str(er["element"]).lower()
            el_words = {w for w in re.findall(r"[a-zа-я0-9]+", el_name) if len(w) > 2}
            task_words = {w for w in re.findall(r"[a-zа-я0-9]+", task_lower) if len(w) > 2}
            common = el_words & task_words

            if common:
                exact_matches.append((len(common), er))

        if exact_matches:
            exact_matches.sort(key=lambda x: -x[0])
            best = exact_matches[0][1]
            found_unit = normalize_unit(str(best["unit"]))
            found_vol = float(best["volume"]) if pd.notna(best["volume"]) else 0.0

            if phase == "rebar":
                vol, unit = to_kg(found_vol, found_unit), "кг"
            else:
                vol, unit = found_vol, found_unit or expected_unit or "м3"
        else:
            items = volume_map.get((wbs, phase), [])
            vol, unit = 0.0, expected_unit or "м3"
            if items:
                if expected_unit:
                    matching = [it for it in items if it["unit"] == expected_unit]
                else:
                    cnt = Counter(it["unit"] for it in items)
                    main_unit = cnt.most_common(1)[0][0]
                    matching = [it for it in items if it["unit"] == main_unit]

                if matching:
                    if phase == "rebar":
                        total = sum(to_kg(it["volume"], it["unit"]) for it in matching)
                        unit = "кг"
                    else:
                        total = sum(it["volume"] for it in matching)
                        unit = matching[0]["unit"]

                    n_tasks = task_count_by_group.get((wbs, phase), 1)
                    if n_tasks == 0:
                        n_tasks = 1
                    vol = total / n_tasks

        if vol == 0.0:
            best_match = None
            best_score = 0
            for _, er in edited_df.iterrows():
                el_name = str(er["element"]).lower()
                if el_name and (el_name in task_lower or task_lower in el_name):
                    score = 100
                else:
                    words_task = {w for w in re.findall(r"\w+", task_lower) if len(w) > 2}
                    words_el = {w for w in re.findall(r"\w+", el_name) if len(w) > 2}
                    score = len(words_task & words_el)
                if score > best_score:
                    best_score = score
                    best_match = er

            if best_match is not None and best_score >= 1:
                found_unit = normalize_unit(str(best_match.get("unit", "")))
                found_vol = (
                    float(best_match["volume"])
                    if pd.notna(best_match["volume"]) else 0.0
                )

                if phase == "rebar":
                    vol = to_kg(found_vol, found_unit)
                    unit = "кг"
                elif expected_unit:
                    if found_unit == expected_unit:
                        vol = found_vol
                        unit = expected_unit
                    else:
                        vol, unit = 0.0, expected_unit
                else:
                    vol, unit = found_vol, found_unit or "м3"

        if vol <= 0:
            duration = 1
        else:
            cfg = WORK_RATES.get(phase, WORK_RATES["general"])
            rate = cfg.get("rate")
            if rate is None:
                conc_rate = WORK_RATES["concrete"]["rate"]
                conc_dur = max(1, int(-(-vol // conc_rate)))
                duration = max(1, int(round(conc_dur * 0.7)))
            else:
                duration = max(1, int(-(-vol // rate)))

        pred_raw = str(r.get("predecessor", "")).strip()
        if not pred_raw or pred_raw.lower() == "null":
            pred_id = ""
        elif pred_raw.upper() == "AUTO":
            pred_id = prev_aid_by_wbs.get(wbs, "")
        else:
            if re.match(r"^A\d+$", pred_raw):
                pred_id = pred_raw
            else:
                pred_id = ""
                for a in activities:
                    if pred_raw.lower() in a["Activity Name"].lower():
                        pred_id = a["Activity ID"]
                        break

        current_aid = f"A{aid_counter}"
        activities.append({
            "Activity ID": current_aid,
            "WBS": wbs,
            "Activity Name": task_name,
            "Original Duration": duration,
            "Activity Status": "Not Started",
            "Category": phase,
            "Predecessor": pred_id,
            "Volume": vol,
            "Unit": unit,
        })
        prev_aid_by_wbs[wbs] = current_aid
        aid_counter += 10

    if not activities:
        return {
            "ok": False,
            "errors": ["Не удалось сопоставить элементы с работами"],
            "activities": [],
        }

    return {"ok": True, "errors": [], "activities": activities}
# ============================================================
# АГЕНТНЫЙ ЦИКЛ (ReAct-style)
# ============================================================

def run_agent(client, model, system_prompt, user_content,
              tools, tool_impl, max_steps=8, max_tokens=4000):
    """
    Цикл: модель вызывает инструменты, получает результаты,
    корректируется. Возвращает (final_text, trace).
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    trace = []

    for step in range(max_steps):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=0,
                max_tokens=max_tokens,
            )
        except Exception as e:
            trace.append({"step": step, "type": "api_error", "error": str(e)})
            return None, trace

        if isinstance(resp, str):
            trace.append({"step": step, "type": "raw_string", "content": resp})
            return resp, trace

        choice = resp.choices[0]
        msg = choice.message
        finish = getattr(choice, "finish_reason", None)

        if finish == "length":
            trace.append({"step": step, "type": "length_limit",
                          "content": msg.content})
            return None, trace

        if finish != "tool_calls" or not getattr(msg, "tool_calls", None):
            trace.append({"step": step, "type": "final", "content": msg.content})
            return msg.content, trace

        messages.append({
            "role": "assistant",
            "content": msg.content,
            "tool_calls": [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name,
                              "arguments": tc.function.arguments}}
                for tc in msg.tool_calls
            ],
        })

        for tc in msg.tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}

            fn = tool_impl.get(name)
            if fn is None:
                result = {"ok": False, "error": f"unknown tool {name}"}
            else:
                try:
                    result = fn(**args)
                except Exception as e:
                    result = {"ok": False, "error": str(e)}

            # Обрезаем pages в трейсе для читаемости UI
            trace_result = dict(result) if isinstance(result, dict) else result
            if isinstance(trace_result, dict) and "pages" in trace_result:
                trace_result = dict(trace_result)
                trace_result["pages"] = f"<{len(trace_result['pages'])} страниц>"

            trace.append({"step": step, "type": "tool",
                          "name": name, "args": args, "result": trace_result})

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result, ensure_ascii=False, default=str),
            })

    trace.append({"step": max_steps, "type": "max_steps_reached"})
    return None, trace


# ============================================================
# СПЕЦИФИКАЦИИ ТУЛОВ ДЛЯ МОДЕЛИ
# ============================================================

AGENT1_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "validate_estimate",
            "description": (
                "Проверить JSON-массив ведомости на корректность: наличие полей, "
                "допустимые unit/category, отсутствие строк 'Итого'/'Всего', "
                "числовые volume. Вызывай после каждого черновика ведомости. "
                "Если ok=false — прочитай errors, исправь JSON и повтори."
            ),
            "parameters": {
                "type": "object",
                "properties": {"json_text": {"type": "string"}},
                "required": ["json_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "classify_work",
            "description": (
                "Определить категорию работы по названию элемента и единице измерения. "
                "Используй, если сомневаешься в category."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "element_name": {"type": "string"},
                    "unit": {"type": "string"},
                },
                "required": ["element_name", "unit"],
            },
        },
    },
]

AGENT1_TOOL_IMPL = {
    "validate_estimate": tool_validate_estimate,
    "classify_work": tool_classify_work,
}



AGENT2_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "compute_duration",
            "description": (
                "Рассчитать длительность задачи в днях по категории и объёму. "
                "Вызывай для проверки своих прикидок, если сомневаешься."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "phase": {"type": "string"},
                    "volume": {"type": "number"},
                },
                "required": ["phase", "volume"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "build_p6_schedule",
            "description": (
                "Собрать итоговую таблицу Primavera P6 из ведомости объёмов "
                "и твоего черновика WBS. Вызывай ОДИН раз, когда черновик готов. "
                "Если вернётся ok=false — прочитай errors, исправь planner_rows_json "
                "и вызови снова."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "estimate_json": {
                        "type": "string",
                        "description": "JSON-массив ведомости объёмов (как во входе).",
                    },
                    "planner_rows_json": {
                        "type": "string",
                        "description": (
                            "JSON-массив твоих задач: "
                            "[{wbs_parent, task_name, phase, predecessor}, ...]"
                        ),
                    },
                },
                "required": ["estimate_json", "planner_rows_json"],
            },
        },
    },
]

AGENT2_TOOL_IMPL = {
    "compute_duration": tool_compute_duration,
    "build_p6_schedule": tool_build_p6_schedule,
}

# ============================================================
# ПРОМПТЫ
# ============================================================

VISION_PROMPT = """Ты — Агент-Замерщик, эксперт по чтению ЛЮБЫХ строительных чертежей (КЖ, КМ, АР, ГП, ЭМ, ВК, ОВ и др.).
Твоя задача — составить полную ведомость всего, что можно посчитать на листах, с привязкой к конструктивам.

ПОРЯДОК РАБОТЫ:
1) Изучи изображения страниц, приложенные в user-сообщении.
   Если изображение нечитаемо (мелкий текст, обрезка, пустой лист) —
   верни пустой JSON-массив [] и в trace это будет видно.
2) Ищи на листах:
   - Таблицы "Спецификация", "Спецификация фундамента", "Ведомость расхода стали", "Ведомость материалов".
   - Если позиции в таблицах нет — геометрические размеры на планах/разрезах.
   - Если нет и размеров — текстовые указания в примечаниях (площади, толщины слоёв, классы уплотнения).
   - Штучные элементы (люки, двери, окна, насосы, резервуары, фиксаторы, болты).
3) Составь ЧЕРНОВИК JSON-массива ведомости.
4) Вызови validate_estimate. Если ok=false — исправь по errors и повтори с шага 3.
5) Сомнительные категории проверяй через classify_work.
6) Только когда validate_estimate вернул ok=true — отдай финальный JSON-массив.

ПРАВИЛА ПРИОРИТЕТА:
- Если позиция ЕСТЬ в спецификации/ведомости — бери объём ТОЛЬКО оттуда, НЕ пересчитывай по габаритам.
- Если в ведомости стали несколько позиций арматуры — выдай КАЖДУЮ отдельной строкой.
- Если объёма нет ни в таблице, ни в размерах — volume = null и в note напиши, чего не хватает.
- НЕ ОБЪЕДИНЯЙ разные материалы в одну строку.
- НЕ ПРОПУСКАЙ: демонтаж, уплотнение, вывоз, монтаж, пусконаладку.

ЗАПРЕЩЕНО:
- Дублировать данные (Спецификация vs Ведомость стали).
- Создавать строки "Итого", "Всего".

ПОЛЕ wbs_parent: ТОЛЬКО марка конструктива (например, "Фундамент Ф4").
ПОЛЕ unit — строго одно из: "м3", "м2", "кг", "т", "шт", "м.п.", "компл.".
ПОЛЕ category — одно из: earthwork, formwork, rebar, concrete, insulation, waterproofing, backfill, installation, demolition, pipeline, finishing, masonry, roofing, electrical, hvac, general.

ФОРМАТ ОТВЕТА — строго JSON-массив объектов с полями:
wbs_parent, element, count, volume, unit, category, source_page, note.
Не добавляй markdown, не пиши пояснений до или после JSON."""


PLANNER_SYSTEM_PROMPT = """Ты — Агент-Планировщик Primavera P6.

ТВОЙ ЦИКЛ РАБОТЫ:
1. Прочитай ведомость объёмов (передана в user-сообщении).
2. Составь ЧЕРНОВИК задач в виде JSON-массива:
   [{"wbs_parent": "...", "task_name": "...", "phase": "...", "predecessor": null|"AUTO"}, ...]
3. Вызови инструмент build_p6_schedule с двумя аргументами:
   - estimate_json: JSON-строка ведомости (скопируй её как есть)
   - planner_rows_json: JSON-строка твоего черновика
4. Если build_p6_schedule вернул {"ok": false, "errors": [...]} —
   прочитай errors, исправь черновик и вызови снова.
5. Когда build_p6_schedule вернул {"ok": true, "activities": [...]} —
   отдай финальный ответ: краткое резюме (сколько задач, по каким WBS)
   и JSON-массив activities.

ПРАВИЛА ПОСТРОЕНИЯ ЧЕРНОВИКА:
- wbs_parent бери БУКВАЛЬНО из ведомости.
- phase — одно из: earthwork, formwork, rebar, concrete, insulation,
  waterproofing, backfill, installation, demolition, pipeline, finishing,
  masonry, roofing, electrical, hvac, general.
- predecessor: null если первая задача WBS, иначе "AUTO".
- Порядок фаз: earthwork → formwork → rebar → concrete →
  waterproofing/insulation → backfill → installation → finishing.
- Арматура: ОДНА задача на WBS, phase="rebar".
- Бетон: ДВЕ задачи (подготовка + конструктив), обе phase="concrete".
- Обратная засыпка: ОДНА задача "Обратная засыпка с уплотнением".
- Если volume=null — в task_name добавь "(объём уточняется)".
- НЕ создавай строки "Итого"/"Всего".

ОПРЕДЕЛЕНИЕ phase ПО НАЗВАНИЮ:
- "арматур", "армирован", "стержн", "сетк" → rebar
- "бетон" → concrete
- "опалубк" → formwork
- "фиксатор", "болт", "люк", "колодец", "насос", "дверь", "окно" → installation
- "гидроизоляц", "обмазк", "мастик", "праймер", "битум" → waterproofing
- "засыпк", "отсев", "уплотнен" → backfill
- "котлован", "разработк", "землян" → earthwork

Не пиши markdown-обёрток вокруг JSON. Не объясняй вызовы инструментов."""


# ============================================================
# SIDEBAR + АВТОРИЗАЦИЯ
# ============================================================

st.sidebar.header("🔑 Авторизация VseGPT")

env_key = (os.getenv("API_KEY") or os.getenv("POPPLER_PATH") or "").strip()

if env_key:
    api_key = env_key
    st.sidebar.success("✅ Ключ успешно загружен из файла .env")
else:
    api_key = st.sidebar.text_input("Введите API Ключ вручную:", type="password").strip()

model_vision = st.sidebar.selectbox(
    "Агент #1 (Зрение):",
    ["anthropic/claude-sonnet-5.5-high", "openai/gpt-4o"],
)
model_planner = st.sidebar.selectbox(
    "Агент #2 (Планировщик):",
    ["openai/gpt-4o", "anthropic/claude-sonnet-5.5-high"],
)

max_pages = st.sidebar.number_input(
    "Сколько страниц PDF читать (0 = все):", min_value=0, value=3, step=1
)

max_steps = st.sidebar.number_input(
    "Макс. шагов агентного цикла:", min_value=2, value=8, step=1
)

if not api_key:
    st.warning("Пожалуйста, добавьте API_KEY в файл .env или введите его на боковой панели.")
    st.stop()

client = OpenAI(
    api_key=api_key,
    base_url="https://api.vsegpt.ru/v1",
    timeout=180,
    max_retries=2,
)

# ============================================================
# SESSION STATE
# ============================================================

if "raw_estimate_df" not in st.session_state:
    st.session_state.raw_estimate_df = None
if "p6_df" not in st.session_state:
    st.session_state.p6_df = None
if "last_file_hash" not in st.session_state:
    st.session_state.last_file_hash = None
if "agent1_trace" not in st.session_state:
    st.session_state.agent1_trace = None
if "agent2_trace" not in st.session_state:
    st.session_state.agent2_trace = None


# ============================================================
# ШАГ 1 — АГЕНТ-ЗАМЕРЩИК
# ============================================================

st.header("1️⃣ Шаг 1: Агент-Замерщик (Сбор объемов с чертежа)")

uploaded_file = st.file_uploader(
    "Загрузите скан спецификации/чертежа (PDF, PNG, JPG)",
    type=["pdf", "png", "jpg", "jpeg"],
    key="uploader",
)

if uploaded_file:
    file_bytes = uploaded_file.read()
    file_hash = hashlib.md5(file_bytes).hexdigest()

    if file_hash != st.session_state.last_file_hash:
        st.session_state.last_file_hash = file_hash
        st.session_state.raw_estimate_df = None
        st.session_state.p6_df = None
        st.session_state.agent1_trace = None
        st.session_state.agent2_trace = None

    if st.button("🔍 Запустить Агента #1: Собрать объемы"):
        with st.spinner("Агент #1: цикл анализа чертежа..."):

            # --- 1) Хост-код готовит изображения страниц ---
            user_content = [{"type": "text", "text": VISION_PROMPT}]
            render_error = None

            try:
                if uploaded_file.type == "application/pdf":
                    last_page = None if max_pages == 0 else int(max_pages)
                    images = convert_from_bytes(
                        file_bytes,
                        dpi=200,  # ↑ подняли DPI
                        first_page=1,
                        last_page=last_page,
                        poppler_path=POPPLER_PATH,
                    )
                    if not images:
                        raise ValueError("PDF не содержит страниц")

                    for page_idx, page_img in enumerate(images, start=1):
                        page_img = shrink_image_to_limit(page_img, max_side=3500)
                        buf = io.BytesIO()
                        page_img.save(buf, format="JPEG", quality=92)
                        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                        user_content.append({
                            "type": "text",
                            "text": f"Страница {page_idx}:",
                        })
                        user_content.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                        })
                        del page_img
                else:
                    # PNG/JPG — отдаём как есть
                    mime = uploaded_file.type or "image/jpeg"
                    user_content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime};base64,{encode_image(file_bytes)}"
                        },
                    })
            except Exception as e:
                render_error = str(e)

            if render_error:
                st.error(f"Не удалось подготовить изображения страниц: {render_error}")
                st.stop()

            # --- 2) Запускаем агентный цикл (без render_pdf_pages) ---
            final_text, trace = run_agent(
                client, model_vision,
                system_prompt="Ты — Агент-Замерщик. Работай строго по своему циклу.",
                user_content=user_content,
                tools=AGENT1_TOOLS,
                tool_impl=AGENT1_TOOL_IMPL,
                max_steps=max_steps,
                max_tokens=8000,
            )

            st.session_state.agent1_trace = trace

            if not final_text:
                st.error("Агент #1 не завершил цикл. Смотрите трассировку ниже.")
            else:
                try:
                    data_vision = extract_json_array(final_text)
                    if not data_vision:
                        st.warning("🤖 Агент #1 не нашёл конструкций на чертеже.")
                        st.code(final_text)
                    else:
                        df = pd.DataFrame(data_vision)

                        if "volume" in df.columns:
                            df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
                        else:
                            df["volume"] = None

                        for col in ("wbs_parent", "element", "count", "unit",
                                    "category", "note", "source_page"):
                            if col not in df.columns:
                                df[col] = ""

                        df["wbs_parent"] = df["wbs_parent"].apply(clean_wbs)
                        df["unit"] = df["unit"].apply(normalize_unit)

                        df["category"] = (
                            df["category"].fillna("").astype(str)
                            .str.strip().str.lower()
                        )
                        mask_empty = df["category"] == ""
                        if mask_empty.any():
                            df.loc[mask_empty, "category"] = [
                                classify_work(r["element"], r["unit"])
                                for _, r in df[mask_empty].iterrows()
                            ]

                        df = df[
                            ~df["element"].astype(str).str.lower().str.contains(
                                r"\bитого\b|\bвсего\b", regex=True, na=False
                            )
                        ].reset_index(drop=True)

                        st.session_state.raw_estimate_df = df
                        st.session_state.p6_df = None
                        st.success("🤖 Агент #1 успешно сформировал ведомость объёмов!")
                except Exception as e:
                    st.error(f"Не удалось распарсить финальный JSON: {e}")
                    st.code(final_text)

        if st.session_state.agent1_trace:
            with st.expander("🧠 Трассировка Агента #1"):
                st.json(st.session_state.agent1_trace)


# ============================================================
# РЕДАКТОР ВЕДОМОСТИ
# ============================================================

if st.session_state.raw_estimate_df is not None:
    st.subheader("📋 Ведомость объемов (можно отредактировать перед шагом 2)")
    edited_df = st.data_editor(
        st.session_state.raw_estimate_df,
        num_rows="dynamic",
        key="editor_volumes",
    )

    edited_df = edited_df[
        edited_df["element"].notna()
        & (edited_df["element"].astype(str).str.strip() != "")
    ]
    edited_df = edited_df[
        edited_df["volume"].isna()
        | (pd.to_numeric(edited_df["volume"], errors="coerce") > 0)
    ]

    edited_df["unit"] = edited_df["unit"].apply(normalize_unit)
    edited_df["category"] = (
        edited_df["category"].fillna("").astype(str).str.strip().str.lower()
    )
    mask_empty = edited_df["category"] == ""
    if mask_empty.any():
        edited_df.loc[mask_empty, "category"] = [
            classify_work(r["element"], r["unit"])
            for _, r in edited_df[mask_empty].iterrows()
        ]

    edited_df["wbs_parent"] = edited_df["wbs_parent"].apply(clean_wbs)
    st.session_state.raw_estimate_df = edited_df

    estimate_buffer = io.BytesIO()
    edited_df.to_excel(estimate_buffer, index=False, engine="openpyxl")
    st.download_button(
        label="📥 Скачать ведомость объемов (Excel)",
        data=estimate_buffer.getvalue(),
        file_name="Ведомость_объемов_WBS.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    # ============================================================
    # ШАГ 2 — АГЕНТ-ПЛАНИРОВЩИК
    # ============================================================

    st.markdown("---")
    st.header("2️⃣ Шаг 2: Агент-Планировщик (Primavera P6)")

    if st.button("📊 Запустить Агента #2: Создать структуру Primavera"):
        with st.spinner("Агент #2: цикл планирования и валидации..."):
            cols_for_llm = ["wbs_parent", "element", "volume",
                            "unit", "category", "note"]
            estimate_json_str = edited_df[cols_for_llm].to_json(
                orient="records", force_ascii=False
            )
            excel_text_context = edited_df[cols_for_llm].to_string(index=False)

            user_content = (
                "ВЕДОМОСТЬ ОБЪЁМОВ (JSON):\n"
                f"{estimate_json_str}\n\n"
                "ТА ЖЕ ВЕДОМОСТЬ (для читаемости):\n"
                f"{excel_text_context}\n\n"
                "Составь черновик WBS и вызови build_p6_schedule."
            )

            final_text, trace = run_agent(
                client, model_planner,
                system_prompt=PLANNER_SYSTEM_PROMPT,
                user_content=user_content,
                tools=AGENT2_TOOLS,
                tool_impl=AGENT2_TOOL_IMPL,
                max_steps=max_steps,
                max_tokens=8000,
            )

            st.session_state.agent2_trace = trace

            # Достаём последний успешный build_p6_schedule из trace
            activities = None
            for step_info in reversed(trace):
                if (step_info.get("type") == "tool"
                        and step_info.get("name") == "build_p6_schedule"):
                    res = step_info.get("result") or {}
                    if isinstance(res, dict) and res.get("ok") and res.get("activities"):
                        activities = res["activities"]
                        break

            if activities:
                st.session_state.p6_df = pd.DataFrame(activities)
                st.success(f"✅ Агент #2 собрал {len(activities)} задач")
            else:
                st.error("Агент #2 не смог собрать валидное расписание.")
                if final_text:
                    st.code(final_text)

        if st.session_state.agent2_trace:
            with st.expander("🧠 Трассировка Агента #2"):
                st.json(st.session_state.agent2_trace)

    # ============================================================
    # РЕЗУЛЬТАТ
    # ============================================================

    if st.session_state.p6_df is not None:
        st.success("🎯 Структура Primavera P6 сгенерирована!")
        st.subheader("🚀 Итоговая структура задач:")
        st.dataframe(st.session_state.p6_df, use_container_width=True)

        p6_buffer = io.BytesIO()
        with pd.ExcelWriter(p6_buffer, engine="openpyxl") as writer:
            st.session_state.p6_df.to_excel(
                writer, index=False, sheet_name="TASK"
            )

        st.download_button(
            label="📥 Скачать черновик для импорта в Primavera P6 (Excel)",
            data=p6_buffer.getvalue(),
            file_name="Primavera_P6_Import_Final.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        st.caption(
            "⚠️ Формат — черновик для маппинга колонок в P6. "
            "Для прямого импорта понадобится XER/XML-экспорт."
        )