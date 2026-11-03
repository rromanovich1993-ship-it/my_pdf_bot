import streamlit as st
import io
import base64
import json
import os
import hashlib
import re
from dotenv import load_dotenv
import pandas as pd
from openai import OpenAI
from pdf2image import convert_from_bytes
from PIL import Image

# Загружаем переменные окружения
load_dotenv()

st.set_page_config(page_title="Двухэтапный ИИ-Сметчик", layout="wide")
st.title("🏗️ Мультиагентная система: Чертеж ➔ Смета ➔ Primavera P6")

POPPLER_PATH = os.getenv("POPPLER_PATH")

st.sidebar.header("🔑 Авторизация VseGPT")

env_key = (os.getenv("API_KEY") or os.getenv("VSEGPT_API_KEY") or "").strip()

if env_key:
    api_key = env_key
    st.sidebar.success("✅ Ключ успешно загружен из файла .env")
else:
    api_key = st.sidebar.text_input("Введите API Ключ вручную:", type="password").strip()

model_vision = st.sidebar.selectbox(
    "Агент #1 (Зрение):", ["anthropic/claude-sonnet-5.5-high", "openai/gpt-4o"]
)
model_planner = st.sidebar.selectbox(
    "Агент #2 (Планировщик):", ["openai/gpt-4o", "anthropic/claude-sonnet-5.5-high"]
)

max_pages = st.sidebar.number_input(
    "Сколько страниц PDF читать (0 = все):", min_value=0, value=3, step=1
)

if not api_key:
    st.warning("Пожалуйста, добавьте API_KEY в файл .env или введите его на боковой панели.")
    st.stop()

client = OpenAI(
    api_key=api_key,
    base_url="https://api.vsegpt.ru/v1",
    timeout=120,
    max_retries=2,
)


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


def call_model(model: str, messages, max_tokens: int = 16000):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
        max_tokens=max_tokens,
    )
    if isinstance(response, str):
        return response.strip()

    choice = response.choices[0]
    content = (choice.message.content or "").strip()

    finish_reason = getattr(choice, "finish_reason", None)
    if finish_reason == "length":
        raise RuntimeError(
            f"Ответ модели оборван по лимиту max_tokens={max_tokens}."
        )

    return content


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


PHASE_ORDER = {
    "demolition": 0, "earthwork": 1, "formwork": 2, "rebar": 3,
    "concrete": 4, "waterproofing": 5, "insulation": 5, "backfill": 6,
    "installation": 7, "pipeline": 7, "masonry": 8, "roofing": 9,
    "finishing": 10, "electrical": 11, "hvac": 11, "general": 99,
}


if "raw_estimate_df" not in st.session_state:
    st.session_state.raw_estimate_df = None
if "p6_df" not in st.session_state:
    st.session_state.p6_df = None
if "last_file_hash" not in st.session_state:
    st.session_state.last_file_hash = None


VISION_PROMPT = """Ты — Агент-Замерщик, эксперт по чтению ЛЮБЫХ строительных чертежей (КЖ, КМ, АР, ГП, ЭМ, ВК, ОВ и др.).
Твоя задача — составить полную ведомость всего, что можно посчитать на листах, с привязкой к конструктивам.

Твоя задача — СТРОГО извлечь данные из таблиц и текстовых указаний на листе.

ЧТО ИСКАТЬ (только это!):
1) Таблицы: "Спецификация", "Ведомость расхода стали", "Ведомость материалов".
2) Текстовые указания в примечаниях (например: "Гидроизоляция - 2 слоя", "Бетон В25").
3) Штучные элементы, если они прямо указаны в тексте (например: "Люк - 5 шт").

ЧТО ИГНОРИРОВАТЬ (ни при каких условиях!):
- Геометрические размеры на планах и разрезах (не считай площади по клеточкам).
- Штриховки, оси, выноски, размерные линии.
- Мелкие детали, которых нет в спецификации.

ПРАВИЛА:
- Бери данные ТОЛЬКО из текста и таблиц.
- Если в таблице есть объем и единица измерения — бери их оттуда, ничего не пересчитывай!
- Если в таблице нет объема — volume = null.
- Поле "note" — максимум 3 слова (например: "слой 2 мм").
- НЕ создавай строки "Итого" и "Всего".
- Если на листе нет спецификации или таблиц — верни пустой массив [].

ПОЛЕ wbs_parent: Марка конструктива (например, "Фундамент Ф4").
ПОЛЕ unit — строго одно из: "м3", "м2", "кг", "т", "шт", "м.п.", "компл.".
ПОЛЕ category — строго одно из: earthwork, formwork, rebar, concrete, insulation, waterproofing, backfill, installation, demolition, pipeline, finishing, masonry, roofing, electrical, hvac, general.

ФОРМАТ ОТВЕТА — строго JSON-массив объектов:
wbs_parent, element, count, volume, unit, category, source_page, note.
Никакого markdown, никаких пояснений до или после JSON."""


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

    if st.button("🔍 Запустить Агента #1: Собрать объемы"):
        with st.spinner("Агент #1 изучает чертеж и составляет ведомость объемов..."):
            ai_output = None
            try:
                content = [{"type": "text", "text": VISION_PROMPT}]

                if uploaded_file.type == "application/pdf":
                    last_page = None if max_pages == 0 else max_pages
                    images = convert_from_bytes(
                        file_bytes, dpi=250, first_page=1, last_page=last_page,
                        poppler_path=POPPLER_PATH,
                    )
                    if not images:
                        raise ValueError("PDF не содержит страниц")

                    for page_img in images:
                        page_img = shrink_image_to_limit(page_img)
                        buf = io.BytesIO()
                        page_img.save(buf, format="JPEG", quality=90)
                        content.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{encode_image(buf.getvalue())}"},
                        })
                        del page_img
                else:
                    mime = uploaded_file.type or "image/jpeg"
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{encode_image(file_bytes)}"},
                    })

                ai_output = call_model(model_vision, messages=[{"role": "user", "content": content}])
                data_vision = extract_json_array(ai_output)

                if not data_vision:
                    st.warning("🤖 Агент #1 не нашёл конструкций на чертеже.")
                    st.code(ai_output)
                else:
                    df = pd.DataFrame(data_vision)

                    if "volume" in df.columns:
                        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
                    else:
                        df["volume"] = None

                    for col in ("wbs_parent", "element", "count", "unit", "category", "note", "source_page"):
                        if col not in df.columns:
                            df[col] = ""

                    df["wbs_parent"] = df["wbs_parent"].apply(clean_wbs)
                    df["unit"] = df["unit"].apply(normalize_unit)

                    df["category"] = df["category"].fillna("").astype(str).str.strip().str.lower()
                    mask_empty = df["category"] == ""
                    if mask_empty.any():
                        df.loc[mask_empty, "category"] = [
                            classify_work(r["element"], r["unit"]) for _, r in df[mask_empty].iterrows()
                        ]

                    df = df[
                        ~df["element"].astype(str).str.lower().str.contains(r"\bитого\b|\bвсего\b", regex=True, na=False)
                    ].reset_index(drop=True)

                    st.session_state.raw_estimate_df = df
                    st.session_state.p6_df = None
                    st.success("🤖 Агент #1 успешно сформировал ведомость объемов работ!")

            except Exception as e:
                st.error(f"Ошибка Агента #1: {e}")
                if ai_output:
                    st.code(ai_output)


if st.session_state.raw_estimate_df is not None:
    st.subheader("📋 Ведомость объемов (можно отредактировать перед шагом 2)")
    edited_df = st.data_editor(st.session_state.raw_estimate_df, num_rows="dynamic", key="editor_volumes")

    edited_df = edited_df[
        edited_df["element"].notna() & (edited_df["element"].astype(str).str.strip() != "")
    ]
    edited_df = edited_df[
        edited_df["volume"].isna() | (pd.to_numeric(edited_df["volume"], errors="coerce") > 0)
    ]

    edited_df["unit"] = edited_df["unit"].apply(normalize_unit)
    edited_df["category"] = edited_df["category"].fillna("").astype(str).str.strip().str.lower()
    mask_empty = edited_df["category"] == ""
    if mask_empty.any():
        edited_df.loc[mask_empty, "category"] = [
            classify_work(r["element"], r["unit"]) for _, r in edited_df[mask_empty].iterrows()
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

    st.markdown("---")
    st.header("2️⃣ Шаг 2: Агент-Планировщик (Конвертация данных в Primavera P6)")
    if st.button("📊 Запустить Агента #2: Создать структуру Primavera"):
        with st.spinner("Агент #2 формирует WBS и технологические циклы..."):
            cols_for_llm = ["wbs_parent", "element", "volume", "unit", "category", "note"]
            excel_text_context = edited_df[cols_for_llm].to_string(index=False)
            p6_output = None

            try:
                p6_output = call_model(
                    model_planner,
                    messages=[
                        {
                            "role": "user",
                            "content": f"""Ты — Агент-Планировщик Primavera P6.
    Изучи ведомость объёмов:

    {excel_text_context}

    Твоя задача — сформировать иерархическую структуру работ (WBS).
    Для КАЖДОГО конструктива (wbs_parent) создай набор задач в ТЕХНОЛОГИЧЕСКОЙ ПОСЛЕДОВАТЕЛЬНОСТИ.

    КРИТИЧЕСКИ ВАЖНО:
    1. Поле "wbs_parent" бери БУКВАЛЬНО из таблицы выше.
    2. Поле "phase" — тип работы (earthwork, formwork, rebar, concrete, insulation, waterproofing, backfill, installation, demolition, pipeline, finishing, masonry, roofing, electrical, hvac, general).
    3. Поле "predecessor" — null, если задача первая, "AUTO", если предшественник есть.

    ПРАВИЛА ОПРЕДЕЛЕНИЯ phase:
    - Если есть "арматур", "армирован", "стержн", "сетк" — phase="rebar".
    - Если есть "бетон" — phase="concrete".
    - Если "опалубк" — phase="formwork".
    - Если "фиксатор", "болт", "люк", "колодец", "насос", "дверь", "окно" — phase="installation".
    - Если "гидроизоляц", "обмазк", "мастик", "праймер", "битум" — phase="waterproofing".
    - Если "засыпк", "отсев", "уплотнен" — phase="backfill".
    - Если "котлован", "разработк", "землян" — phase="earthwork".

    6. НЕ ДРОБИ одну категорию на несколько задач:
       - Арматура: ОДНА задача "Армирование <WBS>" с phase="rebar".
       - Бетон: ДВЕ задачи (B10 подготовка + B30 конструктив).
       - Обратная засыпка: ОДНА задача "Обратная засыпка с уплотнением".
    7. Если volume = null — создай задачу, в task_name добавь "(объём уточняется)".
    8. ВСЮ арматуру на один WBS ОБЪЕДИНЯЙ в ОДНУ задачу "Монтаж арматуры <WBS>".

    ЛОГИЧЕСКИЙ ПОРЯДОК:
    earthwork → formwork → rebar → concrete → insulation/waterproofing → backfill → installation → finishing.

    Верни строго JSON-массив. Пример:
    [
      {{"wbs_parent": "Фундамент Ф4", "task_name": "Разработка котлована", "phase": "earthwork", "predecessor": null}},
      {{"wbs_parent": "Фундамент Ф4", "task_name": "Укладка бетона В10 (подготовка)", "phase": "concrete", "predecessor": "AUTO"}},
      {{"wbs_parent": "Фундамент Ф4", "task_name": "Монтаж арматуры А500С", "phase": "rebar", "predecessor": "AUTO"}},
      {{"wbs_parent": "Фундамент Ф4", "task_name": "Укладка бетона В30", "phase": "concrete", "predecessor": "AUTO"}},
      {{"wbs_parent": "Фундамент Ф4", "task_name": "Гидроизоляция боковых поверхностей", "phase": "waterproofing", "predecessor": "AUTO"}},
      {{"wbs_parent": "Фундамент Ф4", "task_name": "Обратная засыпка отсевом", "phase": "backfill", "predecessor": "AUTO"}}
    ]""",
                        }
                    ],
                )

                rows = extract_json_array(p6_output)
                if not rows:
                    raise ValueError("Агент #2 вернул пустой список работ")

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


                from collections import Counter

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

                    # 1) Точное сопоставление
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
                        # 2) Fallback на агрегат с делением
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

                    # 3) Дополнительный fallback
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
                            found_vol = float(best_match["volume"]) if pd.notna(best_match["volume"]) else 0.0

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

                    # Расчет длительности
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

                    # Предшественники
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
                    raise ValueError("Не удалось сопоставить элементы с работами.")

                st.session_state.p6_df = pd.DataFrame(activities)

            except Exception as e:
                st.error(f"Ошибка Агента #2: {e}")
                if p6_output:
                    st.code(p6_output)

    if st.session_state.p6_df is not None:
        st.success("🎯 Структура Primavera P6 сгенерирована!")
        st.subheader("🚀 Итоговая структура задач:")
        st.dataframe(st.session_state.p6_df, use_container_width=True)

        p6_buffer = io.BytesIO()
        with pd.ExcelWriter(p6_buffer, engine="openpyxl") as writer:
            st.session_state.p6_df.to_excel(writer, index=False, sheet_name="TASK")

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
#streamlit run one_test.py