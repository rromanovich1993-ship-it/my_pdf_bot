"""
Мультиагентная система: Чертеж ➔ Смета ➔ Primavera P6 (с Агентом #3).

Файлы проекта:
    norms.py          — база норм-часов
    unit_convert.py   — конвертация единиц
    validator.py      — проверки Агента #3
    agent3.py         — автофикс + нормы + отчёт
    app.py            — UI (этот файл)
"""
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

from norms import load_norms
from unit_convert import normalize_unit, convert_volume
from validator import Issue, PHASE_ORDER
from agent3 import run_agent3, build_llm_fix_prompt, Agent3Result


load_dotenv()

st.set_page_config(page_title="ИИ-Сметчик + Агент #3", layout="wide")
st.title("🏗️ Мультиагентная система: Чертеж ➔ Смета ➔ Primavera P6")

POPPLER_PATH = os.getenv("POPPLER_PATH")


# ============================================================
# SIDEBAR: АВТОРИЗАЦИЯ И МОДЕЛИ
# ============================================================

st.sidebar.header("🔑 Авторизация VseGPT")

env_key = (os.getenv("API_KEY") or os.getenv("VSEGPT_API_KEY") or "").strip()

if env_key:
    api_key = env_key
    st.sidebar.success("✅ Ключ успешно загружен из файла .env")
else:
    api_key = st.sidebar.text_input(
        "Введите API Ключ вручную:", type="password"
    ).strip()

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

if not api_key:
    st.warning(
        "Пожалуйста, добавьте API_KEY в файл .env или введите его на боковой панели."
    )
    st.stop()

client = OpenAI(
    api_key=api_key,
    base_url="https://api.vsegpt.ru/v1",
    timeout=120,
    max_retries=2,
)


# ============================================================
# БАЗА НОРМ-ЧАСОВ
# ============================================================

NORMS_FILE = os.getenv(
    "NORMS_FILE", "База для нормы часам и единицам измерениям.xlsx"
)


@st.cache_resource(show_spinner=False)
def get_norms_db(_path: str, _mtime: float):
    return load_norms(_path)


norms_db = None
try:
    if os.path.exists(NORMS_FILE):
        norms_db = get_norms_db(NORMS_FILE, os.path.getmtime(NORMS_FILE))
        st.sidebar.success(f"✅ Нормы загружены ({len(norms_db)} поз.)")
    else:
        st.sidebar.warning("⚠️ Файл норм не найден — нормы не подставятся.")
except Exception as e:
    st.sidebar.error(f"Ошибка загрузки норм: {e}")

norms_upload = st.sidebar.file_uploader(
    "…или загрузите базу норм (.xlsx)", type=["xlsx"], key="norms_upload"
)
if norms_upload is not None:
    try:
        norms_db = load_norms(norms_upload.read())
        st.sidebar.success(f"✅ Нормы из загрузки: {len(norms_db)} поз.")
    except Exception as e:
        st.sidebar.error(f"Ошибка чтения базы норм: {e}")


# ============================================================
# УТИЛИТЫ (используются в Агенте #1)
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


# ============================================================
# СПРАВОЧНИКИ ДЛЯ ДЛИТЕЛЬНОСТИ (fallback, если нормы нет)
# ============================================================

WORK_RATES = {
    "concrete":      {"rate": 14.0,  "unit": "м3",  "name": "Укладка бетона"},
    "formwork":      {"rate": None,  "unit": "",    "name": "Монтаж опалубки"},
    "rebar":         {"rate": 500.0, "unit": "кг",  "name": "Армирование"},
    "insulation":    {"rate": 50.0,  "unit": "м2",  "name": "Гидроизоляция"},
    "waterproofing": {"rate": 50.0,  "unit": "м2",  "name": "Гидроизоляция"},
    "backfill":      {"rate": 20.0,  "unit": "м3",  "name": "Обратная засыпка"},
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
    "waterproofing": "м2", "backfill": "м3", "earthwork": "м3",
    "demolition": "м3", "pipeline": "м.п.", "installation": "шт",
    "finishing": "м2", "masonry": "м3", "roofing": "м2",
    "electrical": "м.п.", "hvac": "м.п.", "general": "",
}


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
        (["землян", "котлован", "выемк", "разработк"], ["м3"], "earthwork"),
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
# EXCEL: ЗАГРУЗКА ГОТОВОЙ ВЕДОМОСТИ
# ============================================================

ESTIMATE_COLUMNS = [
    "wbs_parent", "element", "count", "volume",
    "unit", "category", "note", "source_page",
]

COLUMN_ALIASES = {
    "wbs_parent": ["wbs_parent", "wbs", "раздел", "конструктив", "марка"],
    "element":    ["element", "наименование", "название", "работа",
                   "элемент", "описание"],
    "count":      ["count", "кол-во", "количество", "кол."],
    "volume":     ["volume", "объем", "объём", "обьем", "qty"],
    "unit":       ["unit", "ед", "ед. изм", "ед.изм", "единица",
                   "единица измерения"],
    "category":   ["category", "категория", "тип", "вид работ"],
    "note":       ["note", "примечание", "комментарий"],
    "source_page":["source_page", "страница", "лист", "page"],
}


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]

    rename = {}
    for target, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in df.columns and target not in df.columns:
                rename[alias] = target
                break
    df = df.rename(columns=rename)

    for col in ESTIMATE_COLUMNS:
        if col not in df.columns:
            df[col] = ""

    return df[ESTIMATE_COLUMNS]


def load_estimate_from_excel(file_bytes: bytes) -> pd.DataFrame:
    df = pd.read_excel(io.BytesIO(file_bytes))
    df = _normalize_columns(df)
    df = df.dropna(how="all").reset_index(drop=True)

    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    df["count"] = pd.to_numeric(df["count"], errors="coerce")

    for col in ("wbs_parent", "element", "unit", "category",
                "note", "source_page"):
        df[col] = df[col].fillna("").astype(str).str.strip()

    df = df[df["element"] != ""].reset_index(drop=True)

    df["unit"] = df["unit"].apply(normalize_unit)
    df["wbs_parent"] = df["wbs_parent"].apply(clean_wbs)

    df["category"] = df["category"].str.lower()
    mask_empty = df["category"] == ""
    if mask_empty.any():
        df.loc[mask_empty, "category"] = [
            classify_work(r["element"], r["unit"])
            for _, r in df[mask_empty].iterrows()
        ]

    df = df[
        ~df["element"].str.lower().str.contains(
            r"\bитого\b|\bвсего\b", regex=True, na=False
        )
    ].reset_index(drop=True)

    df["norm_hours"] = None
    return df


# ============================================================
# SESSION STATE
# ============================================================

if "raw_estimate_df" not in st.session_state:
    st.session_state.raw_estimate_df = None
if "p6_df" not in st.session_state:
    st.session_state.p6_df = None
if "agent3_result" not in st.session_state:
    st.session_state.agent3_result = None
if "last_file_hash" not in st.session_state:
    st.session_state.last_file_hash = None
if "last_excel_hash" not in st.session_state:
    st.session_state.last_excel_hash = None
if "estimate_source" not in st.session_state:
    st.session_state.estimate_source = None
if "planner_extra_prompt" not in st.session_state:
    st.session_state.planner_extra_prompt = None
if "manual_norm_choices" not in st.session_state:
    st.session_state.manual_norm_choices = {}


# ============================================================
# ПРОМПТ АГЕНТА #1
# ============================================================

VISION_PROMPT = """Ты — Агент-Замерщик, эксперт по чтению ЛЮБЫХ строительных чертежей (КЖ, КМ, АР, ГП, ЭМ, ВК, ОВ и др.).
Твоя задача — составить полную ведомость всего, что можно посчитать на листах, с привязкой к конструктивам.

ЧТО ИСКАТЬ:
1) Таблицы "Спецификация", "Спецификация фундамента", "Ведомость расхода стали", "Ведомость материалов".
2) Если позиции в таблицах нет — геометрические размеры на планах/разрезах.
3) Если нет и размеров — текстовые указания в примечаниях (площади, толщины слоёв, классы уплотнения).
4) Штучные элементы (люки, двери, окна, насосы, резервуары, фиксаторы, болты).

ПРАВИЛА ПРИОРИТЕТА (важнее всего!):
- Если позиция ЕСТЬ в спецификации/ведомости — бери объём ТОЛЬКО оттуда, НЕ пересчитывай по габаритам.
- Если в ведомости стали несколько позиций арматуры — выдай КАЖДУЮ отдельной строкой.
- Если объёма нет ни в таблице, ни в размерах — volume = null и в note напиши, чего не хватает.
- НЕ ОБЪЕДИНЯЙ разные материалы в одну строку.
- НЕ ПРОПУСКАЙ ничего: демонтаж, уплотнение, вывоз, монтаж, пусконаладку.

ЗАПРЕЩЕНО:
- Дублировать данные (Спецификация vs Ведомость стали).
- Создавать строки "Итого", "Всего".

ПОЛЕ wbs_parent: ТОЛЬКО марка конструктива (например, "Фундамент Ф4").
ПОЛЕ unit — строго одно из: "м3", "м2", "кг", "т", "шт", "м.п.", "компл.", "стык".
ПОЛЕ category — одно из: earthwork, formwork, rebar, concrete, insulation, waterproofing, backfill, installation, demolition, pipeline, finishing, masonry, roofing, electrical, hvac, general.

ФОРМАТ ОТВЕТА — строго JSON-массив объектов с полями:
wbs_parent, element, count, volume, unit, category, source_page, note.
Не добавляй markdown, не пиши пояснений до или после JSON."""
# ============================================================
# UI: ВЫБОР ИСТОЧНИКА ВЕДОМОСТИ
# ============================================================

st.header("1️⃣ Шаг 1: Источник ведомости объёмов")

tab_draw, tab_xlsx = st.tabs([
    "📐 Из чертежа (Агент #1)",
    "📊 Готовая ведомость (Excel)",
])


def _enrich_norms_for_estimate(df: pd.DataFrame, norms_db) -> pd.DataFrame:
    """
    Первичное обогащение ведомости нормами (для просмотра в редакторе).
    Это НЕ Агент #3 — просто предзаполнение norm_hours,
    чтобы пользователь видел нормы до запуска Агента #2.
    """
    df = df.copy()
    if "norm_hours" not in df.columns:
        df["norm_hours"] = None
    df["norm_hours"] = pd.to_numeric(df["norm_hours"], errors="coerce")

    if norms_db is None:
        return df

    for i, r in df.iterrows():
        if pd.notna(df.at[i, "norm_hours"]):
            continue
        try:
            rec = norms_db.find(
                str(r.get("element", "")),
                str(r.get("unit", "")),
                require_same_unit=True,
            )
        except Exception:
            rec = None
        if rec is not None and rec.hours is not None:
            df.at[i, "norm_hours"] = rec.hours

    return df


# -------- Вкладка 1: чертёж --------
with tab_draw:
    uploaded_file = st.file_uploader(
        "Загрузите скан спецификации/чертежа (PDF, PNG, JPG)",
        type=["pdf", "png", "jpg", "jpeg"],
        key="uploader_draw",
    )

# -------- Вкладка 2: готовый Excel --------
with tab_xlsx:
    uploaded_xlsx = st.file_uploader(
        "Загрузите ведомость объёмов (.xlsx)",
        type=["xlsx", "xls"],
        key="uploader_xlsx",
    )

    st.caption(
        "Ожидаемые колонки (гибко, регистр не важен): "
        "`wbs_parent`, `element`, `volume`, `unit`, `category`, `count`, "
        "`note`, `source_page`. Русские названия («Наименование», «Объём», "
        "«Ед. изм.», «Раздел»…) тоже распознаются."
    )

    if uploaded_xlsx:
        xlsx_bytes = uploaded_xlsx.read()
        xlsx_hash = hashlib.md5(xlsx_bytes).hexdigest()

        if xlsx_hash != st.session_state.last_excel_hash:
            st.session_state.last_excel_hash = xlsx_hash
            st.session_state.raw_estimate_df = None
            st.session_state.p6_df = None
            st.session_state.agent3_result = None

        if st.button("📥 Загрузить ведомость из Excel", key="btn_load_xlsx"):
            try:
                df_xlsx = load_estimate_from_excel(xlsx_bytes)
                if df_xlsx.empty:
                    st.warning(
                        "В файле не найдено ни одной строки с заполненным "
                        "«наименованием» (element). Проверьте колонки."
                    )
                else:
                    df_xlsx = _enrich_norms_for_estimate(df_xlsx, norms_db)
                    st.session_state.raw_estimate_df = df_xlsx
                    st.session_state.p6_df = None
                    st.session_state.agent3_result = None
                    st.session_state.estimate_source = "excel"
                    st.success(
                        f"✅ Загружено {len(df_xlsx)} строк из Excel "
                        f"(нормы предзаполнены где найдено)."
                    )
            except Exception as e:
                st.error(f"Не удалось прочитать Excel: {e}")


# ============================================================
# ЛОГИКА АГЕНТА #1
# ============================================================

if uploaded_file:
    file_bytes = uploaded_file.read()

    file_hash = hashlib.md5(file_bytes).hexdigest()
    if file_hash != st.session_state.last_file_hash:
        st.session_state.last_file_hash = file_hash
        st.session_state.last_excel_hash = None
        st.session_state.raw_estimate_df = None
        st.session_state.p6_df = None
        st.session_state.agent3_result = None
        st.session_state.estimate_source = None
        st.session_state.planner_extra_prompt = None
        st.session_state.manual_norm_choices = {}

    if st.button("🔍 Запустить Агента #1: Собрать объемы"):
        with st.spinner("Агент #1 изучает чертеж и составляет ведомость объемов..."):
            ai_output = None
            try:
                content = [{"type": "text", "text": VISION_PROMPT}]

                if uploaded_file.type == "application/pdf":
                    last_page = None if max_pages == 0 else max_pages
                    images = convert_from_bytes(
                        file_bytes, dpi=150, first_page=1,
                        last_page=last_page, poppler_path=POPPLER_PATH,
                    )
                    if not images:
                        raise ValueError("PDF не содержит страниц")

                    for page_img in images:
                        page_img = shrink_image_to_limit(page_img)
                        buf = io.BytesIO()
                        page_img.save(buf, format="JPEG", quality=90)
                        content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,"
                                       f"{encode_image(buf.getvalue())}"
                            },
                        })
                        del page_img
                else:
                    mime = uploaded_file.type or "image/jpeg"
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime};base64,"
                                   f"{encode_image(file_bytes)}"
                        },
                    })

                ai_output = call_model(
                    model_vision,
                    messages=[{"role": "user", "content": content}],
                )
                data_vision = extract_json_array(ai_output)

                if not data_vision:
                    st.warning("🤖 Агент #1 не нашёл конструкций на чертеже.")
                    st.code(ai_output)
                else:
                    df = pd.DataFrame(data_vision)

                    if "volume" in df.columns:
                        df["volume"] = pd.to_numeric(
                            df["volume"], errors="coerce"
                        )
                    else:
                        df["volume"] = None

                    for col in ("wbs_parent", "element", "count",
                                "unit", "category", "note", "source_page"):
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

                    df = _enrich_norms_for_estimate(df, norms_db)
                    df["volume"] = pd.to_numeric(
                        df["volume"], errors="coerce"
                    )

                    st.session_state.raw_estimate_df = df
                    st.session_state.p6_df = None
                    st.session_state.agent3_result = None
                    st.session_state.estimate_source = "vision"
                    st.success(
                        "🤖 Агент #1 успешно сформировал ведомость "
                        "(нормы предзаполнены где найдено)!"
                    )

            except Exception as e:
                st.error(f"Ошибка Агента #1: {e}")
                if ai_output:
                    st.code(ai_output)

# ============================================================
# РЕДАКТОР ВЕДОМОСТИ
# ============================================================

if st.session_state.raw_estimate_df is not None:
    src = st.session_state.estimate_source
    badge = {
        "vision": "📐 из чертежа (Агент #1)",
        "excel":  "📊 из Excel",
    }.get(src, "❓ источник неизвестен")
    st.caption(f"Источник: {badge}")

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

    if "norm_hours" not in edited_df.columns:
        edited_df["norm_hours"] = None
    edited_df["norm_hours"] = pd.to_numeric(
        edited_df["norm_hours"], errors="coerce"
    )

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
    # ШАГ 2: АГЕНТ-ПЛАНИРОВЩИК
    # ============================================================

    st.markdown("---")
    st.header("2️⃣ Шаг 2: Агент-Планировщик (Конвертация данных в Primavera P6)")

    def _build_planner_prompt(est_df: pd.DataFrame, extra_prompt: str = "") -> str:
        cols_for_llm = ["wbs_parent", "element", "volume",
                        "unit", "category", "norm_hours", "note"]
        context = est_df[cols_for_llm].to_string(index=False)

        base = f"""Ты — Агент-Планировщик Primavera P6.
Изучи ведомость объёмов:

{context}

ВАЖНО: колонка «norm_hours» — норма человеко-часов на ЕДИНИЦУ.
Тебе НЕ нужно её пересчитывать — просто верни задачи, не трогая объёмы.

Твоя задача — сформировать иерархическую структуру работ (WBS).
Для КАЖДОГО конструктива (wbs_parent) создай набор задач в ТЕХНОЛОГИЧЕСКОЙ ПОСЛЕДОВАТЕЛЬНОСТИ.

ПРАВИЛА:
1. Поле "wbs_parent" бери БУКВАЛЬНО из таблицы выше.
2. Поле "phase" — тип работы (earthwork, formwork, rebar, concrete,
   insulation, waterproofing, backfill, installation, demolition,
   pipeline, finishing, masonry, roofing, electrical, hvac, general).
3. Поле "predecessor" — null, если задача первая, "AUTO", если предшественник есть.

ОПРЕДЕЛЕНИЕ phase:
- "арматур", "армирован", "стержн", "сетк" → rebar
- "бетон" → concrete
- "опалубк" → formwork
- "фиксатор", "болт", "люк", "колодец", "насос", "дверь", "окно" → installation
- "гидроизоляц", "обмазк", "мастик", "праймер", "битум" → waterproofing
- "засыпк", "отсев", "уплотнен" → backfill
- "котлован", "разработк", "землян" → earthwork

НЕ ДРОБИ:
- Арматура: ОДНА задача "Армирование <WBS>" с phase="rebar".
- Бетон: ДВЕ задачи (B10 подготовка + B30 конструктив).
- Обратная засыпка: ОДНА задача.

ЛОГИЧЕСКИЙ ПОРЯДОК:
earthwork → formwork → rebar → concrete → waterproofing/insulation →
backfill → installation → finishing.

Верни строго JSON-массив:
[{{"wbs_parent": "...", "task_name": "...", "phase": "...", "predecessor": null|"AUTO"}}, ...]
"""
        if extra_prompt:
            base += "\n\n" + extra_prompt
        return base

    # ---------- Кнопка «Отправить на повтор» (Вариант В) ----------
    if st.session_state.planner_extra_prompt:
        st.info(
            "🔄 Агент #2 подготовлен к повторному запуску с учётом замечаний "
            "Агента #3."
        )
        if st.button("▶️ Повторить Агента #2 с замечаниями"):
            st.session_state.planner_extra_prompt = None
            st.rerun()

    # ---------- Основной запуск ----------
    if st.button("📊 Запустить Агента #2: Создать структуру Primavera"):
        with st.spinner("Агент #2 формирует WBS и технологические циклы..."):
            extra = st.session_state.planner_extra_prompt or ""
            prompt = _build_planner_prompt(edited_df, extra)
            p6_output = None

            try:
                p6_output = call_model(
                    model_planner,
                    messages=[{"role": "user", "content": prompt}],
                )
                rows = extract_json_array(p6_output)
                if not rows:
                    raise ValueError("Агент #2 вернул пустой список работ")

                known_wbs = set(
                    str(x).strip()
                    for x in edited_df["wbs_parent"].dropna().unique()
                )

                def match_wbs(raw_wbs: str) -> str:
                    raw = (raw_wbs or "").strip()
                    if raw in known_wbs:
                        return raw
                    raw_l = raw.lower()
                    candidates = [
                        k for k in known_wbs
                        if raw_l in k.lower() or k.lower() in raw_l
                    ]
                    if candidates:
                        return min(candidates, key=len)
                    first_word = raw.split()[0] if raw.split() else ""
                    if first_word:
                        candidates = [
                            k for k in known_wbs
                            if first_word.lower() in k.lower()
                        ]
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

                CATEGORY_ALIAS = {
                    "insulation": "waterproofing",
                    "waterproofing": "insulation",
                }

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
                        volume_map.setdefault((wbs_key, alias), []).append(
                            dict(item)
                        )

                activities = []
                aid_counter = 2000

                rows_sorted = sorted(
                    rows,
                    key=lambda x: (
                        str(x.get("wbs_parent", "")),
                        PHASE_ORDER.get(
                            str(x.get("phase", "general")).lower(), 99
                        ),
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

                    # 1) точное сопоставление
                    exact_matches = []
                    for _, er in edited_df.iterrows():
                        if str(er["wbs_parent"]).strip() != wbs:
                            continue
                        if str(er["category"]).strip().lower() != phase:
                            continue
                        el_name = str(er["element"]).lower()
                        el_words = {
                            w for w in re.findall(r"[a-zа-я0-9]+", el_name)
                            if len(w) > 2
                        }
                        task_words = {
                            w for w in re.findall(r"[a-zа-я0-9]+", task_lower)
                            if len(w) > 2
                        }
                        if el_words & task_words:
                            exact_matches.append((len(el_words & task_words), er))

                    if exact_matches:
                        exact_matches.sort(key=lambda x: -x[0])
                        best = exact_matches[0][1]
                        found_unit = normalize_unit(str(best["unit"]))
                        found_vol = (
                            float(best["volume"])
                            if pd.notna(best["volume"]) else 0.0
                        )
                        if phase == "rebar":
                            vol, unit = to_kg(found_vol, found_unit), "кг"
                        else:
                            vol, unit = (
                                found_vol, found_unit or expected_unit or "м3"
                            )
                    else:
                        # 2) fallback на агрегат
                        items = volume_map.get((wbs, phase), [])
                        vol, unit = 0.0, expected_unit or "м3"
                        if items:
                            if expected_unit:
                                matching = [
                                    it for it in items
                                    if it["unit"] == expected_unit
                                ]
                            else:
                                cnt = Counter(it["unit"] for it in items)
                                main_unit = cnt.most_common(1)[0][0]
                                matching = [
                                    it for it in items
                                    if it["unit"] == main_unit
                                ]
                            if matching:
                                if phase == "rebar":
                                    total = sum(
                                        to_kg(it["volume"], it["unit"])
                                        for it in matching
                                    )
                                    unit = "кг"
                                else:
                                    total = sum(
                                        it["volume"] for it in matching
                                    )
                                    unit = matching[0]["unit"]
                                n_tasks = task_count_by_group.get(
                                    (wbs, phase), 1
                                ) or 1
                                vol = total / n_tasks

                    # 3) дополнительный fallback
                    if vol == 0.0:
                        best_match = None
                        best_score = 0
                        for _, er in edited_df.iterrows():
                            el_name = str(er["element"]).lower()
                            if el_name and (el_name in task_lower
                                            or task_lower in el_name):
                                score = 100
                            else:
                                wt = {
                                    w for w in re.findall(r"\w+", task_lower)
                                    if len(w) > 2
                                }
                                we = {
                                    w for w in re.findall(r"\w+", el_name)
                                    if len(w) > 2
                                }
                                score = len(wt & we)
                            if score > best_score:
                                best_score = score
                                best_match = er

                        if best_match is not None and best_score >= 1:
                            found_unit = normalize_unit(
                                str(best_match.get("unit", ""))
                            )
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

                    # длительность — на этом шаге без норм (Агент #3 пересчитает)
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

                    # предшественники
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
                    raise ValueError(
                        "Не удалось сопоставить элементы с работами."
                    )

                st.session_state.p6_df = pd.DataFrame(activities)
                st.session_state.agent3_result = None

            except Exception as e:
                st.error(f"Ошибка Агента #2: {e}")
                if p6_output:
                    st.code(p6_output)

    # ============================================================
    # ШАГ 3: АГЕНТ-ВЕРИФИКАТОР
    # ============================================================

    if st.session_state.p6_df is not None:
        st.markdown("---")
        st.header("3️⃣ Шаг 3: Агент-Верификатор (проверка логики + нормы)")

        if st.button("🔍 Запустить Агента #3: Проверить и обогатить нормы"):
            with st.spinner("Агент #3 проверяет логику и подставляет нормы..."):
                activities_list = st.session_state.p6_df.to_dict(
                    orient="records"
                )
                try:
                    result = run_agent3(
                        activities_list, edited_df, norms_db
                    )
                    st.session_state.agent3_result = result
                    st.success("✅ Агент #3 завершил проверку.")
                except Exception as e:
                    st.error(f"Ошибка Агента #3: {e}")

    # ============================================================
    # РЕЗУЛЬТАТ АГЕНТА #3
    # ============================================================

    if st.session_state.agent3_result is not None:
        res: Agent3Result = st.session_state.agent3_result
        summary = res.summary()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("❌ Ошибки", summary["errors"])
        col2.metric("⚠️ Замечания", summary["warnings"])
        col3.metric("🔧 Автоисправлено", summary["autofixed"])
        col4.metric("❓ Ручной выбор", summary["manual_selection"])

        # ---------- Замечания ----------
        if res.issues:
            with st.expander(
                f"🔍 Замечания Агента #3 ({len(res.issues)})",
                expanded=True,
            ):
                for i in res.issues:
                    icon = "❌" if i.severity == "error" else "⚠️"
                    st.markdown(
                        f"{icon} **[{i.code}]** {i.message}"
                        + (f"  \n`WBS: {i.wbs} | ID: {i.activity_id}`"
                           if i.wbs or i.activity_id else "")
                    )

        # ---------- Автофиксы ----------
        if res.autofix_log:
            with st.expander(
                f"🔧 Автоисправления ({len(res.autofix_log)})",
                expanded=False,
            ):
                for line in res.autofix_log:
                    st.text(f"• {line}")

        # ---------- Кнопка «Отправить на повтор» ----------
        if res.needs_llm_retry:
            st.warning(
                "⚠️ Остались семантические замечания, которые автофикс "
                "не устранил. Можно отправить Агента #2 на повтор."
            )
            if st.button("🔁 Отправить Агента #2 на повтор с замечаниями"):
                fix_prompt = build_llm_fix_prompt(
                    res.issues,
                    res.activities,
                    edited_df,
                )
                st.session_state.planner_extra_prompt = fix_prompt
                st.session_state.p6_df = pd.DataFrame(res.activities)
                st.session_state.agent3_result = None
                st.rerun()

        # ---------- Итоговая таблица P6 ----------
        st.subheader("🚀 Итоговая структура задач:")
        result_df = pd.DataFrame(res.activities)

        # пересчитываем длительность по нормам (если есть)
        def _recalc_duration(row):
            nh = row.get("Norm Hours")
            th = row.get("Total Hours")
            if pd.notna(th) and th not in ("", None):
                try:
                    return max(1, int(round(float(th) / 8.0)))
                except (TypeError, ValueError):
                    pass
            return row.get("Original Duration", 1)

        result_df["Original Duration"] = result_df.apply(
            _recalc_duration, axis=1
        )

        st.dataframe(result_df, use_container_width=True)

        # ---------- Сумма человеко-часов ----------
        try:
            th = pd.to_numeric(result_df["Total Hours"], errors="coerce").fillna(0)
            total_h = float(th.sum())
            if total_h > 0:
                st.info(
                    f"Σ человеко-часов: **{total_h:,.2f} ч** "
                    f"(≈ {total_h / 8.0:,.1f} смен по 8 ч)"
                )
        except Exception:
            pass

        # ---------- Ручной выбор нормы ----------
        if res.manual_selection_rows:
            with st.expander(
                f"❓ Требуют ручного выбора нормы "
                f"({len(res.manual_selection_rows)})",
                expanded=True,
            ):
                st.caption(
                    "Для этих задач норма не найдена автоматически. "
                    "Выберите позицию из базы вручную."
                )
                if norms_db is None:
                    st.error(
                        "База норм не загружена — ручной выбор недоступен."
                    )
                else:
                    options = [""] + [
                        f"{r.code} | {r.name} | {r.unit} | {r.hours} ч"
                        for r in norms_db.records
                        if r.hours is not None
                    ]
                    code_to_record = {
                        r.code: r for r in norms_db.records
                        if r.hours is not None
                    }

                    for row in res.manual_selection_rows:
                        aid = row["Activity ID"]
                        key = f"manual_{aid}"
                        st.markdown(
                            f"**{aid}** — {row['Activity Name']}  \n"
                            f"`WBS: {row['WBS']} | "
                            f"{row['Volume']} {row['Unit']}`  \n"
                            f"_Причина: {row['reason']}_"
                        )
                        choice = st.selectbox(
                            "Выберите норму:",
                            options,
                            key=key,
                            label_visibility="collapsed",
                        )
                        if choice:
                            code = choice.split(" | ")[0]
                            rec = code_to_record.get(code)
                            if rec is not None:
                                st.session_state.manual_norm_choices[aid] = {
                                    "code": rec.code,
                                    "hours": rec.hours,
                                    "unit": rec.unit,
                                }

                    if st.button("✅ Применить ручной выбор"):
                        # обновляем activities
                        acts = [dict(a) for a in res.activities]
                        for a in acts:
                            aid = a.get("Activity ID")
                            if aid in st.session_state.manual_norm_choices:
                                ch = st.session_state.manual_norm_choices[aid]
                                a["Norm Hours"] = ch["hours"]
                                a["Norm Source"] = ch["code"]
                                a["Norm Confidence"] = "manual"
                                vol = a.get("Volume", 0)
                                try:
                                    a["Total Hours"] = float(vol) * float(
                                        ch["hours"]
                                    )
                                except (TypeError, ValueError):
                                    a["Total Hours"] = ""
                        st.session_state.agent3_result.activities = acts
                        st.rerun()

        # ---------- Экспорт ----------
        p6_buffer = io.BytesIO()
        with pd.ExcelWriter(p6_buffer, engine="openpyxl") as writer:
            result_df.to_excel(writer, index=False, sheet_name="TASK")

        st.download_button(
            label="📥 Скачать итоговую P6-таблицу (Excel)",
            data=p6_buffer.getvalue(),
            file_name="Primavera_P6_Final_with_Norms.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        st.caption(
            "⚠️ Колонки `Norm Hours`, `Total Hours`, `Norm Source`, "
            "`Norm Confidence` — результат Агента #3. "
            "Формат — черновик для маппинга в Primavera P6."
        )