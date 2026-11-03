import streamlit as st
import io
import base64
import json
import os
import hashlib
import re
import time
from dotenv import load_dotenv
import pandas as pd
from openai import OpenAI
from pdf2image import convert_from_bytes
from PIL import Image
from collections import Counter

load_dotenv()

st.set_page_config(page_title="Мультиагентный ИИ-Сметчик", layout="wide")
st.title("🏗️ Мультиагентная система: Чертёж ➔ Смета(ВОР) ➔ Primavera P6")

POPPLER_PATH = os.getenv("POPPLER_PATH")

st.sidebar.header("🔑 Авторизация VseGPT")
env_key = (os.getenv("API_KEY") or os.getenv("VSEGPT_API_KEY") or "").strip()

if env_key:
    api_key = env_key
    st.sidebar.success("✅ Ключ успешно загружен из файла .env")
else:
    api_key = st.sidebar.text_input("Введите API Ключ вручную:", type="password").strip()

model_vision = st.sidebar.selectbox(
    "Агент #1 (Зрение):", ["anthropic/claude-sonnet-5.5-high", "openai/gpt-4o", "openai/gpt-4o-mini"]
)
model_planner = st.sidebar.selectbox(
    "Агент #2 (Планировщик):", ["openai/gpt-4o", "anthropic/claude-sonnet-5.5-high"]
)

st.sidebar.header("⚙️ Настройки обработки")
project_section = st.sidebar.selectbox(
    "Раздел проекта:",
    ["Универсальный", "КЖ (Железобетон)", "КМ (Металлоконструкции)", "ЭС (Электрика)", "АР (Архитектура)"]
)

skip_pages = st.sidebar.number_input(
    "Пропуск первых листов (ТУ, общие данные):", min_value=0, value=2, step=1
)

max_pages = st.sidebar.number_input(
    "Сколько страниц читать (0 = все):", min_value=0, value=0, step=1
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
    if not text: raise ValueError("Пустой ответ модели")
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)

    start = text.find("[")
    if start == -1: raise ValueError("В ответе модели нет JSON-массива")

    end = text.rfind("]") + 1
    if end > start:
        candidate = text[start:end]
        try: return json.loads(candidate)
        except json.JSONDecodeError: pass

    body = text[start:]
    objects, depth, buf, in_string, escape = [], 0, [], False, False
    for ch in body:
        if escape: buf.append(ch); escape = False; continue
        if ch == "\\": buf.append(ch); escape = True; continue
        if ch == '"': in_string = not in_string; buf.append(ch); continue
        if in_string: buf.append(ch); continue
        if ch == "{": depth += 1; buf.append(ch)
        elif ch == "}":
            depth -= 1; buf.append(ch)
            if depth == 0 and buf:
                try: objects.append(json.loads("".join(buf)))
                except json.JSONDecodeError: pass
                buf = []
        elif depth > 0: buf.append(ch)

    if not objects: raise ValueError("Не удалось извлечь ни одного корректного объекта.")
    return objects


def call_model(model: str, messages, max_tokens: int = 16000):
    response = client.chat.completions.create(
        model=model, messages=messages, temperature=0, max_tokens=max_tokens,
    )

    # --- СЧИТАЕМ ТОКЕНЫ И СТОИМОСТЬ ---
    if hasattr(response, "usage") and response.usage:
        prompt_t = response.usage.prompt_tokens
        completion_t = response.usage.completion_tokens
        st.session_state.total_prompt_tokens += prompt_t
        st.session_state.total_completion_tokens += completion_t

        # Тарифы VseGPT (в рублях за 1М токенов) для популярных моделей
        # Вы можете уточнить актуальные цены в личном кабинете VseGPT
        PRICES = {
            "anthropic/claude-sonnet-5.5-high": {"in": 600, "out": 3000},  # Примерная цена
            "openai/gpt-4o": {"in": 480, "out": 1930},  # Примерная цена
        }
        # Если модели нет в списке, берем среднюю цену
        price = PRICES.get(model, {"in": 500, "out": 2000})

        cost = (prompt_t / 1_000_000 * price["in"]) + (completion_t / 1_000_000 * price["out"])
        st.session_state.total_cost_rub += cost
    # -----------------------------------

    if isinstance(response, str): return response.strip()
    choice = response.choices[0]
    content = (choice.message.content or "").strip()
    if getattr(choice, "finish_reason", None) == "length":
        raise RuntimeError(f"Ответ модели оборван по лимиту max_tokens={max_tokens}.")
    return content
WORK_RATES = {
    "concrete": {"rate": 14.0, "unit": "м3"}, "formwork": {"rate": None, "unit": ""},
    "rebar": {"rate": 500.0, "unit": "кг"}, "insulation": {"rate": 50.0, "unit": "м2"},
    "waterproofing": {"rate": 50.0, "unit": "м2"}, "backfill": {"rate": 20.0, "unit": "м3"},
    "earthwork": {"rate": 60.0, "unit": "м3"}, "demolition": {"rate": 10.0, "unit": "м3"},
    "pipeline": {"rate": 30.0, "unit": "м.п."}, "installation": {"rate": 3.0, "unit": "шт"},
    "finishing": {"rate": 40.0, "unit": "м2"}, "masonry": {"rate": 5.0, "unit": "м3"},
    "roofing": {"rate": 30.0, "unit": "м2"}, "electrical": {"rate": 30.0, "unit": "м.п."},
    "hvac": {"rate": 10.0, "unit": "м.п."}, "general": {"rate": 10.0, "unit": ""},
}

EXPECTED_UNIT = {
    "concrete": "м3", "formwork": "м2", "rebar": "кг", "insulation": "м2",
    "waterproofing": "м2", "backfill": "м3", "earthwork": "м3", "demolition": "м3",
    "pipeline": "м.п.", "installation": "шт", "finishing": "м2", "masonry": "м3",
    "roofing": "м2", "electrical": "м.п.", "hvac": "м.п.", "general": "",
}

UNIT_NORMALIZE = {
    "м³": "м3", "m3": "м3", "куб": "м3", "куб.м": "м3", "м²": "м2", "m2": "м2",
    "кв": "м2", "кв.м": "м2", "кг": "кг", "kg": "кг", "т": "т", "тн": "т",
    "ton": "т", "тонн": "т", "шт": "шт", "шт.": "шт", "pcs": "шт", "ед": "шт",
    "ед.": "шт", "м.п.": "м.п.", "мп": "м.п.", "пог.м": "м.п.", "пог.м.": "м.п.",
    "м.п": "м.п.", "м": "м.п.", "компл.": "компл.", "компл": "компл.",
}

def normalize_unit(u) -> str:
    if u is None: return "м3"
    try:
        if pd.isna(u): return "м3"
    except (TypeError, ValueError): pass
    return UNIT_NORMALIZE.get(str(u).strip().lower(), str(u).strip().lower() or "м3")

def to_kg(vol: float, unit: str) -> float:
    return vol * 1000.0 if unit == "т" else vol

def clean_wbs(w) -> str:
    s = str(w or "").strip()
    return re.split(r"[/|—–:]", s)[0].strip() or s or "Общие работы"

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
        (["гидроизоляц", "обмазк", "мастик", "рубероид", "мембран", "праймер", "битум"], ["м2"], "waterproofing"),
        (["труб", "трубопровод", "воздуховод"], ["м.п.", "м", "мп"], "pipeline"),
        (["монтаж", "установк", "резервуар", "насос", "люк", "колодец", "дверь", "окно", "фиксатор", "болт"], ["шт", "компл."], "installation"),
        (["штукатур", "окраск", "покраск", "отделк"], ["м2"], "finishing"),
        (["кладк", "кирпич", "газобетон", "пеноблок"], ["м3"], "masonry"),
    ]
    for keywords, units, category in RULES:
        if any(k in name for k in keywords):
            if not units or u in units: return category
    return "general"

PHASE_ORDER = {
    "demolition": 0, "earthwork": 1, "formwork": 2, "rebar": 3, "concrete": 4,
    "waterproofing": 5, "insulation": 5, "backfill": 6, "installation": 7,
    "pipeline": 7, "masonry": 8, "roofing": 9, "finishing": 10, "electrical": 11,
    "hvac": 11, "general": 99,
}

if "raw_estimate_df" not in st.session_state: st.session_state.raw_estimate_df = None
if "p6_df" not in st.session_state: st.session_state.p6_df = None
if "last_file_hash" not in st.session_state: st.session_state.last_file_hash = None
if "total_prompt_tokens" not in st.session_state: st.session_state.total_prompt_tokens = 0
if "total_completion_tokens" not in st.session_state: st.session_state.total_completion_tokens = 0
if "total_cost_rub" not in st.session_state: st.session_state.total_cost_rub = 0.0

PROMPTS = {
    "Универсальный": """Ты — универсальный Агент-Замерщик. Найди на листе таблицы (Спецификация, Ведомость) и перепиши их в JSON. Бери только данные из таблиц и текста. Не считай площади по клеточкам. Если объема нет — volume = null.""",

    "КЖ (Железобетон)": """Ты — эксперт по КЖ. Ищи спецификации арматуры, ведомости бетона, объемы опалубки, гидроизоляции. 
     Категория арматуры - 'rebar' (единица 'кг'). Бетон - 'concrete' (м3). 
     Если в таблице есть 'Масса ед.' и 'Кол-во' — пиши их в volume и count.""",

    "КМ (Металлоконструкции)": """Ты — эксперт по КМ. Ищи ведомости металла, сортамент, массу (швеллер, уголок, балка, труба). 
     Если есть 'Масса 1 ед.' и 'Кол-во' — ОБЯЗАТЕЛЬНО перемножай и пиши результат в volume. 
     Единица измерения металла всегда 'т' (тонны). Категория 'metal'. 
     Для окраски используй 'finishing' (м2).""",

    "ЭС (Электрика)": """Ты — эксперт по ЭС. Ищи кабельные журналы, спецификации лотков, проводов, щитов. 
     Для кабеля и лотков используй 'electrical' и 'м.п.'. 
     Для щитов, светильников и розеток — 'installation' и 'шт'.""",

    "АР (Архитектура)": """Ты — эксперт по АР. Ищи ведомости отделки, заполнения проемов (двери, окна), полы, потолки. Категория 'finishing' (м2)."""
}

# Этот блок добавляет общие правила к ЛЮБОМУ выбранному разделу
common_rules = """
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

 ПОЛЕ wbs_parent: Марка конструктива или название системы.
 ФОРМАТ ОТВЕТА — строго JSON-массив объектов: wbs_parent, element, count, volume, unit, category, source_page, note.
 Никакого markdown, никаких пояснений до или после JSON."""

# Итоговый промпт = Специализация раздела + Общие правила
current_prompt = PROMPTS[project_section] + common_rules

st.header("1️⃣ Шаг 1: Агент-ПТО (Сбор объемов)")

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
        with st.spinner("Агент #1 изучает чертежи..."):
            all_data = []
            try:
                if uploaded_file.type == "application/pdf":
                    first_page_to_read = skip_pages + 1
                    last_page = None if max_pages == 0 else max_pages

                    images = convert_from_bytes(
                        file_bytes, dpi=250, first_page=first_page_to_read,
                        last_page=last_page, poppler_path=POPPLER_PATH,
                    )
                    if not images:
                        raise ValueError("PDF не содержит страниц для обработки (проверьте пропуск).")

                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    for i, page_img in enumerate(images):
                        page_num = first_page_to_read + i
                        status_text.text(f"Анализ листа {page_num} (в обработке {i + 1} из {len(images)})...")

                        page_img = shrink_image_to_limit(page_img)
                        buf = io.BytesIO()
                        page_img.save(buf, format="JPEG", quality=90)

                        page_content = [
                            {"type": "text", "text": current_prompt},
                            {"type": "image_url",
                             "image_url": {"url": f"data:image/jpeg;base64,{encode_image(buf.getvalue())}"}}
                        ]

                        try:
                            ai_output = call_model(model_vision, messages=[{"role": "user", "content": page_content}])
                            if ai_output and ai_output.strip() != "[]":
                                data_vision = extract_json_array(ai_output)
                                for row in data_vision:
                                    row["pdf_page"] = page_num
                                all_data.extend(data_vision)
                        except Exception as e:
                            st.warning(f"Лист {page_num}: Ошибка распознавания ({e}). Пропускаем...")

                        progress_bar.progress((i + 1) / len(images))
                        time.sleep(1.5)

                    status_text.text("Обработка PDF завершена.")
                else:
                    mime = uploaded_file.type or "image/jpeg"
                    page_content = [
                        {"type": "text", "text": current_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encode_image(file_bytes)}"}}
                    ]
                    ai_output = call_model(model_vision, messages=[{"role": "user", "content": page_content}])
                    data_vision = extract_json_array(ai_output)
                    for row in data_vision: row["pdf_page"] = 1
                    all_data.extend(data_vision)

                if not all_data:
                    st.warning("🤖 Агент #1 не нашёл данных ни на одном листе.")
                else:
                    df = pd.DataFrame(all_data)
                    if "volume" in df.columns:
                        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
                    else:
                        df["volume"] = None
                    if "count" in df.columns:
                        df["count"] = pd.to_numeric(df["count"], errors="coerce")
                    else:
                        df["count"] = None

                    for col in ("wbs_parent", "element", "unit", "category", "note", "source_page", "pdf_page"):
                        if col not in df.columns: df[col] = ""

                    df["wbs_parent"] = df["wbs_parent"].apply(clean_wbs)
                    df["unit"] = df["unit"].apply(normalize_unit)
                    df["category"] = df["category"].fillna("").astype(str).str.strip().str.lower()

                    mask_empty = df["category"] == ""
                    if mask_empty.any():
                        df.loc[mask_empty, "category"] = [
                            classify_work(r["element"], r["unit"]) for _, r in df[mask_empty].iterrows()
                        ]

                    df = df[~df["element"].astype(str).str.lower().str.contains(r"\bитого\b|\bвсего\b", regex=True,
                                                                                na=False)].reset_index(drop=True)

                    st.session_state.raw_estimate_df = df
                    st.session_state.p6_df = None
                    st.success(f"🤖 Агент #1 успешно обработал листы и собрал {len(df)} строк!")

            except Exception as e:
                st.error(f"Критическая ошибка Агента #1: {e}")
if st.session_state.raw_estimate_df is not None:
    st.subheader("📋 Ведомость объемов (можно отредактировать перед шагом 2)")
    edited_df = st.data_editor(st.session_state.raw_estimate_df, num_rows="dynamic", key="editor_volumes")

    edited_df = edited_df[edited_df["element"].notna() & (edited_df["element"].astype(str).str.strip() != "")]
    edited_df = edited_df[edited_df["volume"].isna() | (pd.to_numeric(edited_df["volume"], errors="coerce") > 0)]

    edited_df["unit"] = edited_df["unit"].apply(normalize_unit)
    edited_df["category"] = edited_df["category"].fillna("").astype(str).str.strip().str.lower()
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
    st.header("2️⃣ Шаг 2: Агент-Планер (Конвертация в Primavera P6)")

    if st.button("📊 Запустить Агента #2: Создать структуру"):
        with st.spinner("Агент #2 формирует WBS и циклы..."):
            cols_for_llm = ["wbs_parent", "element", "volume", "unit", "category", "note"]
            excel_text_context = edited_df[cols_for_llm].to_string(index=False)
            p6_output = None

            try:

                # --- ДИНАМИЧЕСКИЕ ПРАВИЛА ДЛЯ РАЗНЫХ РАЗДЕЛОВ ---
                # --- 1. ЭТАЛОННЫЙ СПРАВОЧНИК РАБОТ (ИЗ ВАШЕГО EXCEL) ---
                REFERENCE_WORKS = {
                    "Железобетон (КЖ)": [
                        "Разработка грунта механизированным способом",
                        "Доработка вручную",
                        "Работа на отвал, погрузка и транспортировка грунта",
                        "Уплотнение грунта дна котлована вибротрамбовками",
                        "Устройство бетонной подготовки "
                        "Устройство опалубки основного конструктива",
                        "Устройство армокаркаса",
                        "Монтаж закладных деталей",
                        "Бетонирование основного конструктива",
                        "Уход за бетоном"
                        "Очистка бетонных поверхностей щетками",
                        "Обеспыливание бетонных поверхностей",
                        "Гидроизоляция фундамента (2 слоя)",
                        "Обратная засыпка пазух с уплотнением из отсева"
                    ],
                    "Металлоконструкции (КМ)": [
                        "Монтаж металлоконструкций",
                        "Абразивная очистка поверхности м/к",
                        "Обеспыливание и обезжиривание поверхности м/к",
                        "Окраска м/к поверхностей",
                        "Нанесение огнезащиты"
                    ],
                    "Электроснабжение (ЭС)": [
                        "Монтаж оборудования",
                        "Монтаж мачт освещения",
                        "Монтаж линии нВРУ в траншее",
                        "Монтаж линии нВРУ в ПНД трубе в траншее",
                        "Монтаж линии НГРУПП в траншее",
                        "Монтаж линии НГРУПП в ПНД трубе в траншее",
                        "Прокладка кабельной продукции в ПНД трубе в траншее",
                        "Опрессовка концевых разделок кабельной продукции",
                        "Разводка и подключение кабелей к устройствам",
                        "Монтаж кабельных бирок",
                        "Разработка грунта для траншей",
                        "Обратная засыпка траншеи просеянной землей",
                        "Обратная засыпка траншеи обычным грунтом",
                        "Укладка плит ПСК в траншею"
                    ]
                }

                # Формируем текстовый список для промпта
                reference_text = ""
                for section, works in REFERENCE_WORKS.items():
                    reference_text += f"\n- {section}: " + "; ".join(works)

                # --- 2. ДИНАМИЧЕСКИЕ ПРАВИЛА ДЛЯ ПЛАНИРОВЩИКА ---
                PLANNER_PROMPTS = {
                    "КЖ (Железобетон)": "ЛОГИЧЕСКИЙ ПОРЯДОК: earthwork → formwork → rebar → concrete → waterproofing → backfill.",
                    "КМ (Металлоконструкции)": "ЛОГИЧЕСКИЙ ПОРЯДОК: installation → finishing.",
                    "ЭС (Электрика)": "ЛОГИЧЕСКИЙ ПОРЯДОК: earthwork → installation → electrical → finishing.",
                    "АР (Архитектура)": "ЛОГИЧЕСКИЙ ПОРЯДОК: masonry → installation → finishing.",
                    "Универсальный": "ЛОГИЧЕСКИЙ ПОРЯДОК: earthwork → formwork → rebar → concrete → waterproofing → backfill → installation → finishing."
                }
                section_rules = PLANNER_PROMPTS.get(project_section, PLANNER_PROMPTS["Универсальный"])

                # --- 3. ВЫЗОВ МОДЕЛИ ---
                p6_output = call_model(
                    model_planner,
                    messages=[{
                        "role": "user",
                        "content": f"""Ты — Агент-Планировщик Primavera P6.
                Изучи ведомость объёмов:
                {excel_text_context}

                Твоя задача — сформировать иерархическую структуру работ (WBS).
                Раздел проекта: {project_section}.

                Вот ЭТАЛОННЫЙ СПРАВОЧНИК ТИПОВЫХ РАБОТ (технологическая карта):
                {reference_text}

                КРИТИЧЕСКИ ВАЖНО:
                
                1. Поле "wbs_parent" бери БУКВАЛЬНО из таблицы объёмов выше.
                2. Сверься со справочником. Если для конструктива не хватает какой-то технологической работы — ОБЯЗАТЕЛЬНО добавь её в план.
                3. ДЛЯ КАЖДОЙ задачи найди её объем в таблице выше.
                - НЕ бери объем от других работ (не подставляй бетон для опалубки, а кабель для щитов).
                4. ЕСЛИ РАБОТА ЕСТЬ В СПРАВОЧНИКЕ, НО ЕЁ НЕТ В ТАБЛИЦЕ ОБЪЕМОВ (или объем не найден):
                - В поле "volume" ставь 0.
                - В поле "task_name" ОБЯЗАТЕЛЬНО добавь в конец "(объём уточняется)".
                5. НЕ ДРОБИ одну категорию на несколько задач (если это одна работа — делай одну задачу).
                6. {section_rules}
                

                Верни строго JSON-массив. Пример:
                [{{"wbs_parent": "Фундамент Фм-1", "task_name": "Разработка грунта", "phase": "earthwork", "predecessor": null}},
                 {{"wbs_parent": "Фундамент Фм-1", "task_name": "Монтаж арматуры", "phase": "rebar", "predecessor": "AUTO"}}]"""
                    }]
                )

                rows = extract_json_array(p6_output)
                if not rows: raise ValueError("Агент #2 вернул пустой список.")

                known_wbs = set(str(x).strip() for x in edited_df["wbs_parent"].dropna().unique())


                def match_wbs(raw_wbs: str) -> str:
                    raw = (raw_wbs or "").strip()
                    if raw in known_wbs: return raw
                    candidates = [k for k in known_wbs if raw.lower() in k.lower() or k.lower() in raw.lower()]
                    return min(candidates, key=len) if candidates else raw


                from collections import Counter

                task_count_by_group = Counter(
                    (match_wbs(str(x.get("wbs_parent", "")).strip()), str(x.get("phase", "general")).strip().lower())
                    for x in rows)

                volume_map = {}
                for _, r in edited_df.iterrows():
                    wbs_key, cat_key = str(r["wbs_parent"]).strip(), str(r["category"]).strip().lower()
                    vol = float(r["volume"]) if pd.notna(r["volume"]) else 0.0
                    unit = normalize_unit(str(r.get("unit", "")))
                    if cat_key == "rebar" and unit == "т": vol, unit = vol * 1000.0, "кг"
                    item = {"volume": vol, "unit": unit, "element": str(r.get("element", "")).strip()}
                    volume_map.setdefault((wbs_key, cat_key), []).append(item)

                activities = []
                aid_counter = 2000


                def custom_sort_key(x):
                    wbs = str(x.get("wbs_parent", ""))
                    phase = str(x.get("phase", "general")).lower()
                    task_name = str(x.get("task_name", "")).lower()

                    # Если это КЖ и задача про бетонную подготовку, поднимаем ее выше (сразу после земляных работ)
                    if "подготовк" in task_name and phase in ["concrete", "formwork"]:
                        phase_order = 1.5  # Между earthwork (1) и formwork (2)
                    else:
                        phase_order = PHASE_ORDER.get(phase, 99)

                    return (wbs, phase_order)


                rows_sorted = sorted(rows, key=custom_sort_key)
                prev_aid_by_wbs = {}

                for r in rows_sorted:
                    wbs = match_wbs(str(r.get("wbs_parent", "Общие работы")).strip())
                    task_name = str(r.get("task_name", "")).strip()
                    phase = str(r.get("phase", "general")).strip().lower()
                    expected_unit, task_lower = EXPECTED_UNIT.get(phase, ""), task_name.lower()

                    # 1. Берем объем, который вернула нейросеть (Агент #2)
                    vol = float(r.get("volume", 0.0)) if r.get("volume") is not None else 0.0
                    unit = str(r.get("unit", "")).strip() or EXPECTED_UNIT.get(phase, "м3")

                    # 2. Если нейросеть вернула 0, пробуем найти точное совпадение по названию элемента
                    if vol == 0.0:
                        exact_matches = []
                        for _, er in edited_df.iterrows():
                            if str(er["wbs_parent"]).strip() != wbs or str(er["category"]).strip().lower() != phase:
                                continue

                            el_name_lower = str(er["element"]).lower()

                            # Умный фильтр для бетона (В10 vs В25)
                            if phase == "concrete" or "подготовк" in task_lower:
                                if "подготовк" in task_lower and "в10" not in el_name_lower and "b10" not in el_name_lower: continue
                                if "основн" in task_lower and (
                                        "в10" in el_name_lower or "b10" in el_name_lower): continue

                            # Ищем совпадения только по значимым словам (игнорируем "устройство", "монтаж")
                            STOP_WORDS = {"устройство", "монтаж", "работы", "слоя", "слоев", "поверхности", "основания"}
                            el_words = {w for w in re.findall(r"[a-zа-я0-9]+", el_name_lower) if
                                        len(w) > 3 and w not in STOP_WORDS}
                            task_words = {w for w in re.findall(r"[a-zа-я0-9]+", task_lower) if
                                          len(w) > 3 and w not in STOP_WORDS}

                            if el_words & task_words:
                                exact_matches.append((len(el_words & task_words), er))

                        if exact_matches:
                            exact_matches.sort(key=lambda x: -x[0])
                            best = exact_matches[0][1]
                            found_unit = normalize_unit(str(best["unit"]))
                            found_vol = float(best["volume"]) if pd.notna(best["volume"]) else 0.0
                            # Если это арматура, переводим в кг
                            if phase == "rebar":
                                vol, unit = to_kg(found_vol, found_unit), "кг"
                            else:
                                vol, unit = found_vol, found_unit or expected_unit or "м3"

                    # 3. Умная корректировка единиц (универсальная для всех разделов)
                    if phase == "rebar" and unit.lower() in ["т", "тонн", "ton"]:
                        vol = vol * 1000.0
                        unit = "кг"
                    elif phase == "metal" and unit.lower() in ["кг", "kg"]:
                        vol = vol / 1000.0
                        unit = "т"
                    elif phase == "concrete" and unit.lower() in ["л", "литр"]:
                        vol = vol / 1000.0
                        unit = "м3"

                    # 4. РАСЧЕТ ДЛИТЕЛЬНОСТИ И УНИВЕРСАЛЬНАЯ ПРОВЕРКА НА "УТОЧНЕНИЕ"
                    if vol <= 0:
                        vol = 0.0
                        unit = expected_unit or ""
                        duration = 1
                        # Добавляем пометку для ЛЮБОЙ работы, если объема нет
                        if "(объём уточняется)" not in task_name and "(объем уточняется)" not in task_name:
                            task_name = f"{task_name} (объём уточняется)"
                    else:
                        rate = WORK_RATES.get(phase, WORK_RATES["general"]).get("rate")
                        duration = max(1, int(-(-vol // (rate if rate is not None else 10.0)))) if rate else 1

                    # --- ЛОГИКА ПРЕДШЕСТВЕННИКОВ ---
                    pred_raw = str(r.get("predecessor", "")).strip()
                    pred_id = ""

                    # Если модель вернула "AUTO" или пустоту, берем предыдущую задачу этого же WBS
                    if not pred_raw or pred_raw.lower() == "null" or pred_raw.upper() == "AUTO":
                        pred_id = prev_aid_by_wbs.get(wbs, "")
                    else:
                        # Если модель вернула конкретный ID (например, A2000)
                        if re.match(r"^A\d+$", pred_raw):
                            pred_id = pred_raw
                        # Если модель вернула название задачи, ищем её ID
                        else:
                            for a in activities:
                                if pred_raw.lower() in a["Activity Name"].lower():
                                    pred_id = a["Activity ID"]
                                    break
                            # Если не нашли по названию, ставим предыдущую
                            if not pred_id:
                                pred_id = prev_aid_by_wbs.get(wbs, "")

                    current_aid = f"A{aid_counter}"
                    activities.append({
                        "Activity ID": current_aid, "WBS": wbs, "Activity Name": task_name,
                        "Original Duration": duration, "Activity Status": "Not Started",
                        "Category": phase, "Predecessor": pred_id, "Volume": vol, "Unit": unit
                    })
                    prev_aid_by_wbs[wbs] = current_aid
                    aid_counter += 10

                if not activities: raise ValueError("Не удалось сопоставить элементы с работами.")
                st.session_state.p6_df = pd.DataFrame(activities)

            except Exception as e:
                st.error(f"Ошибка Агента #2: {e}")
                if p6_output: st.code(p6_output)

    if st.session_state.p6_df is not None:
        st.success("🎯 Структура Primavera P6 сгенерирована!")
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
        st.caption("⚠️ Формат — черновик для маппинга колонок в P6. Для прямого импорта понадобится XER/XML-экспорт.")
        # --- БЛОК СТАТИСТИКИ ---
        st.markdown("---")
        st.subheader("📊 Статистика расходов (VseGPT)")

        col1, col2, col3 = st.columns(3)
        col1.metric("Токенов на вход", f"{st.session_state.total_prompt_tokens:,}")
        col2.metric("Токенов на выход", f"{st.session_state.total_completion_tokens:,}")
        col3.metric("Примерная стоимость", f"{st.session_state.total_cost_rub:.2f} ₽")

        st.caption("Стоимость рассчитана по тарифам VseGPT. Точные цифры смотрите в личном кабинете.")

        if st.button("🔄 Сбросить статистику"):
            st.session_state.total_prompt_tokens = 0
            st.session_state.total_completion_tokens = 0
            st.session_state.total_cost_rub = 0.0
            st.rerun()

#streamlit run one_test.py