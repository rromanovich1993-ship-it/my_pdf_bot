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

st.set_page_config(page_title="ИИ-Сметчик (DeepSeek)", layout="wide")
st.title("🏗️ Мультиагентная система: Чертеж ➔ Смета ➔ Primavera P6")

POPPLER_PATH = os.getenv("POPPLER_PATH")

st.sidebar.header("🔑 Авторизация DeepSeek")

# Ключ теперь ищем в APIDEPSIK_KEY
env_key = (os.getenv("APIDEPSIK_KEY") or "").strip()

if env_key:
    api_key = env_key
    st.sidebar.success("✅ Ключ DeepSeek загружен из .env")
else:
    api_key = st.sidebar.text_input("Введите DeepSeek API Key:", type="password").strip()

model_vision = st.sidebar.selectbox(
    "Агент #1 (Зрение):", ["deepseek-flash"]
)
model_planner = st.sidebar.selectbox(
    "Агент #2 (Планировщик):", ["deepseek-flash", "deepseek-v4-pro"]
)

max_pages = st.sidebar.number_input(
    "Сколько страниц PDF читать (0 = все):", min_value=0, value=3, step=1
)

if not api_key:
    st.warning("Пожалуйста, добавьте APIDEPSIK_KEY в файл .env или введите его на боковой панели.")
    st.stop()

# Настраиваем клиент на DeepSeek
client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com",
    timeout=120,
    max_retries=2,
)

# Счетчик токенов
if "total_tokens_used" not in st.session_state:
    st.session_state.total_tokens_used = 0
if "prompt_tokens_used" not in st.session_state:
    st.session_state.prompt_tokens_used = 0
if "completion_tokens_used" not in st.session_state:
    st.session_state.completion_tokens_used = 0

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

def call_model(model: str, messages, max_tokens: int = 8192):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
        max_tokens=max_tokens,
        # response_format={"type": "json_object"} # УБРАЛИ, чтобы не ломать зрение
    )
    if isinstance(response, str):
        return response.strip()
    choice = response.choices[0]
    content = (choice.message.content or "").strip()
    if hasattr(response, "usage") and response.usage:
        st.session_state.prompt_tokens_used += response.usage.prompt_tokens
        st.session_state.completion_tokens_used += response.usage.completion_tokens
        st.session_state.total_tokens_used += response.usage.total_tokens
    finish_reason = getattr(choice, "finish_reason", None)
    if finish_reason == "length":
        raise RuntimeError(f"Ответ модели оборван по лимиту max_tokens={max_tokens}.")
    return content

WORK_RATES = {
    "concrete": {"rate": 14.0}, "formwork": {"rate": None}, "rebar": {"rate": 500.0},
    "insulation": {"rate": 50.0}, "waterproofing": {"rate": 50.0}, "backfill": {"rate": 20.0},
    "earthwork": {"rate": 60.0}, "demolition": {"rate": 10.0}, "pipeline": {"rate": 30.0},
    "installation": {"rate": 3.0}, "finishing": {"rate": 40.0}, "masonry": {"rate": 5.0},
    "roofing": {"rate": 30.0}, "electrical": {"rate": 30.0}, "hvac": {"rate": 10.0},
    "general": {"rate": 10.0},
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
    u = str(u).strip().lower()
    return UNIT_NORMALIZE.get(u, u or "м3")

def to_kg(vol: float, unit: str) -> float:
    return vol * 1000.0 if unit == "т" else vol

def clean_wbs(w) -> str:
    s = str(w or "").strip()
    if not s: return "Общие работы"
    parts = re.split(r"[/|—–:]", s)
    return parts[0].strip() or s

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
    "demolition": 0, "earthwork": 1, "formwork": 2, "rebar": 3,
    "concrete": 4, "waterproofing": 5, "insulation": 5, "backfill": 6,
    "installation": 7, "pipeline": 7, "masonry": 8, "roofing": 9,
    "finishing": 10, "electrical": 11, "hvac": 11, "general": 99,
}

if "raw_estimate_df" not in st.session_state: st.session_state.raw_estimate_df = None
if "p6_df" not in st.session_state: st.session_state.p6_df = None
if "last_file_hash" not in st.session_state: st.session_state.last_file_hash = None

PROMPT_TABLES = """Ты — эксперт по таблицам. Найди на чертеже ЛЮБЫЕ таблицы (Спецификация, Ведомость).
Извлеки данные и верни JSON-массив:
[{"wbs_parent": "Марка", "element": "Наименование", "volume": 10.5, "unit": "м3", "category": "concrete", "note": "из таблицы"}]

СТРОГИЕ ПРАВИЛА:
1. НЕ создавай строки со словами "Итого", "Всего", "Итог". Только конкретные материалы.
2. В колонке "category" используй ТОЛЬКО эти значения: rebar (арматура), concrete (бетон), formwork (опалубка), waterproofing (гидроизоляция), installation (монтаж), earthwork (земля), backfill (засыпка), masonry (кладка), general (прочее).
3. Поле "note" — максимум 3 слова.
4. Если таблиц нет — верни [].

ФОРМАТ ОТВЕТА — строго JSON-массив объектов с полями:
wbs_parent, element, count, volume, unit, category, source_page, note.
Не добавляй markdown, не пиши пояснений до или после JSON."""

PROMPT_NOTES = """Ты — эксперт по текстовым указаниям на чертежах.
Найди на чертеже ВСЕ текстовые примечания, ТУ (Технические условия), выноски (например: "Бетон В25", "Гидроизоляция 2 слоя", "Уплотнение послойно").
Верни JSON-массив:
[{"wbs_parent": "Марка", "element": "Что сказано", "volume": null, "unit": "м2", "category": "waterproofing", "note": "примечание"}]
Если в примечаниях нет цифр, ставь volume: null. Игнорируй сами таблицы.

ФОРМАТ ОТВЕТА — строго JSON-массив объектов с полями:
wbs_parent, element, count, volume, unit, category, source_page, note.
Не добавляй markdown, не пиши пояснений до или после JSON."""

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
            try:
                # --- 1. ПОДГОТОВКА КАРТИНОК ---
                content_images = []
                if uploaded_file.type == "application/pdf":
                    last_page = None if max_pages == 0 else max_pages
                    images = convert_from_bytes(
                        file_bytes, dpi=150, first_page=1, last_page=last_page,
                        poppler_path=POPPLER_PATH,
                    )
                    if not images:
                        raise ValueError("PDF не содержит страниц")

                    for page_img in images:
                        page_img = shrink_image_to_limit(page_img)
                        buf = io.BytesIO()
                        page_img.save(buf, format="JPEG", quality=90)
                        content_images.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{encode_image(buf.getvalue())}"},
                        })
                        del page_img

                # --- ЦИКЛ ПО СТРАНИЦАМ (АГЕНТ #1 ЧИТАЕТ ПО ОДНОЙ) ---
                all_data_tables = []
                all_data_notes = []

                progress_bar = st.progress(0)
                status_text = st.empty()

                for i, img_content in enumerate(content_images):
                    page_num = i + 1
                    status_text.text(f"Читаем страницу {page_num} из {len(content_images)}...")

                    # Добавляем в промпт просьбу указать номер страницы
                    # (передаем его через f-строку в system content)
                    prompt_tables_with_page = PROMPT_TABLES.replace(
                        "source_page", f"{page_num}"
                    )

                    messages_tables = [
                        {"role": "system", "content": prompt_tables_with_page},
                        {"role": "user", "content": [img_content]}
                    ]
                    try:
                        out_tables = call_model(model_vision, messages_tables)
                        data = extract_json_array(out_tables) if out_tables else []
                        # Принудительно ставим страницу, даже если модель забыла
                        for row in data:
                            row["source_page"] = page_num
                        all_data_tables.extend(data)
                    except Exception as e:
                        st.warning(f"Страница {page_num} (таблицы): {e}")

                    # То же для примечаний
                    messages_notes = [
                        {"role": "system", "content": PROMPT_NOTES},
                        {"role": "user", "content": [img_content]}
                    ]
                    try:
                        out_notes = call_model(model_vision, messages_notes)
                        data = extract_json_array(out_notes) if out_notes else []
                        for row in data:
                            row["source_page"] = page_num
                        all_data_notes.extend(data)
                    except Exception as e:
                        st.warning(f"Страница {page_num} (примечания): {e}")

                    progress_bar.progress((i + 1) / len(content_images))

                # --- СЛИЯНИЕ ВСЕХ СТРАНИЦ ---
                all_data = all_data_tables + all_data_notes

                if not all_data:
                    st.warning("Ничего не найдено ни в таблицах, ни в примечаниях.")
                else:
                    df = pd.DataFrame(all_data)

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
                # Для отладки выведем, что ответила модель
                if 'out_tables' in locals() and out_tables:
                    st.caption("Ответ по таблицам:")
                    st.code(out_tables)
                if 'out_notes' in locals() and out_notes:
                    st.caption("Ответ по примечаниям:")
                    st.code(out_notes)

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
            # 1. ГРУППИРУЕМ ВЕДОМОСТЬ ПО WBS (МАРКАМ)
            grouped = edited_df.groupby("wbs_parent")
            total_groups = len(grouped)

            all_activities = []
            aid_counter = 2000

            progress_bar = st.progress(0)
            status_text = st.empty()

            # 2. ИДЕМ ПО КАЖДОМУ КОНСТРУКТИВУ ОТДЕЛЬНО
            for i, (wbs_name, group_df) in enumerate(grouped):
                status_text.text(f"Планируем: {wbs_name} ({i + 1} из {total_groups})...")

                # Отправляем ИИ только строки этого конструктива
                excel_text_context = group_df[["element", "volume", "unit", "category"]].to_string(index=False)

                prompt = f"""Ты — Агент-Планировщик Primavera P6.
    Изучи ведомость объёмов для конструктива: "{wbs_name}"

    {excel_text_context}

    Твоя задача — сформировать иерархическую структуру работ (WBS) ТОЛЬКО ДЛЯ ЭТОГО КОНСТРУКТИВА.
    Создай задачи в ТЕХНОЛОГИЧЕСКОЙ ПОСЛЕДОВАТЕЛЬНОСТИ (от земляных работ до отделки).

    ПРАВИЛА:
    1. Поле "wbs_parent" пиши строго: "{wbs_name}"
    2. НЕ ДРОБИ арматуру (все диаметры в ОДНУ задачу). Бетон — ДВЕ задачи (подготовка + конструктив).
    3. Поле "phase" — тип работы: earthwork, formwork, rebar, concrete, insulation, waterproofing, backfill, installation, demolition, pipeline, finishing, masonry, roofing, electrical, hvac, general.
    4. Поле "predecessor" — оставь null, мы сами проставим связи в коде.

    Верни строго JSON-массив:
    [{{"wbs_parent": "{wbs_name}", "task_name": "Название задачи", "phase": "rebar", "predecessor": null}}]
    """
                try:
                    # Вызываем модель (сохранив счетчик токенов)
                    p6_output = call_model(model_planner, messages=[{"role": "user", "content": prompt}])
                    rows = extract_json_array(p6_output)

                    # 3. СКЛЕИВАЕМ И ПРОСТАВЛЯЕМ ID И ПРЕДШЕСТВЕННИКОВ
                    prev_aid = ""
                    # ... (внутри цикла планирования задач) ...
                    for r in rows:
                        current_aid = f"A{aid_counter}"
                        phase = str(r.get("phase", "general")).strip().lower()

                        # Берем объем из ведомости по фазе (упрощенно)
                        vol = 0.0
                        unit = EXPECTED_UNIT.get(phase, "")
                        source_sheet = "—"

                        matches = group_df[group_df["category"] == phase]
                        if not matches.empty:
                            vol = matches.iloc[0]["volume"] if pd.notna(matches.iloc[0]["volume"]) else 0.0
                            unit = normalize_unit(matches.iloc[0]["unit"])

                            # ИЩЕМ ЛИСТ, С КОТОРОГО ПРИШЛА ЭТА ФАЗА
                            if "source_page" in matches.iloc[0]:
                                source_sheet = f"Лист {matches.iloc[0]['source_page']}"

                        # ... (расчет duration) ...

                        all_activities.append({
                            "Activity ID": current_aid,
                            "WBS": wbs_name,
                            "Activity Name": str(r.get("task_name", "")).strip(),
                            "Original Duration": duration,
                            "Activity Status": "Not Started",
                            "Category": phase,
                            "Predecessor": prev_aid,
                            "Volume": vol,
                            "Unit": unit,
                            "Source Sheet": source_sheet,  # <--- ДОБАВИЛИ ЛИСТ
                        })
                        prev_aid = current_aid
                        aid_counter += 10

                except Exception as e:
                    st.warning(f"Ошибка планирования для {wbs_name}: {e}")

                progress_bar.progress((i + 1) / total_groups)

            # 4. ФИНАЛ
            if not all_activities:
                st.error("Агент #2 не смог создать ни одной задачи.")
            else:
                st.session_state.p6_df = pd.DataFrame(all_activities)
                st.success(f"🎯 Структура Primavera P6 сгенерирована! Задач: {len(all_activities)}")

    if st.session_state.p6_df is not None:
        st.success("🎯 Структура Primavera P6 сгенерирована!")
        st.subheader("🚀 Итоговая структура задач:")
        st.dataframe(st.session_state.p6_df, use_container_width=True)

        # --- БЛОК ВЫВОДА СТАТИСТИКИ ТОКЕНОВ ---
        st.markdown("---")
        st.subheader("📊 Замер расхода токенов (DeepSeek)")

        if st.session_state.total_tokens_used > 0:
            col1, col2, col3 = st.columns(3)
            col1.metric("Всего токенов", f"{st.session_state.total_tokens_used:,}")
            col2.metric("Входящие (Prompt)", f"{st.session_state.prompt_tokens_used:,}")
            col3.metric("Исходящие (Completion)", f"{st.session_state.completion_tokens_used:,}")

            # Оценка стоимости (очень грубая, по средним ценам DeepSeek Flash)
            cost = (st.session_state.prompt_tokens_used / 1_000_000 * 0.15) + \
                   (st.session_state.completion_tokens_used / 1_000_000 * 0.60)
            st.caption(f"Примерная стоимость этого прогона: **${cost:.4f}** (по тарифам DeepSeek Flash)")
        else:
            st.info("Статистика токенов пуста. Запустите Агента #1 и #2, чтобы увидеть расход.")

        # --- СКАЧИВАНИЕ ИТОГОВОГО ФАЙЛА ---
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

    # Кнопка сброса счетчика
    if st.sidebar.button("🔄 Сбросить счетчик токенов"):
        st.session_state.total_tokens_used = 0
        st.session_state.prompt_tokens_used = 0
        st.session_state.completion_tokens_used = 0
        st.rerun()

# streamlit run one_test1.py
