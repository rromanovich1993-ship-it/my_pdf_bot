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

load_dotenv()

st.set_page_config(page_title="ИИ-Сметчик (Универсальный)", layout="wide")
st.title("🏗️ Мультиагентная система: Чертёж ➔ Смета ➔ Primavera P6")

POPPLER_PATH = os.getenv("POPPLER_PATH")

st.sidebar.header("🔑 Авторизация VseGPT")

env_key = (os.getenv("VSEGPT_API_KEY") or os.getenv("API_KEY") or "").strip()

if env_key:
    api_key = env_key
    st.sidebar.success("✅ Ключ VseGPT загружен из .env")
else:
    api_key = st.sidebar.text_input("Введите VseGPT API Key:", type="password").strip()

model_vision = st.sidebar.selectbox(
    "Агент #1 (Зрение):", ["openai/gpt-4o", "openai/gpt-4o-mini"]
)
model_planner = st.sidebar.selectbox(
    "Агент #2 (Планировщик):", ["openai/gpt-4o", "openai/gpt-4o-mini"]
)

max_pages = st.sidebar.number_input(
    "Сколько страниц PDF читать (0 = все):", min_value=0, value=3, step=1
)

if not api_key:
    st.warning("Добавьте VSEGPT_API_KEY в .env или введите на панели.")
    st.stop()

client = OpenAI(
    api_key=api_key,
    base_url="https://api.vsegpt.ru/v1",
    timeout=120,
    max_retries=2,
)

if "total_tokens_used" not in st.session_state: st.session_state.total_tokens_used = 0
if "prompt_tokens_used" not in st.session_state: st.session_state.prompt_tokens_used = 0
if "completion_tokens_used" not in st.session_state: st.session_state.completion_tokens_used = 0

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

def extract_json_array(text: str, strict: bool = False):
    if not text: raise ValueError("Пустой ответ модели")
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    start = text.find("[")
    if start == -1: raise ValueError("В ответе модели нет JSON-массива")
    end = text.rfind("]") + 1
    if end > start:
        candidate = text[start:end]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
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
    if strict and depth > 0:
        raise ValueError("JSON-массив обрезан (незакрытый объект).")
    if not objects: raise ValueError("Не удалось извлечь ни одного объекта.")
    return objects

def call_model(model: str, messages, max_tokens: int = 4096):
    response = client.chat.completions.create(
        model=model, messages=messages, temperature=0, max_tokens=max_tokens,
    )
    if isinstance(response, str): return response.strip()
    choice = response.choices[0]
    content = (choice.message.content or "").strip()
    if hasattr(response, "usage") and response.usage:
        st.session_state.prompt_tokens_used += response.usage.prompt_tokens
        st.session_state.completion_tokens_used += response.usage.completion_tokens
        st.session_state.total_tokens_used += response.usage.total_tokens
    if getattr(choice, "finish_reason", None) == "length":
        raise RuntimeError(f"Ответ оборван по лимиту max_tokens={max_tokens}.")
    if not content:
        raise RuntimeError("Модель вернула пустой ответ.")
    return content

EXPECTED_UNIT = {
    "concrete": "м3", "formwork": "м2", "rebar": "кг", "insulation": "м2",
    "waterproofing": "м2", "backfill": "м3", "earthwork": "м3", "demolition": "м3",
    "pipeline": "м.п.", "installation": "шт", "finishing": "м2", "masonry": "м3",
    "roofing": "м2", "electrical": "м.п.", "hvac": "м.п.", "metal": "т", "general": "",
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
    except: pass
    u = str(u).strip().lower()
    return UNIT_NORMALIZE.get(u, u or "м3")

def clean_wbs(w) -> str:
    s = str(w or "").strip()
    if not s: return "Общие работы"
    parts = re.split(r"[/|—–:]", s)
    return parts[0].strip() or s

PHASE_ORDER = {
    "demolition": 0, "earthwork": 1, "formwork": 2, "rebar": 3,
    "concrete": 4, "waterproofing": 5, "insulation": 5, "backfill": 6,
    "installation": 7, "pipeline": 7, "masonry": 8, "metal": 8, "roofing": 9,
    "finishing": 10, "electrical": 11, "hvac": 11, "general": 99,
}

CATEGORY_KEYWORDS = {
    "rebar":         r"армат|армокаркас|стержн|сетк|A500|AIII|A400",
    "concrete":      r"бетон|В10|В15|В20|В25|В30|раствор",
    "formwork":      r"опалуб|щит|бортоснаст",
    "metal":         r"металлконстр|балк|швеллер|уголок|профиль|лист\s*г/к|труб.*стальн",
    "pipeline":      r"трубопровод|фасонн|труба\s*(технолог|стальн|ПНД|ПЭ)",
    "electrical":    r"кабел|провод|лоток|заземлен",
    "hvac":          r"воздуховод|вентиляц",
    "insulation":    r"утеплит|изоляц|минват",
    "waterproofing": r"гидроизол|мембран|бентонит",
    "masonry":       r"кирпич|блок\s*газобет|кладк",
    "roofing":       r"кровл|профнастил",
    "finishing":     r"штукатур|окраск|шпакл|облицов",
    "earthwork":     r"землян|грунт|песок|щебен|планировк",
    "demolition":    r"демонтаж",
    "installation":  r"оборудован|насос|задвижк|прибор|монтаж\s*оборуд",
}

def classify_categories_from_elements(elements):
    found = []
    for el in elements:
        for cat, pattern in CATEGORY_KEYWORDS.items():
            if re.search(pattern, el, re.IGNORECASE) and cat not in found:
                found.append(cat)
    return found if found else ["general"]

if "raw_estimate_df" not in st.session_state: st.session_state.raw_estimate_df = None
if "p6_df" not in st.session_state: st.session_state.p6_df = None
if "last_file_hash" not in st.session_state: st.session_state.last_file_hash = None

REFERENCE_WORKS = {
    "Земляные работы": [
        "Срезка растительного слоя", "Планировка территории", "Разработка грунта механизированным способом",
        "Доработка вручную", "Уплотнение грунта", "Обратная засыпка", "Погрузка и вывоз грунта"
    ],
    "Бетонные работы": [
        "Устройство бетонной подготовки", "Устройство опалубки", "Устройство армокаркаса",
        "Бетонирование фундаментной плиты", "Бетонирование фундамента", "Очистка бетонных поверхностей"
    ],
    "Гидроизоляция": ["Гидроизоляция фундамента (2 слоя)", "Укладка бентонитового мата"],
    "Металлоконструкции (КМ)": ["Монтаж металлических конструкций", "Монтаж опор", "Монтаж метизов"],
    "Трубопроводы": ["Монтаж трубопровода", "Сварка трубопровода", "Гидроиспытания трубопровода"],
    "Электрика (ЭС/ЭГ)": ["Монтаж кабельной продукции", "Монтаж заземляющих устройств", "Опрессовка концевых разделок"],
    "Антикоррозийная защита (АКЗ)": ["Абразивная очистка поверхности", "Нанесение грунтовки", "Окраска м/к поверхностей"],
    "Прочее": ["Установка бортовой опалубки", "Подливка безусадочным составом", "Комплекс пусконаладочных работ"]
}

PROMPT_TABLES = """Ты — OCR-сканер. Найди на чертеже таблицы (Спецификация, Ведомость, Кабельный журнал) и перепиши их в JSON.
ФОРМАТ:
[{"wbs_parent": "Марка", "element": "Наименование", "count": 10, "volume": 5.5, "unit": "м3", "category": "general", "source_page": "1", "note": "что видишь"}]

ЖЁСТКИЕ ПРАВИЛА:
1. КАТЕГОРИЯ — тип работы/материала. Выбирай СТРОГО из списка:
   "concrete", "rebar", "formwork", "metal", "pipeline", "electrical", "hvac", "insulation", "waterproofing", "masonry", "roofing", "finishing", "earthwork", "demolition", "installation", "general".
2. ЕСЛИ В ТАБЛИЦЕ ЕСТЬ КОЛОНКИ "МАССА ЕД." И "КОЛ-ВО":
   - "Массу ед." → в "volume" (например, 0.34).
   - "Кол-во" → в "count" (например, 64).
   - НЕ ПЕРЕМНОЖАЙ — это сделает код.
   - Если есть колонка "Итого" — пиши её в "note" (например, "21.76 кг").
3. ЕСЛИ ОДНА ЦИФРА (например, "Бетон В30 — 6.5 м3") — пиши её в "volume".
4. НЕ добавляй строки "Итого" / "Всего".
5. Номер листа из штампа пиши в "source_page".
ФОРМАТ ОТВЕТА — строго JSON-массив.
"""

PROMPT_NOTES = """Ты — эксперт по чертежам. Найди на листе ВСЕ текстовые примечания, ТУ, выноски.
Верни JSON-массив:
[{"wbs_parent": "Марка", "element": "Текст примечания", "volume": null, "unit": "", "category": "general", "source_page": "1", "note": "примечание"}]
ПРАВИЛА:
1. Номер листа из штампа пиши в "source_page".
2. Если есть цифры — вытащи в "volume" и "unit".
3. Не дублируй таблицы, пиши только текст.
ФОРМАТ ОТВЕТА — строго JSON-массив.
"""
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
                else:
                    img = Image.open(io.BytesIO(file_bytes))
                    img = shrink_image_to_limit(img)
                    buf = io.BytesIO()
                    img.save(buf, format="JPEG", quality=90)
                    content_images.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{encode_image(buf.getvalue())}"},
                    })

                all_data_tables = []
                all_data_notes = []
                progress_bar = st.progress(0)
                status_text = st.empty()

                for i, img_content in enumerate(content_images):
                    page_num = i + 1
                    status_text.text(f"Читаем страницу {page_num} из {len(content_images)}...")

                    messages_tables = [
                        {"role": "system", "content": PROMPT_TABLES},
                        {"role": "user", "content": [img_content]}
                    ]
                    try:
                        out_tables = call_model(model_vision, messages_tables)
                        if out_tables and "[" in out_tables:
                            data = extract_json_array(out_tables, strict=False)
                            for row in data:
                                row["pdf_page"] = page_num
                                if not row.get("source_page"):
                                    row["source_page"] = str(page_num)
                            all_data_tables.extend(data)
                    except Exception as e:
                        st.warning(f"Страница {page_num} (таблицы): {e}")

                    messages_notes = [
                        {"role": "system", "content": PROMPT_NOTES},
                        {"role": "user", "content": [img_content]}
                    ]
                    try:
                        out_notes = call_model(model_vision, messages_notes)
                        if out_notes and "[" in out_notes:
                            data = extract_json_array(out_notes, strict=False)
                            for row in data:
                                row["pdf_page"] = page_num
                                if not row.get("source_page"):
                                    row["source_page"] = str(page_num)
                            all_data_notes.extend(data)
                    except Exception as e:
                        st.warning(f"Страница {page_num} (примечания): {e}")

                    progress_bar.progress((i + 1) / len(content_images))

                all_data = all_data_tables + all_data_notes

                if not all_data:
                    st.warning("Ничего не найдено.")
                else:
                    df = pd.DataFrame(all_data)
                    for col in ("wbs_parent", "element", "count", "volume", "unit",
                                "category", "note", "source_page", "pdf_page"):
                        if col not in df.columns:
                            df[col] = None

                    bad_wbs = r"Спецификация|Ведомость|ГОСТ|Марка"
                    df = df[~df["wbs_parent"].astype(str).str.contains(bad_wbs, case=False, na=False) |
                            df["wbs_parent"].astype(str).str.match(r"^Марка\s*[\dA-Za-zА-Яа-я]")]

                    bad_el = r"мм|мкм|расход|плотность|t=|Ø\d+\s*мм"
                    df = df[~df["element"].astype(str).str.contains(bad_el, case=False, na=False)]

                    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
                    df["count"] = pd.to_numeric(df["count"], errors="coerce")
                    df = df[~df["wbs_parent"].astype(str).str.contains("Материалы|^\\d+$", na=False)]

                    mask_bad_fix = (
                            (df["category"] == "general") &
                            (df["element"].astype(str).str.contains("Фиксатор|Крепление|Болт", case=False, na=False))
                    )
                    df = df[~mask_bad_fix]

                    df["unit"] = df["unit"].apply(normalize_unit)
                    df["category"] = df["category"].fillna("").astype(str).str.strip().str.lower()

                    has_total_in_note = df["note"].astype(str).str.contains(r"итого|всего", case=False, na=False)
                    mask_small_vol = (
                            (df["volume"] < 10) &
                            (df["count"].notna()) &
                            (df["count"] > 1) &
                            (~has_total_in_note)
                    )
                    df.loc[mask_small_vol, "volume"] = df.loc[mask_small_vol, "volume"] * df.loc[
                        mask_small_vol, "count"]
                    df = df[df["volume"] < 50000]


                    def fix_volume_from_note(row):
                        if pd.notna(row["volume"]) and row["volume"] > 1000:
                            note_str = str(row.get("note", ""))
                            match = re.search(r"(\d+[.,]?\d*)\s*(кг|т|м3|м2|шт|м\.п\.)", note_str, re.IGNORECASE)
                            if match:
                                return float(match.group(1).replace(",", "."))
                        return row["volume"]


                    df["volume"] = df.apply(fix_volume_from_note, axis=1)
                    df = df.drop_duplicates(subset=["wbs_parent", "element", "volume"], keep="first")
                    df = df[df["volume"].notna() & (df["volume"] > 0)]
                    df = df.drop_duplicates(subset=["wbs_parent", "element"], keep="first").reset_index(drop=True)

                    mask_r = (df["category"] == "rebar")
                    df.loc[mask_r & (df["unit"] == "т"), "volume"] = df.loc[
                                                                         mask_r & (df["unit"] == "т"), "volume"] * 1000
                    df.loc[mask_r, "unit"] = "кг"

                    mask_m = (df["category"] == "metal")
                    df.loc[mask_m & (df["unit"] == "кг"), "volume"] = df.loc[mask_m & (
                                df["unit"] == "кг"), "volume"] / 1000
                    df.loc[mask_m, "unit"] = "т"

                    df["wbs_parent"] = df["wbs_parent"].apply(clean_wbs)
                    df["unit"] = df["unit"].apply(normalize_unit)

                    st.session_state.raw_estimate_df = df
                    st.session_state.p6_df = None
                    st.success("🤖 Агент #1 успешно сформировал ведомость (универсально)!")

            except Exception as e:
                st.error(f"Ошибка Агента #1: {e}")
                if 'out_tables' in locals() and out_tables:
                    st.code(out_tables)
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
            grouped = list(edited_df.groupby("wbs_parent"))
            total_groups = len(grouped)

            all_activities = []
            aid_counter = 2000

            progress_bar = st.progress(0)
            status_text = st.empty()

            NORM_HOURS_DB = {
                "Срезка растительного слоя": 0.05, "Планировка территории": 0.05,
                "Разработка грунта механизированным способом": 0.05, "Доработка вручную": 2.4,
                "Уплотнение грунта": 0.2, "Обратная засыпка": 0.2, "Погрузка и вывоз грунта": 0.001,
                "Устройство бетонной подготовки": 5.0, "Устройство опалубки": 2.0,
                "Устройство армокаркаса": 130.0, "Бетонирование фундаментной плиты": 5.0,
                "Бетонирование фундамента": 5.0, "Очистка бетонных поверхностей": 0.1,
                "Гидроизоляция фундамента (2 слоя)": 0.3, "Укладка бентонитового мата": 0.3,
                "Монтаж металлических конструкций": 77.0, "Монтаж опор": 1.0,
                "Монтаж метизов": 0.01, "Установка химических анкеров": 0.5,
                "Монтаж трубопровода": 4.533, "Сварка трубопровода": 15.0,
                "Гидроиспытания трубопровода": 40.0, "Монтаж фасонных элементов": 30.0,
                "Монтаж кабельной продукции": 0.6, "Монтаж заземляющих устройств": 12.0,
                "Опрессовка концевых разделок": 0.1,
                "Абразивная очистка поверхности": 5.4, "Нанесение грунтовки": 0.5,
                "Окраска м/к поверхностей": 0.5, "Нанесение огнезащиты": 1.0,
                "Установка бортовой опалубки": 2.0, "Подливка безусадочным составом": 15.0,
                "Комплекс пусконаладочных работ": 100.0,
            }

            for i, (wbs_name, group_df) in enumerate(grouped):
                status_text.text(f"Планируем: {wbs_name} ({i + 1} из {total_groups})...")

                categories_present = [c for c in group_df["category"].unique().tolist() if c and c != "general"]

                if not categories_present:
                    categories_present = classify_categories_from_elements(
                        group_df["element"].astype(str).tolist()
                    )

                reference_list = ""
                for section, works in REFERENCE_WORKS.items():
                    reference_list += f"\n- {section}: {', '.join(works)}"

                prompt = f"""Ты — профессиональный Агент-Планировщик (P6).
Для конструктива "{wbs_name}" есть следующие виды материалов (категории): {', '.join(categories_present)}.

Вот ЭТАЛОННЫЙ СПРАВОЧНИК РАБОТ:
{reference_list}

ПРАВИЛА ФОРМИРОВАНИЯ:
1. Выбери из справочника ТОЛЬКО те работы, которые соответствуют этим категориям.
2. Выстрой работы в СТРОГОЙ ТЕХНОЛОГИЧЕСКОЙ ПОСЛЕДОВАТЕЛЬНОСТИ.
3. НЕ ВЫДУМЫВАЙ названия. Бери их строго из справочника.
4. Для каждой задачи укажи "phase" (категорию).

Верни строго JSON-массив:
[{{"wbs_parent": "{wbs_name}", "task_name": "Название из справочника", "phase": "earthwork", "predecessor": null}}]
"""
                try:
                    p6_output = call_model(model_planner, messages=[{"role": "user", "content": prompt}])
                    rows = extract_json_array(p6_output, strict=False)

                    rows.sort(key=lambda r: PHASE_ORDER.get(str(r.get("phase", "general")).strip().lower(), 99))

                    prev_aid = ""
                    for r in rows:
                        current_aid = f"A{aid_counter}"
                        phase = str(r.get("phase", "general")).strip().lower()
                        task_name = str(r.get("task_name", "")).strip()

                        matches = group_df[group_df["category"] == phase]
                        vol = 0.0
                        unit = EXPECTED_UNIT.get(phase, "")
                        source_sheet = "—"

                        if not matches.empty:
                            if phase == "concrete":
                                if "подготовк" in task_name.lower():
                                    match_b10 = matches[
                                        matches["element"].astype(str).str.contains("В10|B10", case=False, na=False)]
                                    if not match_b10.empty:
                                        matches = match_b10
                                elif "плит" in task_name.lower() or "фундамент" in task_name.lower():
                                    match_b30 = matches[
                                        matches["element"].astype(str).str.contains("В30|B30|В25|B25", case=False, na=False)]
                                    if not match_b30.empty:
                                        matches = match_b30

                            vol_series = pd.to_numeric(matches["volume"], errors="coerce")
                            vol = vol_series.sum() if not vol_series.isna().all() else 0.0

                            raw_sheet = str(matches.iloc[0].get("source_page", "")).strip()
                            if raw_sheet and raw_sheet != "nan":
                                source_sheet = raw_sheet if "лист" in raw_sheet.lower() else f"Лист {raw_sheet}"

                        norm_rate = NORM_HOURS_DB.get(task_name)
                        if norm_rate is None:
                            st.warning(f"Нет нормы для '{task_name}', применено 10.0 ч/ед.")
                            norm_rate = 10.0

                        if vol <= 0:
                            duration = 1
                            total_hours = 0.0
                        else:
                            total_hours = vol * norm_rate
                            duration = max(1, int(round(total_hours / 8.0)))

                        pred = r.get("predecessor")
                        if not pred:
                            pred = prev_aid

                        all_activities.append({
                            "Activity ID": current_aid,
                            "WBS": wbs_name,
                            "Activity Name": task_name,
                            "Original Duration": duration,
                            "Activity Status": "Not Started",
                            "Category": phase,
                            "Predecessor": pred,
                            "Volume": vol,
                            "Unit": unit,
                            "Norm Hours": norm_rate,
                            "Total Hours": total_hours,
                            "Source Sheet": source_sheet,
                        })
                        prev_aid = current_aid
                        aid_counter += 10

                except Exception as e:
                    st.warning(f"Ошибка планирования для {wbs_name}: {e}")

                progress_bar.progress((i + 1) / total_groups)

            if not all_activities:
                st.error("Агент #2 не смог создать ни одной задачи.")
            else:
                clean_activities = [
                    a for a in all_activities
                    if a["Volume"] > 0 or a["Category"] in ["installation", "general"]
                ]
                st.session_state.p6_df = pd.DataFrame(clean_activities)
                st.success(f"🎯 Структура Primavera P6 сгенерирована! Задач: {len(clean_activities)}")

    if st.session_state.p6_df is not None:
        st.subheader("🚀 Итоговая структура задач:")
        st.dataframe(st.session_state.p6_df, use_container_width=True)

        st.markdown("---")
        st.subheader("📊 Замер расхода токенов (VseGPT)")

        if st.session_state.total_tokens_used > 0:
            col1, col2, col3 = st.columns(3)
            col1.metric("Всего токенов", f"{st.session_state.total_tokens_used:,}")
            col2.metric("Входящие (Prompt)", f"{st.session_state.prompt_tokens_used:,}")
            col3.metric("Исходящие (Completion)", f"{st.session_state.completion_tokens_used:,}")

            cost = (st.session_state.prompt_tokens_used / 1_000_000 * 5.0) + \
                   (st.session_state.completion_tokens_used / 1_000_000 * 15.0)
            st.caption(f"Примерная стоимость: **${cost:.4f}** (по тарифам VseGPT)")
        else:
            st.info("Статистика токенов пуста.")

        p6_buffer = io.BytesIO()
        with pd.ExcelWriter(p6_buffer, engine="openpyxl") as writer:
            st.session_state.p6_df.to_excel(writer, index=False, sheet_name="TASK")

        st.download_button(
            label="📥 Скачать черновик для импорта в Primavera P6 (Excel)",
            data=p6_buffer.getvalue(),
            file_name="Primavera_P6_Import_Final.xlsx",
            mime="application/v")







# streamlit run one_test1.py

# streamlit run one_test1.py
# streamlit run one_test1.py
