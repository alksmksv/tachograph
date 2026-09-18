import io
import numpy as np
import pandas as pd
import streamlit as st

# ==============================================================================
# БЛОК 1: ИНИЦИАЛИЗАЦИЯ И ГЛОБАЛЬНЫЕ СТИЛИ
# ==============================================================================

# Настраиваем конфигурацию страницы Streamlit: делаем макет широким (wide) 
# и задаем заголовок вкладки в браузере.
st.set_page_config(layout="wide", page_title="Мониторинг смен РТО")

# Инъекция кастомных CSS-стилей для управления внешним видом элементов интерфейса
st.markdown(
    """
<style>
    /* Основной фон всего приложения (светло-серый оттенок) */
    .stApp { background-color: #F1F5F9; }
    
    /* Стилизация боковой панели (Sidebar): белый фон и правая граница */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #CBD5E1 !important;
    }
    /* Внутренние отступы контейнера сайдбара */
    section[data-testid="stSidebar"] .block-container {
        padding: 0.4rem !important;
    }

    /* Визуальная карточка-контейнер для каждого фильтра в сайдбаре */
    .filter-card {
        border: 1px solid #CBD5E1;
        background-color: #F8FAFC;
        border-radius: 5px;
        padding: 4px 6px;
        margin-bottom: 5px;
    }
    /* Заголовок внутри карточки фильтра */
    .filter-card-title {
        font-size: 11px;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 2px;
    }

    /* Оформление выбранных тегов (плашек) в мультиселектах */
    [data-baseweb="tag"] {
        background-color: #2563EB !important;
        border-radius: 3px !important;
        height: 18px !important;
        margin: 1px !important;
        padding-left: 4px !important;
        padding-right: 4px !important;
    }
    [data-baseweb="tag"] * {
        color: #FFFFFF !important;
        fill: #FFFFFF !important;
        font-size: 10px !important;
    }

    /* Универсальные стили рамок, шрифтов и полей для инпутов */
    div[data-baseweb="select"] > div, 
    div[data-testid="stNumberInput"] input,
    div[data-testid="stDateInput"] input {
        border: 1px solid #94A3B8 !important;
        border-radius: 4px !important;
        background-color: #FFFFFF !important;
        font-size: 11px !important;
        min-height: 26px !important;
        padding-top: 0px !important;
        padding-bottom: 0px !important;
    }

    /* Скрываем стандартные текстовые метки (label) над элементами в сайдбаре для компактности */
    div[data-testid="stSidebar"] label {
        display: none !important;
    }
    
    /* Стилизация кнопок внутри боковой панели */
    div[data-testid="stSidebar"] button {
        background-color: #1E3A8A !important;
        color: white !important;
        border-radius: 3px !important;
        border: none !important;
        font-size: 10px !important;
        height: 20px !important;
        min-height: 20px !important;
        padding: 0px !important;
    }
    div[data-testid="stSidebar"] button:hover {
        background-color: #1E40AF !important;
    }
    
    /* Главный заголовок разделов интерфейса */
    .main-header {
        font-size: 18px;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 10px;
    }
</style>
""",
    unsafe_allow_html=True,
)


def get_cell_style(color_type):
    """
    Функция возвращает CSS-стиль оформления ячейки таблицы 
    в зависимости от типа зоны нарушения или статуса РТО.
    """
    if color_type == "blue":
        return "background-color: #DBEAFE; font-weight: bold; color: #1E3A8A;"
    elif color_type == "yellow":
        return "background-color: #FEF9C3; color: #713F12;"
    elif color_type == "orange":
        return "background-color: #FFEDD5; color: #9A3412; font-weight: bold;"
    elif color_type == "green":
        return "background-color: #DCFCE7; font-weight: bold; color: #166534;"
    elif color_type == "red":
        return "background-color: #FEE2E2; font-weight: bold; color: #991B1B;"
    elif color_type == "critical":
        return "background-color: #7F1D1D; font-weight: bold; color: #FFFFFF;"
    return ""


# ==============================================================================
# БЛОК 2: ЯДРО ОБРАБОТКИ ДАННЫХ И РАСЧЕТ РТО С ПАКЕТНЫМ ПОГАШЕНИЕМ
# ==============================================================================

@st.cache_data(show_spinner="Обработка файла и проверка РТО...")
def process_file_fast(file_bytes, file_name):
    """
    Основная функция загрузки, очистки и расчетов РТО. 
    Принимает байты файла и возвращает обработанный DataFrame.
    """
    if file_name.endswith(".csv"):
        df = pd.read_csv(io.BytesIO(file_bytes))
    else:
        df = pd.read_excel(io.BytesIO(file_bytes), engine="openpyxl")

    # Формируем штампы времени начала и конца активностей
    df["start_datetime"] = pd.to_datetime(
        df["start_date"].astype(str) + " " + df["start_time"].astype(str)
    )
    df["end_datetime"] = pd.to_datetime(
        df["end_date"].astype(str) + " " + df["end_time"].astype(str)
    )

    if "group" not in df.columns:
        df["group"] = "Н/Д"

    if "country" not in df.columns:
        df["country"] = "N/A"
    else:
        df["country"] = df["country"].fillna("N/A").astype(str).str.upper().str.strip()

    df = df.sort_values(
        ["vehicle_name", "driver_name", "group", "start_datetime"]
    ).reset_index(drop=True)

    act_type = df["activity_type"].astype(str).str.upper()
    dur_min = df["duration_minutes"].fillna(0)

    # Определяем полноценный отдых (RESTING или CARDLESS от 9 часов / 540 минут)
    is_rest = act_type.isin(["RESTING", "CARDLESS"]) & (dur_min >= 540)

    change_group = (
        (df["vehicle_name"] != df["vehicle_name"].shift())
        | (df["driver_name"] != df["driver_name"].shift())
        | (df["group"] != df["group"].shift())
    )
    df["shift_id"] = (is_rest | change_group).cumsum()

    df["rest_hours"] = np.where(is_rest, (dur_min / 60), np.nan)
    df["rest_country"] = np.where(is_rest, df["country"], np.nan)
    df["pause_start_dt"] = np.where(is_rest, df["start_datetime"], pd.NaT)
    df["pause_end_dt"] = np.where(is_rest, df["end_datetime"], pd.NaT)

    rest_subset = df[is_rest].copy()
    if not rest_subset.empty:
        idx_max = rest_subset.groupby("shift_id")["rest_hours"].idxmax()
        rest_filtered = rest_subset.loc[idx_max]
    else:
        rest_filtered = rest_subset

    rest_before_hours = rest_filtered.set_index("shift_id")["rest_hours"]
    rest_before_country = rest_filtered.set_index("shift_id")["rest_country"]
    pause_start_before = rest_filtered.set_index("shift_id")["pause_start_dt"]
    pause_end_before = rest_filtered.set_index("shift_id")["pause_end_dt"]

    work_df = df[~is_rest].copy()
    if work_df.empty:
        return pd.DataFrame()

    agg_df = work_df.groupby(
        ["shift_id", "vehicle_name", "driver_name", "group"], as_index=False
    ).agg(
        shift_start=("start_datetime", "min"),
    )

    agg_df["dt_date"] = agg_df["shift_start"].dt.date
    agg_df["date"] = agg_df["shift_start"].dt.strftime("%Y-%m-%d")

    days_ru = {0: "ПН", 1: "ВТ", 2: "СР", 3: "ЧТ", 4: "ПТ", 5: "СБ", 6: "ВС"}
    agg_df["weekday"] = agg_df["shift_start"].dt.dayofweek.map(days_ru)

    extracted_code = agg_df["vehicle_name"].astype(str).str[1:3].str.upper()
    agg_df["vehicle_country"] = np.where(
        extracted_code.isin(["CZ", "SK"]), extracted_code, "Other"
    )

    agg_df["rest_before_shift_hours"] = agg_df["shift_id"].map(rest_before_hours)
    agg_df["rest_country_before_shift"] = (
        agg_df["shift_id"].map(rest_before_country).fillna("N/A")
    )
    agg_df["pause_start"] = agg_df["shift_id"].map(pause_start_before)
    agg_df["pause_end"] = agg_df["shift_id"].map(pause_end_before)

    agg_df["week_start"] = agg_df["shift_start"].apply(
        lambda x: (x - pd.Timedelta(days=x.dayofweek)).normalize()
    )

    processed_rows = []
    
    # Цикл обработки по каждой машине (аккумулируем очередь долгов)
    for v_name, v_group in agg_df.sort_values(["vehicle_name", "shift_start"]).groupby("vehicle_name"):
        v_group = v_group.reset_index(drop=True)
        active_debts = []  # Очередь накопленных долгов
        
        for idx, row in v_group.iterrows():
            h = row["rest_before_shift_hours"]
            p_end = row["pause_end"]
            shift_dt = row["shift_start"]
            
            # Рулетка 4 недель (28 дней): удаляем долги старше 28 дней
            four_weeks_ago_limit = shift_dt - pd.Timedelta(days=28)
            active_debts = [d for d in active_debts if d["created_at"] >= four_weeks_ago_limit]
            
            color_type = ""
            display_str = f"{h:.2f}" if pd.notna(h) else ""
            violation_description = ""
            
            if pd.isna(h):
                row["status_color"] = ""
                row["display_text"] = ""
                row["violation_description"] = ""
                debt_bal = float(sum(d["debt_hours"] for d in active_debts))
                row["debt_balance"] = debt_bal
                if active_debts and debt_bal > 0:
                    oldest_debt = active_debts[0]
                    row["debt_days_counter"] = (shift_dt.normalize() - oldest_debt["created_at"].normalize()).days + 1
                else:
                    row["debt_days_counter"] = 0
                processed_rows.append(row)
                continue

            end_weekday = p_end.dayofweek if pd.notna(p_end) else shift_dt.dayofweek
            is_weekend_end = end_weekday in [0, 6]  # Вс или Пн
            is_midweek_end = end_weekday in [1, 2, 3, 4, 5]  # Вт – Сб

            total_debt_hours = sum(d["debt_hours"] for d in active_debts)

            # 1. Критическое нарушение: пауза меньше 9 часов
            if h < 9.0:
                color_type = "critical"
                violation_description = "Критическое нарушение: Слишком короткая суточная пауза (менее 9 часов)"

            # 2. Сокращенная суточная пауза: от 9 до 11 часов
            elif 9.0 <= h < 11.0:
                if is_weekend_end:
                    color_type = "red"
                    violation_description = "Нарушение: Сокращенная суточная пауза в воскресенье/понедельник"
                else:
                    color_type = "yellow"
                    violation_description = "Сокращенная суточная пауза (будни)"

            # 3. Пауза от 11 часов и более (Разделение по дням недели и пакетное погашение)
            elif h >= 11.0:
                if is_midweek_end:
                    # Вт, Ср, Чт, Пт, Сб: схема 11+ (возраст долгов ДО 7 дней включительно)
                    req_h = 11.0 + total_debt_hours
                    max_debt_age_11 = 7
                    all_debts_are_fresh = all(
                        (shift_dt.normalize() - d["created_at"].normalize()).days <= max_debt_age_11 
                        for d in active_debts
                    )
                    
                    if active_debts and h >= req_h and all_debts_are_fresh:
                        extra_h = h - 11.0
                        color_type = "green"
                        display_str = f"11+{extra_h:.2f}"
                        violation_description = "Компенсация всего пакета долгов (11+)"
                        active_debts.clear()  # ПАКЕТНОЕ ПОГАШЕНИЕ ВСЕЙ СУММЫ РАЗОМ
                    else:
                        color_type = ""
                        violation_description = ""
                
                else:  # Это Вс или Пн
                    if h < 24.0:
                        color_type = "red"
                        violation_description = "Нарушение: Пауза менее 24 часов на выходных"
                    elif 24.0 <= h < 45.0:
                        # Сокращенная еженедельная на выходных — рождает новый долг
                        color_type = "orange"
                        debt_val = 45.0 - h  
                        violation_description = "Сокращенная еженедельная пауза (создан долг)"
                        weekend_end_ref = p_end if pd.notna(p_end) else shift_dt
                        
                        active_debts.append({
                            "debt_hours": debt_val, 
                            "source_weekend_end": weekend_end_ref,
                            "created_at": shift_dt
                        })
                    else:  # h >= 45.0 на выходных
                        # Вс, Пн: схема 45+ (возраст долгов ДО 28 дней)
                        req_h = 45.0 + total_debt_hours
                        max_debt_age_45 = 28
                        all_debts_are_valid = all(
                            (shift_dt.normalize() - d["created_at"].normalize()).days <= max_debt_age_45 
                            for d in active_debts
                        )
                        
                        if active_debts and h >= req_h and all_debts_are_valid:
                            extra_h = h - 45.0
                            color_type = "green"
                            display_str = f"45+{extra_h:.2f}"
                            violation_description = "Полноценная компенсация всего пакета долгов (45+)"
                            active_debts.clear()  # ПАКЕТНОЕ ПОГАШЕНИЕ ВСЕЙ СУММЫ РАЗОМ
                        else:
                            color_type = "blue"
                            violation_description = ""

            row["status_color"] = color_type
            row["display_text"] = display_str
            row["violation_description"] = violation_description
            
            debt_bal = float(sum(d["debt_hours"] for d in active_debts))
            row["debt_balance"] = debt_bal
            
            if active_debts and debt_bal > 0:
                oldest_debt = active_debts[0]
                row["debt_days_counter"] = (shift_dt.normalize() - oldest_debt["created_at"].normalize()).days + 1
            else:
                row["debt_days_counter"] = 0
                
            processed_rows.append(row)

    res_df = pd.DataFrame(processed_rows)
    res_df["pause_start"] = pd.to_datetime(res_df["pause_start"]).dt.strftime("%Y-%m-%d %H:%M").fillna("-")
    res_df["pause_end"] = pd.to_datetime(res_df["pause_end"]).dt.strftime("%Y-%m-%d %H:%M").fillna("-")

    cols = [
        "dt_date", "date", "weekday", "group", "vehicle_country", "vehicle_name",
        "driver_name", "rest_country_before_shift", "rest_before_shift_hours",
        "display_text", "status_color", "violation_description", "pause_start",
        "pause_end", "debt_balance", "debt_days_counter",
    ]
    return res_df[cols]


# ==============================================================================
# БЛОК 3: ВСПОМОГАТЕЛЬНЫЕ УТИЛИТЫ И ЭКСПОРТ
# ==============================================================================

def convert_df_to_excel(df):
    """Конвертирует DataFrame в байтовый Excel-файл для скачивания."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='РТО_Мониторинг')
    return output.getvalue()


# ==============================================================================
# БЛОК 4: ИНТЕРФЕЙС ПОЛЬЗОВАТЕЛЯ И ПАНЕЛЬ ФИЛЬТРОВ (SIDEBAR)
# ==============================================================================

with st.sidebar:
    st.markdown('<div class="main-header">Параметры и фильтры</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Загрузить файл данных", type=["csv", "xlsx"])
    if uploaded_file is None:
        st.info("Пожалуйста, загрузите файл с данными (CSV или XLSX) для начала работы.")
        st.stop()

    file_bytes = uploaded_file.getvalue()
    df = process_file_fast(file_bytes, uploaded_file.name)

    if df.empty:
        st.warning("В загруженном файле нет данных для анализа.")
        st.stop()

    df["dt_date"] = pd.to_datetime(df["dt_date"])

    st.markdown('<div class="filter-card">', unsafe_allow_html=True)
    st.markdown('<div class="filter-card-title">Раздел приложения</div>', unsafe_allow_html=True)
    page = st.radio(
        "Раздел", 
        ["Main (Таблица)", "Calendar (Матрица)", "Compensation (Отчет)"], 
        label_visibility="collapsed"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    min_d = df["dt_date"].min().date()
    max_d = df["dt_date"].max().date()
    
    st.markdown('<div class="filter-card">', unsafe_allow_html=True)
    st.markdown('<div class="filter-card-title">Период дат</div>', unsafe_allow_html=True)
    date_range = st.date_input("Период", [min_d, max_d], label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="filter-card">', unsafe_allow_html=True)
    st.markdown('<div class="filter-card-title">Быстрые фильтры</div>', unsafe_allow_html=True)
    only_with_debt = st.checkbox("Только с долгом (> 0 ч)")
    only_violations = st.checkbox("Только с нарушениями")
    st.markdown('</div>', unsafe_allow_html=True)

    all_groups = sorted(df["group"].dropna().unique().tolist())
    st.markdown('<div class="filter-card">', unsafe_allow_html=True)
    st.markdown('<div class="filter-card-title">Группа</div>', unsafe_allow_html=True)
    selected_groups = st.multiselect("Группа", all_groups, default=all_groups, label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

    all_veh_countries = sorted(df["vehicle_country"].dropna().unique().tolist())
    st.markdown('<div class="filter-card">', unsafe_allow_html=True)
    st.markdown('<div class="filter-card-title">Страна ТС</div>', unsafe_allow_html=True)
    selected_veh_countries = st.multiselect("Страна ТС", all_veh_countries, default=all_veh_countries, label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

    all_vehicles = sorted(df["vehicle_name"].dropna().unique().tolist())
    st.markdown('<div class="filter-card">', unsafe_allow_html=True)
    st.markdown('<div class="filter-card-title">Машина</div>', unsafe_allow_html=True)
    selected_vehicles = st.multiselect("Машина", all_vehicles, default=all_vehicles, label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

    all_drivers = sorted(df["driver_name"].dropna().unique().tolist())
    st.markdown('<div class="filter-card">', unsafe_allow_html=True)
    st.markdown('<div class="filter-card-title">Водитель</div>', unsafe_allow_html=True)
    selected_drivers = st.multiselect("Водитель", all_drivers, default=all_drivers, label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

    all_viols = sorted(df["violation_description"].replace("", np.nan).dropna().unique().tolist())
    st.markdown('<div class="filter-card">', unsafe_allow_html=True)
    st.markdown('<div class="filter-card-title">Тип нарушения</div>', unsafe_allow_html=True)
    selected_viols = st.multiselect("Нарушение", all_viols, default=all_viols, label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("Сбросить все фильтры"):
        st.rerun()

    filtered_df = df.copy()

    if len(date_range) == 2:
        start_d, end_d = date_range
        filtered_df = filtered_df[
            (filtered_df["dt_date"].dt.date >= start_d) & 
            (filtered_df["dt_date"].dt.date <= end_d)
        ]

    if only_with_debt:
        filtered_df = filtered_df[filtered_df["debt_balance"] > 0]
        
    if only_violations:
        filtered_df = filtered_df[filtered_df["violation_description"] != ""]

    filtered_df = filtered_df[
        filtered_df["group"].isin(selected_groups) &
        filtered_df["vehicle_country"].isin(selected_veh_countries) &
        filtered_df["vehicle_name"].isin(selected_vehicles) &
        filtered_df["driver_name"].isin(selected_drivers)
    ]


# ==============================================================================
# БЛОК 5: ОТРИСОВКА СТРАНИЦ ПРИЛОЖЕНИЯ
# ==============================================================================

if page == "Main (Таблица)":
    st.markdown('<div class="main-header">Мониторинг смен и соблюдения РТО</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Всего смен в отчете", len(filtered_df))
    with col2:
        st.metric("Активных машин", filtered_df["vehicle_name"].nunique())
    with col3:
        violations_count = len(filtered_df[filtered_df["violation_description"] != ""])
        st.metric("Всего нарушений/меток", violations_count)
    with col4:
        max_debt = filtered_df["debt_balance"].max() if not filtered_df.empty else 0
        st.metric("Макс. баланс долга (ч)", f"{max_debt:.2f}")

    st.markdown("---")

    display_table = filtered_df[[
        "date", "weekday", "group", "vehicle_country", "vehicle_name", 
        "driver_name", "rest_country_before_shift", "rest_before_shift_hours", 
        "display_text", "violation_description", "pause_start", "pause_end", 
        "debt_balance", "debt_days_counter"
    ]].copy()

    display_table.columns = [
        "Дата", "День", "Группа", "Страна ТС", "Машина", 
        "Водитель", "Страна отдыха", "Отдых (ч)", "Отображение", 
        "Описание / Нарушение", "Начало паузы", "Конец паузы", 
        "Долг (баланс)", "Дней долга"
    ]

    def style_dataframe_rows(row):
        original_idx = row.name
        if original_idx in filtered_df.index:
            c_type = filtered_df.loc[original_idx, "status_color"]
            return [get_cell_style(c_type)] * len(row)
        return [''] * len(row)

    styled_table = display_table.style.apply(style_dataframe_rows, axis=1)
    st.dataframe(styled_table, use_container_width=True, height=500)

    excel_bytes = convert_df_to_excel(filtered_df)
    st.download_button(
        label="📥 Скачать отфильтрованный отчет в Excel",
        data=excel_bytes,
        file_name="tacho_rt_monitoring_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

elif page == "Calendar (Матрица)":
    st.markdown('<div class="main-header">Календарная матрица отдыха и баланса долгов ТС</div>', unsafe_allow_html=True)
    
    if filtered_df.empty:
        st.warning("Нет данных для построения календарной матрицы по выбранным фильтрам.")
    else:
        filtered_df["date_str"] = filtered_df["dt_date"].dt.strftime("%Y-%m-%d")
        
        pivot_display = filtered_df.pivot_table(
            index="vehicle_name", 
            columns="date_str", 
            values="display_text", 
            aggfunc="first"
        ).fillna("")

        st.markdown("### Сводная таблица пауз по дням")
        st.dataframe(pivot_display, use_container_width=True, height=450)

elif page == "Compensation (Отчет)":
    st.markdown('<div class="main-header">Отчет по текущим компенсациям и накопленным долгам</div>', unsafe_allow_html=True)
    
    if filtered_df.empty:
        st.warning("Нет данных для формирования отчета по компенсациям.")
    else:
        comp_summary = filtered_df.groupby("vehicle_name").agg(
            Последняя_дата=("dt_date", "max"),
            Текущий_долг_часов=("debt_balance", "last"),
            Дней_накопления=("debt_days_counter", "last"),
            Последняя_группа=("group", "last"),
            Последний_водитель=("driver_name", "last")
        ).reset_index()

        comp_summary = comp_summary.sort_values(by="Текущий_долг_часов", ascending=False)
        st.dataframe(comp_summary, use_container_width=True, height=450)
