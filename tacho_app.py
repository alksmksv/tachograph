import io
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(layout="wide", page_title="Мониторинг смен РТО")

# ==============================================================================
# CSS-СТИЛИ
# ==============================================================================
st.markdown(
    """
<style>
    .stApp { background-color: #F1F5F9; }
    
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #CBD5E1 !important;
    }
    section[data-testid="stSidebar"] .block-container {
        padding: 0.4rem !important;
    }

    .filter-card {
        border: 1px solid #CBD5E1;
        background-color: #F8FAFC;
        border-radius: 5px;
        padding: 4px 6px;
        margin-bottom: 5px;
    }
    .filter-card-title {
        font-size: 11px;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 2px;
    }

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

    div[data-testid="stSidebar"] label {
        display: none !important;
    }
    
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


# ==============================================================================
# ВСПОМОГАТЕЛЬНАЯ ФУНКЦИЯ ЦВЕТОКОДИНГА
# ==============================================================================
def get_cell_style(color_type):
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
    return ""


# ==============================================================================
# ОБРАБОТКА ДАННЫХ И РТО-ЛОГИКА
# ==============================================================================
@st.cache_data(show_spinner="Обработка файла и проверка РТО...")
def process_file_fast(file_bytes, file_name):
    if file_name.endswith(".csv"):
        df = pd.read_csv(io.BytesIO(file_bytes))
    else:
        df = pd.read_excel(io.BytesIO(file_bytes), engine="openpyxl")

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

    days_ru = {
        0: "ПН", 1: "ВТ", 2: "СР", 
        3: "ЧТ", 4: "ПТ", 5: "СБ", 6: "ВС"
    }
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
    
    for v_name, v_group in agg_df.sort_values(["vehicle_name", "shift_start"]).groupby("vehicle_name"):
        v_group = v_group.reset_index(drop=True)
        active_debts = []
        
        for idx, row in v_group.iterrows():
            h = row["rest_before_shift_hours"]
            p_end = row["pause_end"]
            shift_dt = row["shift_start"]
            
            active_debts = [d for d in active_debts if d["expiry_date"] >= shift_dt]
            
            current_debt_val = round(sum(d["debt_hours"] for d in active_debts), 2)
            row["debt_balance"] = float(current_debt_val)
            
            color_type = ""
            display_str = f"{h:.2f}" if pd.notna(h) else ""
            
            if pd.isna(h):
                row["status_color"] = ""
                row["display_text"] = ""
                processed_rows.append(row)
                continue

            end_weekday = p_end.dayofweek if pd.notna(p_end) else shift_dt.dayofweek
            is_weekend_end = end_weekday in [0, 6]

            if 9.0 <= h < 11.0:
                if is_weekend_end:
                    color_type = "red"
                else:
                    w_start = row["week_start"]
                    week_shifts_mask = (v_group["week_start"] == w_start) & (v_group.index < idx)
                    short_weekday_count_prior = 0
                    for _, p_row in v_group[week_shifts_mask].iterrows():
                        ph = p_row["rest_before_shift_hours"]
                        pe = p_row["pause_end"]
                        pe_wd = pe.dayofweek if pd.notna(pe) else 0
                        if pd.notna(ph) and 9.0 <= ph < 11.0 and pe_wd not in [0, 6]:
                            short_weekday_count_prior += 1
                    
                    if short_weekday_count_prior >= 3:
                        color_type = "red"
                    else:
                        color_type = "yellow"

            elif 11.0 <= h < 24.0:
                if is_weekend_end:
                    color_type = "red"
                else:
                    can_compensate_weekday = False
                    if active_debts:
                        oldest_debt = active_debts[0]
                        prev_weekend_start = row["week_start"] - pd.Timedelta(days=2)
                        if oldest_debt.get("source_weekend_end") and oldest_debt["source_weekend_end"] >= prev_weekend_start:
                            can_compensate_weekday = True

                    req_h = 11.0 + (active_debts[0]["debt_hours"] if (active_debts and can_compensate_weekday) else 0)
                    
                    if active_debts and can_compensate_weekday and h >= req_h:
                        extra_h = h - 11.0
                        color_type = "green"
                        display_str = f"11+{extra_h:.2f}"
                        active_debts.pop(0)
                    else:
                        color_type = ""

            elif 24.0 <= h < 45.0:
                four_weeks_ago = shift_dt - pd.Timedelta(days=28)
                recent_mask = (v_group["shift_start"] >= four_weeks_ago) & (v_group.index < idx)
                recent_short_count = 0
                for _, p_row in v_group[recent_mask].iterrows():
                    ph = p_row["rest_before_shift_hours"]
                    if pd.notna(ph) and 24.0 <= ph < 45.0:
                        recent_short_count += 1

                if recent_short_count >= 2:
                    color_type = "red"
                else:
                    color_type = "orange"

                debt_val = 45.0 - h
                expiry_dt = shift_dt + pd.Timedelta(days=21)
                weekend_end_ref = p_end if pd.notna(p_end) else shift_dt
                active_debts.append({
                    "debt_hours": debt_val, 
                    "expiry_date": expiry_dt,
                    "source_weekend_end": weekend_end_ref
                })

            elif h >= 45.0:
                if active_debts:
                    oldest_debt = active_debts[0]
                    req_h = 45.0 + oldest_debt["debt_hours"]
                    if h >= req_h:
                        extra_h = h - 45.0
                        color_type = "green"
                        display_str = f"45+{extra_h:.2f}"
                        active_debts.pop(0)
                    else:
                        color_type = "blue"
                else:
                    color_type = "blue"

            active_debts = [d for d in active_debts if d["expiry_date"] >= shift_dt]

            row["status_color"] = color_type
            row["display_text"] = display_str
            processed_rows.append(row)

    res_df = pd.DataFrame(processed_rows)
    
    res_df["pause_start"] = pd.to_datetime(res_df["pause_start"]).dt.strftime("%Y-%m-%d %H:%M").fillna("-")
    res_df["pause_end"] = pd.to_datetime(res_df["pause_end"]).dt.strftime("%Y-%m-%d %H:%M").fillna("-")

    cols = [
        "dt_date",
        "date",
        "weekday",
        "group",
        "vehicle_country",
        "vehicle_name",
        "driver_name",
        "rest_country_before_shift",
        "rest_before_shift_hours",
        "display_text",
        "status_color",
        "pause_start",
        "pause_end",
        "debt_balance",
    ]
    return res_df[cols]


@st.cache_data
def convert_df_to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Мониторинг смен")
        worksheet = writer.sheets["Мониторинг смен"]
        for col in worksheet.columns:
            max_len = max(len(str(cell.value or "")) for cell in col)
            col_letter = col[0].column_letter
            worksheet.column_dimensions[col_letter].width = max(max_len + 3, 12)
    return output.getvalue()


def reset_select_key(key, values):
    st.session_state[key] = values


def reset_range_key(min_k, max_k, max_val):
    st.session_state[min_k] = 0.0
    st.session_state[max_k] = float(max_val)


def reset_all_filters():
    for k in list(st.session_state.keys()):
        if k != "process_file_fast":
            del st.session_state[k]


# ==============================================================================
# ИНТЕРФЕЙС
# ==============================================================================
st.sidebar.markdown(
    "<div style='font-size:13px; font-weight:700; margin-bottom:6px;"
    " color:#0F172A;'>⚙️ Фильтры поиска</div>",
    unsafe_allow_html=True,
)

with st.sidebar.expander("📁 Файл данных", expanded=True):
    uploaded_file = st.file_uploader(
        "Загрузите CSV/XLSX", type=["xlsx", "csv"], label_visibility="collapsed"
    )

if uploaded_file:
    file_bytes = uploaded_file.getvalue()
    daily_df = process_file_fast(file_bytes, uploaded_file.name)

    min_date = daily_df["dt_date"].min()
    max_date = daily_df["dt_date"].max()

    latest_debts_raw = daily_df.sort_values(["vehicle_name", "date"]).groupby("vehicle_name").last()["debt_balance"].astype(float)
    vehicles_with_debt = latest_debts_raw[latest_debts_raw > 0].index.tolist()

    st.sidebar.markdown(
        "<div class='filter-card'><div class='filter-card-title'>"
        "Период дат</div>",
        unsafe_allow_html=True,
    )
    date_range = st.sidebar.date_input(
        "Период",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
        key="date_range",
        label_visibility="collapsed",
    )
    st.sidebar.markdown("</div>", unsafe_allow_html=True)

    st.sidebar.markdown("<div class='filter-card'>", unsafe_allow_html=True)
    only_debt_vehicles = st.sidebar.checkbox("⚠️ Только машины с долгом", value=False, key="only_debt_checkbox")
    st.sidebar.markdown("</div>", unsafe_allow_html=True)

    def render_select_filter(label, options, key_prefix):
        sel_key = f"{key_prefix}_select"
        all_opts = sorted(list(set(options)))

        st.sidebar.markdown(
            f"<div class='filter-card'><div"
            f" class='filter-card-title'>{label}</div>",
            unsafe_allow_html=True,
        )

        b1, b2 = st.sidebar.columns(2)
        b1.button(
            "Все",
            key=f"{key_prefix}_all",
            on_click=reset_select_key,
            args=(sel_key, all_opts),
        )
        b2.button(
            "✖ Сброс",
            key=f"{key_prefix}_clr",
            on_click=reset_select_key,
            args=(sel_key, []),
        )

        selected = st.sidebar.multiselect(
            label,
            options=all_opts,
            key=sel_key,
            label_visibility="collapsed",
            placeholder="Выберите...",
        )

        st.sidebar.markdown("</div>", unsafe_allow_html=True)
        return selected

    selected_vehicle_countries = render_select_filter(
        "Страна авто", ["CZ", "SK", "Other"], "v_countries"
    )
    selected_groups = render_select_filter(
        "Группы", daily_df["group"].unique(), "groups"
    )
    
    available_vehicles = vehicles_with_debt if only_debt_vehicles else daily_df["vehicle_name"].unique()
    selected_vehicles = render_select_filter(
        "Машины", available_vehicles, "vehicles"
    )
    
    selected_drivers = render_select_filter(
        "Водители", daily_df["driver_name"].unique(), "drivers"
    )
    selected_countries = render_select_filter(
        "Страна отдыха ДО смены",
        daily_df["rest_country_before_shift"].unique(),
        "countries",
    )

    def render_range_filter(label, max_val, key_prefix, step=0.5, unit="ч"):
        min_k, max_k = f"{key_prefix}_min", f"{key_prefix}_max"

        if min_k not in st.session_state:
            st.session_state[min_k] = 0.0
        if max_k not in st.session_state:
            st.session_state[max_k] = float(max_val)

        st.sidebar.markdown(
            f"<div class='filter-card'><div class='filter-card-title'>{label}"
            f" ({unit})</div>",
            unsafe_allow_html=True,
        )

        c1, c2, c3 = st.sidebar.columns([2, 2, 1])
        v_min = c1.number_input(
            "От",
            min_value=0.0,
            max_value=float(max_val),
            step=step,
            key=min_k,
            label_visibility="collapsed",
        )
        v_max = c2.number_input(
            "До",
            min_value=0.0,
            max_value=float(max_val),
            step=step,
            key=max_k,
            label_visibility="collapsed",
        )
        c3.button(
            "✖",
            key=f"{key_prefix}_rst",
            on_click=reset_range_key,
            args=(min_k, max_k, max_val),
        )

        st.sidebar.markdown("</div>", unsafe_allow_html=True)
        return v_min, v_max

    max_rest = max(
        float(daily_df["rest_before_shift_hours"].max() or 24.0) + 1.0, 10.0
    )
    rest_min, rest_max = render_range_filter("Отдых ДО смены", max_rest, "rest")

    st.sidebar.button(
        "🔄 Сбросить ВСЕ фильтры",
        use_container_width=True,
        on_click=reset_all_filters,
    )

    nav_page = st.radio(
        "Навигация",
        options=["Main", "Calendar"],
        horizontal=True,
        label_visibility="collapsed"
    )

    mask = pd.Series(True, index=daily_df.index)

    if only_debt_vehicles:
        mask &= daily_df["vehicle_name"].isin(vehicles_with_debt)

    if isinstance(date_range, tuple) and len(date_range) == 2:
        mask &= daily_df["dt_date"].between(date_range[0], date_range[1])

    if selected_vehicle_countries:
        mask &= daily_df["vehicle_country"].isin(selected_vehicle_countries)
    if selected_groups:
        mask &= daily_df["group"].isin(selected_groups)
    if selected_vehicles:
        mask &= daily_df["vehicle_name"].isin(selected_vehicles)
    if selected_drivers:
        mask &= daily_df["driver_name"].isin(selected_drivers)
    if selected_countries:
        mask &= daily_df["rest_country_before_shift"].isin(selected_countries)

    mask &= (
        daily_df["rest_before_shift_hours"].between(rest_min, rest_max)
        | daily_df["rest_before_shift_hours"].isna()
    )

    filtered = daily_df[mask].drop(columns=["dt_date"]).reset_index(drop=True)

    if nav_page == "Main":
        filtered_display = filtered.copy()

        display_df = filtered_display.drop(columns=["status_color"]).rename(
            columns={
                "date": "Дата",
                "weekday": "День недели",
                "group": "Группа",
                "vehicle_country": "Страна авто",
                "vehicle_name": "Машина",
                "driver_name": "Водитель",
                "rest_country_before_shift": "Страна отдыха ДО смены",
                "rest_before_shift_hours": "_raw_hours",
                "display_text": "Отдых ДО смены (ч)",
                "pause_start": "Начало паузы",
                "pause_end": "Конец паузы",
                "debt_balance": "Компенсация",
            }
        )

        if not display_df.empty:
            sub_dfs = []
            for vehicle, group_df in display_df.groupby("Машина", sort=True):
                sorted_sub = group_df.sort_values(by="Дата", ascending=True)
                sub_dfs.append(sorted_sub)
            display_df = pd.concat(sub_dfs, ignore_index=True)

        def apply_table_styling(df, raw_df):
            if df.empty:
                return df.style

            def style_specific_row_and_cell(row):
                styles = ['' for _ in row]
                idx = row.name
                
                c_type = raw_df.loc[idx, "status_color"] if idx in raw_df.index else ""
                h_val = raw_df.loc[idx, "rest_before_shift_hours"] if idx in raw_df.index else np.nan

                cell_style = get_cell_style(c_type)
                if not cell_style and pd.notna(h_val) and h_val >= 45.0:
                    cell_style = get_cell_style("blue")

                if 'Отдых ДО смены (ч)' in df.columns:
                    col_idx = df.columns.get_loc('Отдых ДО смены (ч)')
                    styles[col_idx] = cell_style

                row_blue_style = "background-color: #DBEAFE; color: #1E3A8A;" if (pd.notna(h_val) and h_val >= 24.0) else ""
                if row_blue_style:
                    for i, col_name in enumerate(df.columns):
                        if col_name != 'Отдых ДО смены (ч)':
                            styles[i] = row_blue_style

                return styles

            styler = df.style.apply(style_specific_row_and_cell, axis=1).format({"Компенсация": "{:.2f}"})
            
            def highlight_borders(df_sub):
                css_styles = pd.DataFrame('', index=df_sub.index, columns=df_sub.columns)
                cars = df_sub['Машина'].values
                for i in range(1, len(cars)):
                    if cars[i] != cars[i-1]:
                        css_styles.iloc[i, :] = 'border-top: 3px solid #0F172A !important;'
                return css_styles

            styler.apply(highlight_borders, axis=None)
            return styler

        render_df = display_df.drop(columns=["_raw_hours"])

        col_h, col_b = st.columns([4, 1])
        with col_h:
            st.markdown(
                "<div class='main-header' style='margin-top: 15px;'>Мониторинг смен (найдено:"
                f" {len(render_df)})</div>",
                unsafe_allow_html=True,
            )
        with col_b:
            if not render_df.empty:
                excel_file = convert_df_to_excel(render_df)
                st.download_button(
                    label="📥 Скачать Excel",
                    data=excel_file,
                    file_name="monitoring_smen.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )

        if not render_df.empty:
            styled_display = apply_table_styling(render_df.head(500), filtered)
            st.dataframe(
                styled_display,
                use_container_width=True,
                hide_index=True,
                height=750,
            )
            if len(render_df) > 500:
                st.caption(" Отображены первые 500 строк. Скачайте Excel для получения полного файла.")
        else:
            st.info("Данные не найдены.")

    elif nav_page == "Calendar":
        st.markdown("<div class='main-header' style='margin-top: 15px;'>Календарная матрица отдыха машин</div>", unsafe_allow_html=True)

        if not filtered.empty:
            pivot_df = filtered.copy()

            def safe_max_hours(x):
                valid = x.dropna()
                return valid.max() if not valid.empty else np.nan

            def safe_display_text(x):
                sub = pivot_df.loc[x.index, "display_text"].dropna()
                return " / ".join(sub) if not sub.empty else ""

            def safe_status_color(x):
                sub_hours = pivot_df.loc[x.index, "rest_before_shift_hours"]
                valid_hours = sub_hours.dropna()
                if valid_hours.empty:
                    return ""
                max_idx = valid_hours.idxmax()
                return pivot_df.loc[max_idx, "status_color"] if max_idx in pivot_df.index else ""

            grouped_matrix = (
                pivot_df.groupby(["vehicle_name", "date"])
                .agg({
                    "rest_before_shift_hours": [safe_max_hours, safe_display_text],
                    "status_color": safe_status_color
                })
            )
            grouped_matrix.columns = ["max_hours", "display_text", "status_color"]
            grouped_matrix = grouped_matrix.reset_index()

            # Строим таблицу на чистых числах max_hours, чтобы сортировка по клику работала идеально
            calendar_table = grouped_matrix.pivot(
                index="vehicle_name", columns="date", values="max_hours"
            )

            # Сохраняем текстовое представление для отрисовки красивых форматов с плюсами (если нужно)
            text_matrix = grouped_matrix.pivot(
                index="vehicle_name", columns="date", values="display_text"
            ).fillna("")

            # Добавляем столбец «Компенсация» как число (float)
            latest_debts_float = daily_df.sort_values(["vehicle_name", "date"]).groupby("vehicle_name").last()["debt_balance"].astype(float)
            calendar_table["Компенсация"] = calendar_table.index.map(latest_debts_float).fillna(0.0)

            new_column_names = {}
            for col in calendar_table.columns:
                if col == "Компенсация":
                    new_column_names[col] = "Компенсация"
                    continue
                try:
                    dt = pd.to_datetime(col)
                    wd = dt.dayofweek
                    wd_map = {0: "Пн", 1: "Вт", 2: "Ср", 3: "Чт", 4: "Пт", 5: "Сб", 6: "Вс"}
                    suffix = f" [{wd_map[wd]}]"
                    if wd in [5, 6]:
                        suffix = f" 🔥{wd_map[wd]}"
                    new_column_names[col] = f"{col}{suffix}"
                except:
                    pass
            
            calendar_table = calendar_table.rename(columns=new_column_names)
            text_matrix = text_matrix.rename(columns=new_column_names)

            color_matrix = grouped_matrix.pivot(
                index="vehicle_name", columns="date", values="status_color"
            ).fillna("").rename(columns=new_column_names)
            
            hours_matrix = grouped_matrix.pivot(
                index="vehicle_name", columns="date", values="max_hours"
            ).rename(columns=new_column_names)

            def style_calendar_cell(data):
                df_styles = pd.DataFrame('', index=calendar_table.index, columns=calendar_table.columns)
                
                for orig_col, new_col in new_column_names.items():
                    if new_col == "Компенсация":
                        for idx in calendar_table.index:
                            try:
                                val = float(latest_debts_float.get(idx, 0.0))
                            except:
                                val = 0.0
                            if val > 0:
                                df_styles.loc[idx, new_col] = "font-weight: bold; color: #9A3412; background-color: #FFEDD5;"
                        continue

                    try:
                        raw_date_str = new_col.split(" ")[0]
                        is_weekend = pd.to_datetime(raw_date_str).dayofweek in [5, 6]
                    except:
                        is_weekend = False

                    for idx in calendar_table.index:
                        c_type = color_matrix.loc[idx, new_col] if new_col in color_matrix.columns and idx in calendar_table.index else ""
                        h_val = hours_matrix.loc[idx, new_col] if new_col in hours_matrix.columns and idx in calendar_table.index else np.nan
                        
                        bg_style = get_cell_style(c_type)
                        if not bg_style and pd.notna(h_val) and h_val >= 45.0:
                            bg_style = get_cell_style("blue")

                        if is_weekend:
                            if not bg_style:
                                bg_style = "background-color: #F8FAFC;"
                            else:
                                bg_style += " border-right: 2px dashed #94A3B8; border-left: 2px dashed #94A3B8;"

                        if bg_style:
                            df_styles.loc[idx, new_col] = bg_style

                return df_styles

            # Кастомный форматтер: если в ячейке число, выводим красивый текст из text_matrix (с плюсами), а сортируем по числу!
            def custom_formatter(val):
                if pd.isna(val):
                    return ""
                return f"{val:.2f}"

            formatted_dict = {col: custom_formatter for col in new_column_names.values() if col != "Компенсация"}
            formatted_dict["Компенсация"] = "{:.2f}"

            # Подменяем вывод ячеек на текстовые значения с сохранением числовой подложки для сортировки
            styled_calendar = calendar_table.style.apply(style_calendar_cell, axis=None)
            
            # Переопределяем отображение через кастомную функцию для каждой ячейки по маске text_matrix
            def format_with_text(x):
                res = pd.DataFrame("", index=x.index, columns=x.columns)
                for c in x.columns:
                    for r in x.index:
                        if c == "Компенсация":
                            val = x.loc[r, c]
                            res.loc[r, c] = f"{val:.2f}" if pd.notna(val) else "0.00"
                        else:
                            txt = text_matrix.loc[r, c] if (r in text_matrix.index and c in text_matrix.columns) else ""
                            res.loc[r, c] = txt
                return res

            styled_calendar.format(format_with_text)

            st.dataframe(styled_calendar, use_container_width=True, height=750)
        else:
            st.info("Нет данных для отображения матрицы.")

else:
    st.markdown(
        "<div class='main-header'>Мониторинг смен</div>", unsafe_allow_html=True
    )
    st.info("Загрузите CSV или Excel файл в панели слева.")
