import io
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(layout="wide", page_title="Мониторинг смен")

# ==============================================================================
# CSS-СТИЛИ: СИНИЕ ТЕГИ, КОМПАКТНЫЙ СУПЕР-UI, ЖИРНЫЕ ГРАНИЦЫ И ПОДСВЕТКА ВЫХОДНЫХ
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

    /* Уменьшенные карточки фильтров (единый стиль для всех) */
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

    /* Синие компактные блоки выбранных значений в multiselect */
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

    /* Единый компактный размер инпутов, селектов и мультиселектов */
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
    
    /* Кнопки "Все" / "Сброс" */
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
# БЫСТРАЯ ВЕКТОРНАЯ ОБРАБОТКА ДАННЫХ
# ==============================================================================
@st.cache_data(show_spinner="Обработка файла...")
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

  # Фиксируем межсменный отдых (RESTING или CARDLESS от 9 часов / 540 мин)
  is_rest = act_type.isin(["RESTING", "CARDLESS"]) & (dur_min >= 540)

  change_group = (
      (df["vehicle_name"] != df["vehicle_name"].shift())
      | (df["driver_name"] != df["driver_name"].shift())
      | (df["group"] != df["group"].shift())
  )
  df["shift_id"] = (is_rest | change_group).cumsum()

  # Запоминаем часы отдыха, страну отдыха, а также начало и конец паузы (до смены)
  df["rest_hours"] = np.where(is_rest, (dur_min / 60).round(2), np.nan)
  df["rest_country"] = np.where(is_rest, df["country"], np.nan)
  df["pause_start_dt"] = np.where(is_rest, df["start_datetime"], pd.NaT)
  df["pause_end_dt"] = np.where(is_rest, df["end_datetime"], pd.NaT)

  rest_before_hours = df[is_rest].groupby("shift_id")["rest_hours"].first()
  rest_before_country = df[is_rest].groupby("shift_id")["rest_country"].first()
  pause_start_before = df[is_rest].groupby("shift_id")["pause_start_dt"].first()
  pause_end_before = df[is_rest].groupby("shift_id")["pause_end_dt"].first()

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

  # День недели выбранной даты (на русском)
  days_ru = {
      0: "Понедельник", 1: "Вторник", 2: "Среда", 
      3: "Четверг", 4: "Пятница", 5: "Суббота", 6: "Воскресенье"
  }
  agg_df["weekday"] = agg_df["shift_start"].dt.dayofweek.map(days_ru)

  # Извлекаем 2-ю и 3-ю буквы (индексы 1:3 в Python)
  extracted_code = agg_df["vehicle_name"].astype(str).str[1:3].str.upper()
  agg_df["vehicle_country"] = np.where(
      extracted_code.isin(["CZ", "SK"]), extracted_code, "Other"
  )

  # Привязываем параметры отдыха перед сменой
  agg_df["rest_before_shift_hours"] = agg_df["shift_id"].map(rest_before_hours)
  agg_df["rest_country_before_shift"] = (
      agg_df["shift_id"].map(rest_before_country).fillna("N/A")
  )
  agg_df["pause_start"] = agg_df["shift_id"].map(pause_start_before)
  agg_df["pause_end"] = agg_df["shift_id"].map(pause_end_before)

  agg_df["pause_start"] = pd.to_datetime(agg_df["pause_start"]).dt.strftime("%Y-%m-%d %H:%M").fillna("-")
  agg_df["pause_end"] = pd.to_datetime(agg_df["pause_end"]).dt.strftime("%Y-%m-%d %H:%M").fillna("-")

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
      "pause_start",
      "pause_end",
  ]
  return agg_df[cols]


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


# ==============================================================================
# CALLBACKS ДЛЯ СБРОСА
# ==============================================================================
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

  # 1. Фильтр Даты (Календарь Диапазон)
  min_date = daily_df["dt_date"].min()
  max_date = daily_df["dt_date"].max()

  st.sidebar.markdown(
      "<div class='filter-card'><div class='filter-card-title'>Период"
      " дат</div>",
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

  # Рендер селекторов (включая фильтр по группам)
  selected_vehicle_countries = render_select_filter(
      "Страна авто", ["CZ", "SK", "Other"], "v_countries"
  )
  selected_groups = render_select_filter(
      "Группы", daily_df["group"].unique(), "groups"
  )
  selected_vehicles = render_select_filter(
      "Машины", daily_df["vehicle_name"].unique(), "vehicles"
  )
  selected_drivers = render_select_filter(
      "Водители", daily_df["driver_name"].unique(), "drivers"
  )
  selected_countries = render_select_filter(
      "Страна отдыха ДО смены",
      daily_df["rest_country_before_shift"].unique(),
      "countries",
  )

  # Диапазонный фильтр для отдыха
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

  # --------------------------------------------------------------------------
  # ВЕКТОРНАЯ ФИЛЬТРАЦИЯ
  # --------------------------------------------------------------------------
  mask = pd.Series(True, index=daily_df.index)

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

  filtered = daily_df[mask].drop(columns=["dt_date"])

  display_df = filtered.rename(
      columns={
          "date": "Дата",
          "weekday": "День недели",
          "group": "Группа",
          "vehicle_country": "Страна авто",
          "vehicle_name": "Машина",
          "driver_name": "Водитель",
          "rest_country_before_shift": "Страна отдыха ДО смены",
          "rest_before_shift_hours": "Отдых ДО смены (ч)",
          "pause_start": "Начало паузы",
          "pause_end": "Конец паузы",
      }
  )

  # --------------------------------------------------------------------------
  # УРОВНИ СОРТИРОВКИ НАД ТАБЛИЦЕЙ
  # --------------------------------------------------------------------------
  st.markdown("<div class='main-header'>Параметры сортировки таблицы</div>", unsafe_allow_html=True)
  
  sort_cols_options = [col for col in display_df.columns if col != "Машина"]
  
  col_s1, col_s2 = st.columns(2)
  with col_s1:
    sort_1 = st.selectbox("Сортировка 1-й очереди", options=sort_cols_options, index=0)
  with col_s2:
    sort_2 = st.selectbox("Сортировка 2-й очереди", options=["Нет"] + sort_cols_options, index=0)

  # Формируем список сортировки: "Машина" ВСЕГДА первая для склейки машин
  sorting_columns = ["Машина", sort_1]
  ascending_flags = [True, True]

  if sort_2 != "Нет" and sort_2 != sort_1:
    sorting_columns.append(sort_2)
    ascending_flags.append(True)

  if not display_df.empty:
    display_df = display_df.sort_values(by=sorting_columns, ascending=ascending_flags)

  # --------------------------------------------------------------------------
  # СТИЛИЗАЦИЯ ТАБЛИЦЫ (Подсветка выходных и границы между машинами)
  # --------------------------------------------------------------------------
  def apply_table_styling(df):
    if df.empty:
      return df.style

    def style_rows(row):
      styles = ['' for _ in row]
      
      # 1. Подсветка выходных дней (Суббота / Воскресенье) светло-синим цветом
      if row.get('День недели') in ['Суббота', 'Воскресенье']:
        styles = ['background-color: #E0F2FE' for _ in row]
        
      return styles

    styler = df.style.apply(style_rows, axis=1)
    
    # 2. Добавление жирной границы между разными машинами для визуального разделения
    def highlight_borders(df_sub):
      css_styles = pd.DataFrame('', index=df_sub.index, columns=df_sub.columns)
      cars = df_sub['Машина'].values
      for i in range(1, len(cars)):
        if cars[i] != cars[i-1]:
          css_styles.iloc[i, :] = 'border-top: 3px solid #0F172A !important;'
      return css_styles

    styler.apply(highlight_borders, axis=None)
    return styler

  # --------------------------------------------------------------------------
  # ВЫВОД ЗАГОЛОВКА И ТАБЛИЦЫ
  # --------------------------------------------------------------------------
  col_h, col_b = st.columns([4, 1])
  with col_h:
    st.markdown(
        "<div class='main-header' style='margin-top: 15px;'>Мониторинг смен (найдено:"
        f" {len(display_df)})</div>",
        unsafe_allow_html=True,
    )
  with col_b:
    if not display_df.empty:
      excel_file = convert_df_to_excel(display_df)
      st.download_button(
          label="📥 Скачать Excel",
          data=excel_file,
          file_name="monitoring_smen.xlsx",
          mime=(
              "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
          ),
          use_container_width=True,
      )

  if not display_df.empty:
    # Применяем стилизатор для визуального разделения машин и подсветки выходных
    styled_display = apply_table_styling(display_df.head(500))
    
    st.dataframe(
        styled_display,
        use_container_width=True,
        hide_index=True,
        height=750,
    )
    if len(display_df) > 500:
      st.caption(
          " Отображены первые 500 строк. Скачайте Excel для получения полного"
          " файла."
      )
  else:
    st.info("Данные не найдены.")

else:
  st.markdown(
      "<div class='main-header'>Мониторинг смен</div>", unsafe_allow_html=True
  )
  st.info("Загрузите CSV или Excel файл в панели слева.")
