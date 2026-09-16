import io
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(layout="wide", page_title="Мониторинг смен")

# ==============================================================================
# CSS-СТИЛИ: СИНИЕ ТЕГИ, КОМПАКТНЫЙ СУПЕР-UI И НАВИГАЦИЯ
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
# ВСПОМОГАТЕЛЬНАЯ ФУНКЦИЯ ЦВЕТОКОДИНГА ЧАСОВ ОТДЫХА
# ==============================================================================
def get_rest_color(hours, is_fourth_restricted=False, is_second_short_weekend=False):
  if pd.isna(hours):
    return ""
  try:
    h = float(hours)
  except:
    return ""

  # Правило: от 9.0 ровно (включительно) до 11.0 (меньше)
  if 9.0 <= h < 11.0:
    if is_fourth_restricted:
      return "background-color: #FEE2E2;"  # Светло-красный (4-я подряд пауза)
    return "background-color: #FEF9C3;"  # Светло-желтый
  elif 24.0 < h <= 45.0:
    if is_second_short_weekend:
      return "background-color: #FEE2E2;"  # Светло-красный (вторая короткая недельная пауза подряд)
    return "background-color: #FFEDD5;"  # Светло-оранжевый
  elif h > 45.0:
    return "background-color: #DCFCE7;"  # Светло-зеленый
  return ""


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

  rest_before_hours = df[is_rest].groupby("shift_id")["rest_hours"].first()
  rest_before_country = df[is_rest].groupby("shift_id")["rest_country"].first()
  
  rest_filtered = df[is_rest]
  pause_start_before = rest_filtered.groupby("shift_id")["pause_start_dt"].first()
  pause_end_before = rest_filtered.groupby("shift_id")["pause_end_dt"].first()

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

  agg_df["pause_start"] = pd.to_datetime(agg_df["pause_start"]).dt.strftime("%Y-%m-%d %H:%M").fillna("-")
  agg_df["pause_end"] = pd.to_datetime(agg_df["pause_end"]).dt.strftime("%Y-%m-%d %H:%M").fillna("-")

  # Подсчет для 4-й паузы (9.0 <= h < 11.0)
  is_fourth_flags = []
  # Подсчет для второй короткой недельной паузы (24.0 < h <= 45.0)
  is_second_short_weekend_flags = []

  for _, v_group in agg_df.sort_values(["vehicle_name", "shift_start"]).groupby("vehicle_name"):
    consecutive_9_11_count = 0
    last_was_short_weekend = False

    for idx, row in v_group.iterrows():
      h = row["rest_before_shift_hours"]
      if pd.isna(h):
        is_fourth_flags.append((idx, False))
        is_second_short_weekend_flags.append((idx, False))
        continue
      
      # Проверка 9.0 <= h < 11.0
      if h > 24.0:
        consecutive_9_11_count = 0
        is_fourth_flags.append((idx, False))
      elif 9.0 <= h < 11.0:
        if consecutive_9_11_count == 3:
          is_fourth_flags.append((idx, True))
          consecutive_9_11_count = 0 
        else:
          consecutive_9_11_count += 1
          is_fourth_flags.append((idx, False))
      else:
        consecutive_9_11_count = 0
        is_fourth_flags.append((idx, False))

      # Проверка второй короткой недельной паузы (24.0 < h <= 45.0)
      if 24.0 < h <= 45.0:
        if last_was_short_weekend:
          is_second_short_weekend_flags.append((idx, True))
          last_was_short_weekend = False # сброс после срабатывания или считаем цепью? Пусть срабатывает на каждую вторую подряд
        else:
          is_second_short_weekend_flags.append((idx, False))
          last_was_short_weekend = True
      elif h > 45.0:
        last_was_short_weekend = False
        is_second_short_weekend_flags.append((idx, False))
      else:
        is_second_short_weekend_flags.append((idx, False))

  fourth_dict = dict(is_fourth_flags)
  agg_df["is_fourth_9_11"] = agg_df.index.map(fourth_dict).fillna(False)

  second_short_dict = dict(is_second_short_weekend_flags)
  agg_df["is_second_short_weekend"] = agg_df.index.map(second_short_dict).fillna(False)

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
      "is_fourth_9_11",
      "is_second_short_weekend",
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

  # ==============================================================================
  # НАВИГАЦИЯ СВЕРХУ (Main / Calendar / Weekend Rest Calendar)
  # ==============================================================================
  nav_page = st.radio(
      "Навигация",
      options=["Main", "Calendar", "Weekend Rest Calendar"],
      horizontal=True,
      label_visibility="collapsed"
  )

  # ==============================================================================
  # ВЕКТОРНАЯ ФИЛЬТРАЦИЯ
  # ==============================================================================
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

  # ==============================================================================
  # СТРАНИЦА: MAIN
  # ==============================================================================
  if nav_page == "Main":
    raw_rest_dict = filtered["rest_before_shift_hours"].to_dict()
    fourth_flag_dict = filtered["is_fourth_9_11"].to_dict()
    second_short_flag_dict = filtered["is_second_short_weekend"].to_dict()

    filtered_display = filtered.copy()
    filtered_display["rest_before_shift_hours"] = filtered_display["rest_before_shift_hours"].apply(
        lambda x: f"{x:.2f}" if pd.notna(x) else "N/A"
    )

    display_df = filtered_display.drop(columns=["is_fourth_9_11", "is_second_short_weekend"]).rename(
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

    if not display_df.empty:
      sub_dfs = []
      for vehicle, group_df in display_df.groupby("Машина", sort=True):
        sorted_sub = group_df.sort_values(by="Дата", ascending=True)
        sub_dfs.append(sorted_sub)
      display_df = pd.concat(sub_dfs, ignore_index=True)

    def apply_table_styling(df):
      if df.empty:
        return df.style

      def style_specific_row_and_cell(row):
        styles = ['' for _ in row]
        idx = row.name
        raw_val = raw_rest_dict.get(idx)
        is_fourth = fourth_flag_dict.get(idx, False)
        is_second_short = second_short_flag_dict.get(idx, False)

        is_rest_ge_24 = pd.notna(raw_val) and raw_val >= 24.0
        if is_rest_ge_24:
          styles = ['background-color: #E0F2FE' for _ in row]

        if 'Отдых ДО смены (ч)' in df.columns:
          col_idx = df.columns.get_loc('Отдых ДО смены (ч)')
          color_style = get_rest_color(raw_val, is_fourth_restricted=is_fourth, is_second_short_weekend=is_second_short)
          if color_style:
            styles[col_idx] = color_style
          elif not is_rest_ge_24:
            styles[col_idx] = ''

        return styles

      styler = df.style.apply(style_specific_row_and_cell, axis=1)
      
      def highlight_borders(df_sub):
        css_styles = pd.DataFrame('', index=df_sub.index, columns=df_sub.columns)
        cars = df_sub['Машина'].values
        for i in range(1, len(cars)):
          if cars[i] != cars[i-1]:
            css_styles.iloc[i, :] = 'border-top: 3px solid #0F172A !important;'
        return css_styles

      styler.apply(highlight_borders, axis=None)
      return styler

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

  # ==============================================================================
  # СТРАНИЦА: CALENDAR (Матрица машин по датам)
  # ==============================================================================
  elif nav_page == "Calendar":
    st.markdown("<div class='main-header' style='margin-top: 15px;'>Календарная матрица отдыха машин</div>", unsafe_allow_html=True)

    if not filtered.empty:
      pivot_df = filtered.copy()
      pivot_df["formatted_hours"] = pivot_df["rest_before_shift_hours"].apply(
          lambda x: f"{x:.2f}" if pd.notna(x) else ""
      )

      grouped_matrix = (
          pivot_df.groupby(["vehicle_name", "date"])
          .agg({
              "formatted_hours": lambda x: " / ".join([str(v) for v in x if v != ""]),
              "is_fourth_9_11": "any",
              "is_second_short_weekend": "any",
              "rest_before_shift_hours": "max"
          })
          .reset_index()
      )

      calendar_table = grouped_matrix.pivot(
          index="vehicle_name", columns="date", values="formatted_hours"
      ).fillna("")

      new_column_names = {}
      for col in calendar_table.columns:
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

      fourth_matrix = grouped_matrix.pivot(
          index="vehicle_name", columns="date", values="is_fourth_9_11"
      ).fillna(False)

      second_short_matrix = grouped_matrix.pivot(
          index="vehicle_name", columns="date", values="is_second_short_weekend"
      ).fillna(False)

      hours_matrix = grouped_matrix.pivot(
          index="vehicle_name", columns="date", values="rest_before_shift_hours"
      )

      def style_calendar_cell(data):
        df_styles = pd.DataFrame('', index=calendar_table.index, columns=calendar_table.columns)
        
        for orig_col, new_col in new_column_names.items():
          try:
            is_weekend = pd.to_datetime(orig_col).dayofweek in [5, 6]
          except:
            is_weekend = False

          for idx in calendar_table.index:
            val = calendar_table.loc[idx, new_col]
            is_fourth = fourth_matrix.loc[idx, orig_col] if orig_col in fourth_matrix.columns and idx in fourth_matrix.index else False
            is_sec_short = second_short_matrix.loc[idx, orig_col] if orig_col in second_short_matrix.columns and idx in second_short_matrix.index else False
            h_max = hours_matrix.loc[idx, orig_col] if orig_col in hours_matrix.columns and idx in hours_matrix.index else np.nan

            bg_style = ""
            if val and not pd.isna(val):
              bg_style = get_rest_color(h_max, is_fourth_restricted=is_fourth, is_second_short_weekend=is_sec_short)
            
            if is_weekend:
              if not bg_style:
                bg_style = "background-color: #F8FAFC;"
              else:
                bg_style += " border-right: 2px dashed #94A3B8; border-left: 2px dashed #94A3B8;"

            if bg_style:
              df_styles.loc[idx, new_col] = bg_style

        return df_styles

      styled_calendar = calendar_table.style.apply(style_calendar_cell, axis=None)
      st.dataframe(styled_calendar, use_container_width=True, height=750)
    else:
      st.info("Нет данных для отображения матрицы.")

  # ==============================================================================
  # СТРАНИЦА: WEEKEND REST CALENDAR (Только паузы > 24 часов)
  # ==============================================================================
  elif nav_page == "Weekend Rest Calendar":
    st.markdown("<div class='main-header' style='margin-top: 15px;'>Weekend Rest Calendar (Паузы > 24 часов)</div>", unsafe_allow_html=True)

    weekend_filtered = filtered[filtered["rest_before_shift_hours"] > 24.0].copy()

    if not weekend_filtered.empty:
      raw_rest_dict_w = weekend_filtered["rest_before_shift_hours"].to_dict()
      fourth_flag_dict_w = weekend_filtered["is_fourth_9_11"].to_dict()
      second_short_flag_dict_w = weekend_filtered["is_second_short_weekend"].to_dict()

      weekend_display = weekend_filtered.copy()
      weekend_display["rest_before_shift_hours"] = weekend_display["rest_before_shift_hours"].apply(
          lambda x: f"{x:.2f}" if pd.notna(x) else "N/A"
      )

      display_w_df = weekend_display.drop(columns=["is_fourth_9_11", "is_second_short_weekend"]).rename(
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

      sub_dfs_w = []
      for vehicle, group_df in display_w_df.groupby("Машина", sort=True):
        sorted_sub = group_df.sort_values(by="Дата", ascending=True)
        sub_dfs_w.append(sorted_sub)
      display_w_df = pd.concat(sub_dfs_w, ignore_index=True)

      def apply_weekend_table_styling(df):
        if df.empty:
          return df.style

        def style_row_cell(row):
          styles = ['' for _ in row]
          idx = row.name
          raw_val = raw_rest_dict_w.get(idx)
          is_fourth = fourth_flag_dict_w.get(idx, False)
          is_sec_short = second_short_flag_dict_w.get(idx, False)

          styles = ['background-color: #E0F2FE' for _ in row]

          if 'Отдых ДО смены (ч)' in df.columns:
            col_idx = df.columns.get_loc('Отдых ДО смены (ч)')
            color_style = get_rest_color(raw_val, is_fourth_restricted=is_fourth, is_second_short_weekend=is_sec_short)
            if color_style:
              styles[col_idx] = color_style

          return styles

        styler = df.style.apply(style_row_cell, axis=1)
        
        def highlight_borders(df_sub):
          css_styles = pd.DataFrame('', index=df_sub.index, columns=df_sub.columns)
          cars = df_sub['Машина'].values
          for i in range(1, len(cars)):
            if cars[i] != cars[i-1]:
              css_styles.iloc[i, :] = 'border-top: 3px solid #0F172A !important;'
          return css_styles

        styler.apply(highlight_borders, axis=None)
        return styler

      col_hw, col_bw = st.columns([4, 1])
      with col_hw:
        st.markdown(f"<div style='font-size:13px; font-weight:600; color:#475569;'>Найдено смен с отдыхом > 24ч: {len(display_w_df)}</div>", unsafe_allow_html=True)
      with col_bw:
        excel_file_w = convert_df_to_excel(display_w_df)
        st.download_button(
            label="📥 Скачать Excel",
            data=excel_file_w,
            file_name="weekend_rest_calendar.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

      styled_weekend = apply_weekend_table_styling(display_w_df)
      st.dataframe(
          styled_weekend,
          use_container_width=True,
          hide_index=True,
          height=750,
      )
    else:
      st.info("Нет данных с отдыхом более 24 часов по заданным фильтрам.")

else:
  st.markdown(
      "<div class='main-header'>Мониторинг смен</div>", unsafe_allow_html=True
  )
  st.info("Загрузите CSV или Excel файл в панели слева.")
