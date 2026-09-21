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
    elif color_type == "critical":
        return "background-color: #7F1D1D; font-weight: bold; color: #FFFFFF;"
    return ""


@st.cache_data(show_spinner="Обработка файла и проверка РТО...")
def process_file_fast(file_bytes, file_name):
    if file_name.lower().endswith(".csv"):
        df = pd.read_csv(io.BytesIO(file_bytes))
    else:
        df = pd.read_excel(io.BytesIO(file_bytes), engine="openpyxl")

    required = {
        "start_date", "start_time", "end_date", "end_time",
        "vehicle_name", "driver_name", "activity_type", "duration_minutes",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError("В файле отсутствуют обязательные колонки: " + ", ".join(missing))

    df["start_datetime"] = pd.to_datetime(
        df["start_date"].astype(str) + " " + df["start_time"].astype(str),
        errors="coerce",
    )
    df["end_datetime"] = pd.to_datetime(
        df["end_date"].astype(str) + " " + df["end_time"].astype(str),
        errors="coerce",
    )
    df = df.dropna(subset=["start_datetime", "end_datetime"]).copy()
    df = df[df["end_datetime"] > df["start_datetime"]].copy()

    if "group" not in df.columns:
        df["group"] = "Н/Д"
    df["group"] = df["group"].fillna("Н/Д").astype(str)
    if "country" not in df.columns:
        df["country"] = "N/A"
    df["country"] = df["country"].fillna("N/A").astype(str).str.upper().str.strip()
    df["activity_type"] = df["activity_type"].fillna("").astype(str).str.upper().str.strip()

    group_cols = ["vehicle_name", "driver_name", "group"]

    def merge_intervals(intervals):
        merged = []
        for start_dt, end_dt in sorted(intervals, key=lambda x: x[0]):
            if not merged or start_dt > merged[-1][1]:
                merged.append([start_dt, end_dt])
            else:
                merged[-1][1] = max(merged[-1][1], end_dt)
        return [(x[0], x[1]) for x in merged]

    # CARDLESS: весь период считаем отдыхом. Параллельную не-REST активность удаляем.
    normalized_groups = []
    for _, part in df.groupby(group_cols, dropna=False, sort=False):
        part = part.copy()
        cardless = part[part["activity_type"] == "CARDLESS"]
        cardless_intervals = merge_intervals(
            list(zip(cardless["start_datetime"], cardless["end_datetime"]))
        )

        if cardless_intervals:
            overlap_cardless = pd.Series(False, index=part.index)
            for c_start, c_end in cardless_intervals:
                overlap_cardless |= (
                    (part["start_datetime"] < c_end)
                    & (part["end_datetime"] > c_start)
                )
            keep = ~(
                overlap_cardless
                & ~part["activity_type"].eq("RESTING")
            )
            part = part[keep].copy()

            base = cardless.iloc[0].copy()
            additions = []
            for c_start, c_end in cardless_intervals:
                row = base.copy()
                row["activity_type"] = "RESTING"
                row["start_datetime"] = c_start
                row["end_datetime"] = c_end
                row["duration_minutes"] = (c_end - c_start).total_seconds() / 60.0
                row["start_date"] = c_start.date()
                row["start_time"] = c_start.time()
                row["end_date"] = c_end.date()
                row["end_time"] = c_end.time()
                additions.append(row)
            if additions:
                part = pd.concat([part, pd.DataFrame(additions)], ignore_index=True)

        # Объединяем пересекающиеся RESTING/CARDLESS-интервалы, чтобы не удваивать отдых.
        rest_rows = part[part["activity_type"] == "RESTING"].copy()
        non_rest_rows = part[part["activity_type"] != "RESTING"].copy()
        if not rest_rows.empty:
            rest_rows = rest_rows.sort_values(["start_datetime", "end_datetime"])
            merged_rest = []
            current_rows = []
            current_start = None
            current_end = None
            for _, r in rest_rows.iterrows():
                if current_start is None or r["start_datetime"] > current_end:
                    if current_rows:
                        template = current_rows[-1].copy()
                        template["start_datetime"] = current_start
                        template["end_datetime"] = current_end
                        template["duration_minutes"] = (current_end - current_start).total_seconds() / 60.0
                        template["start_date"] = current_start.date()
                        template["start_time"] = current_start.time()
                        template["end_date"] = current_end.date()
                        template["end_time"] = current_end.time()
                        merged_rest.append(template)
                    current_rows = [r]
                    current_start = r["start_datetime"]
                    current_end = r["end_datetime"]
                else:
                    current_rows.append(r)
                    current_end = max(current_end, r["end_datetime"])
            if current_rows:
                template = current_rows[-1].copy()
                template["start_datetime"] = current_start
                template["end_datetime"] = current_end
                template["duration_minutes"] = (current_end - current_start).total_seconds() / 60.0
                template["start_date"] = current_start.date()
                template["start_time"] = current_start.time()
                template["end_date"] = current_end.date()
                template["end_time"] = current_end.time()
                merged_rest.append(template)
            part = pd.concat([non_rest_rows, pd.DataFrame(merged_rest)], ignore_index=True)

        normalized_groups.append(part)

    if not normalized_groups:
        return pd.DataFrame()
    df = pd.concat(normalized_groups, ignore_index=True)
    df["duration_minutes"] = (
        (df["end_datetime"] - df["start_datetime"]).dt.total_seconds() / 60.0
    )
    df = df.sort_values(group_cols + ["start_datetime", "end_datetime"]).reset_index(drop=True)

    act_type = df["activity_type"].astype(str).str.upper()
    dur_min = pd.to_numeric(df["duration_minutes"], errors="coerce").fillna(0)
    is_rest = act_type.eq("RESTING") & (dur_min >= 540)
    change_group = (
        (df["vehicle_name"] != df["vehicle_name"].shift())
        | (df["driver_name"] != df["driver_name"].shift())
        | (df["group"] != df["group"].shift())
    )
    df["shift_id"] = (is_rest | change_group).cumsum()
    df["rest_hours"] = np.where(is_rest, dur_min / 60.0, np.nan)
    df["rest_country"] = np.where(is_rest, df["country"], np.nan)
    df["pause_start_dt"] = pd.to_datetime(np.where(is_rest, df["start_datetime"], pd.NaT))
    df["pause_end_dt"] = pd.to_datetime(np.where(is_rest, df["end_datetime"], pd.NaT))

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
    ).agg(shift_start=("start_datetime", "min"))
    agg_df["rest_before_shift_hours"] = agg_df["shift_id"].map(rest_before_hours)
    agg_df["rest_country_before_shift"] = (
        agg_df["shift_id"].map(rest_before_country).fillna("N/A")
    )
    agg_df["pause_start"] = agg_df["shift_id"].map(pause_start_before)
    agg_df["pause_end"] = agg_df["shift_id"].map(pause_end_before)

    # Дата строки/паузы определяется по окончанию паузы; при отсутствии паузы — по смене.
    agg_df["control_dt"] = agg_df["pause_end"].fillna(agg_df["shift_start"])
    agg_df["dt_date"] = agg_df["control_dt"].dt.date
    agg_df["date"] = agg_df["control_dt"].dt.strftime("%Y-%m-%d")
    days_ru = {0: "ПН", 1: "ВТ", 2: "СР", 3: "ЧТ", 4: "ПТ", 5: "СБ", 6: "ВС"}
    agg_df["weekday"] = agg_df["control_dt"].dt.dayofweek.map(days_ru)
    extracted_code = agg_df["vehicle_name"].astype(str).str[1:3].str.upper()
    agg_df["vehicle_country"] = np.where(
        extracted_code.isin(["CZ", "SK"]), extracted_code, "Other"
    )
    agg_df["week_start"] = agg_df["control_dt"].apply(
        lambda x: (x - pd.Timedelta(days=x.dayofweek)).normalize()
    )

    def weekend_pair_id(pause_end):
        """Идентификатор пары ВС-ПН: дата воскресенья этой пары."""
        d = pd.Timestamp(pause_end).normalize()
        if d.dayofweek == 6:  # воскресенье
            return d
        if d.dayofweek == 0:  # понедельник
            return d - pd.Timedelta(days=1)
        return pd.NaT

    def build_period_comment(as_of, daily_week_counts, weekly_events_raw):
        as_of = pd.Timestamp(as_of)
        limit = as_of - pd.Timedelta(days=28)
        comments = []

        affected_weeks = sorted(
            w for w, cnt in daily_week_counts.items()
            if cnt >= 4 and w >= limit.normalize()
        )
        for week_start in affected_weeks:
            week_end = week_start + pd.Timedelta(days=6)
            comments.append(
                "4 сокращенных паузы в неделю с "
                f"{week_start.strftime('%d.%m.%Y')} по {week_end.strftime('%d.%m.%Y')}"
            )

        raw_window = [
            e for e in weekly_events_raw
            if limit <= e["pause_end"] <= as_of
        ]

        # Последние четыре пары выходных относительно текущей даты.
        latest_sunday = as_of.normalize() - pd.Timedelta(days=(as_of.dayofweek - 6) % 7)
        last_four_pairs = {
            latest_sunday - pd.Timedelta(weeks=i) for i in range(4)
        }
        pair_events = {}
        for event in raw_window:
            pair_id = event["pair_id"]
            if pd.notna(pair_id) and pair_id in last_four_pairs:
                pair_events.setdefault(pair_id, []).append(event)

        pair_ids = sorted(pair_events)
        pair_representatives = []
        for pair_id in pair_ids:
            events = sorted(pair_events[pair_id], key=lambda e: e["pause_end"])
            # Одна пара считается один раз. Для статуса берём событие этой пары.
            representative = events[-1]
            representative = dict(representative)
            representative["pair_id"] = pair_id
            pair_representatives.append(representative)

        count = len(pair_representatives)
        if count >= 3:
            chosen = pair_representatives[-3:]
            compensated = sum(bool(e["compensated"]) for e in chosen)
            dates = ", ".join(e["pause_end"].strftime("%d.%m.%Y") for e in chosen)
            comments.append(
                f"Более 2-х сокращенных еженедельных пауз в периоде "
                f"(компенсировано {compensated}/3): {dates}"
            )
        elif count == 2:
            chosen = pair_representatives
            compensated = sum(bool(e["compensated"]) for e in chosen)
            consecutive = (chosen[1]["pair_id"] - chosen[0]["pair_id"]).days == 7
            sequence_text = "подряд" if consecutive else "не подряд"
            dates = ", ".join(e["pause_end"].strftime("%d.%m.%Y") for e in chosen)
            comments.append(
                f"2 сокращенных еженедельных паузы в периоде, {sequence_text} "
                f"(компенсировано {compensated}/2): {dates}"
            )

        same_date_counts = {}
        for event in raw_window:
            key = event["pause_end"].date()
            same_date_counts[key] = same_date_counts.get(key, 0) + 1
        duplicate_dates = sorted(d for d, cnt in same_date_counts.items() if cnt >= 2)
        if duplicate_dates:
            dates = ", ".join(pd.Timestamp(d).strftime("%d.%m.%Y") for d in duplicate_dates)
            comments.append(f"две паузы на одну дату - {dates}")

        return " • ".join(comments)

    processed_rows = []
    for _, v_group in agg_df.sort_values(
        ["vehicle_name", "control_dt", "shift_start"]
    ).groupby("vehicle_name", sort=False):
        v_group = v_group.reset_index(drop=True)
        active_debts = []
        weekly_events_raw = []
        daily_week_counts = {}

        for _, row in v_group.iterrows():
            h = row["rest_before_shift_hours"]
            p_end = row["pause_end"]
            control_dt = row["control_dt"]
            limit = control_dt - pd.Timedelta(days=28)
            active_debts = [d for d in active_debts if d["created_at"] >= limit]

            color_type = ""
            display_str = f"{h:.2f}" if pd.notna(h) else ""
            violation_description = ""

            if pd.isna(h):
                debt_bal = float(sum(d["debt_hours"] for d in active_debts))
                row["debt_balance"] = debt_bal
                if active_debts and debt_bal > 0:
                    oldest = min(active_debts, key=lambda d: d["created_at"])
                    row["debt_days_counter"] = (
                        control_dt.normalize() - oldest["created_at"].normalize()
                    ).days + 1
                else:
                    row["debt_days_counter"] = 0
                row["status_color"] = ""
                row["display_text"] = ""
                row["violation_description"] = ""
                row["period_comment"] = build_period_comment(
                    control_dt, daily_week_counts, weekly_events_raw
                )
                processed_rows.append(row)
                continue

            end_weekday = p_end.dayofweek if pd.notna(p_end) else control_dt.dayofweek
            is_weekend_end = end_weekday in [0, 6]
            total_debt_hours = sum(d["debt_hours"] for d in active_debts)

            if h < 9.0:
                color_type = "critical"
                violation_description = "Критическое нарушение: Слишком короткая суточная пауза"

            elif 9.0 <= h < 11.0:
                if is_weekend_end:
                    color_type = "red"
                    violation_description = "Нарушение: Пауза менее 24 часов на выходных"
                else:
                    color_type = "yellow"
                    violation_description = "Сокращенная суточная пауза (будни)"
                week_key = row["week_start"]
                daily_week_counts[week_key] = daily_week_counts.get(week_key, 0) + 1
                if daily_week_counts[week_key] >= 4:
                    color_type = "red"
                    violation_description = "Превышение лимита сокращенных суточных пауз в неделю"

            elif h >= 11.0:
                if h >= 45.0:
                    required_h = 45.0 + total_debt_hours
                    if active_debts and h >= required_h:
                        extra_h = h - 45.0
                        color_type = "green"
                        display_str = f"45+{extra_h:.2f}"
                        violation_description = "Компенсация всего пакета долгов"
                        compensation_dt = pd.Timestamp(p_end if pd.notna(p_end) else control_dt)
                        for debt in active_debts:
                            if compensation_dt > debt["created_at"]:
                                debt["event"]["compensated"] = True
                                debt["event"]["compensated_at"] = compensation_dt
                        active_debts.clear()
                    else:
                        color_type = "blue"

                elif is_weekend_end:
                    if h < 24.0:
                        color_type = "red"
                        violation_description = "Нарушение: Пауза менее 24 часов на выходных"
                    else:
                        pause_end_ref = pd.Timestamp(p_end if pd.notna(p_end) else control_dt)
                        pair_id = weekend_pair_id(pause_end_ref)
                        event = {
                            "pause_end": pause_end_ref,
                            "pair_id": pair_id,
                            "debt_hours": 45.0 - h,
                            "compensated": False,
                            "compensated_at": None,
                        }
                        weekly_events_raw.append(event)
                        active_debts.append({
                            "debt_hours": 45.0 - h,
                            "created_at": pause_end_ref,
                            "event": event,
                        })

                        latest_sunday = pause_end_ref.normalize() - pd.Timedelta(
                            days=(pause_end_ref.dayofweek - 6) % 7
                        )
                        last_four_pairs = {
                            latest_sunday - pd.Timedelta(weeks=i) for i in range(4)
                        }
                        unique_pairs = {
                            e["pair_id"] for e in weekly_events_raw
                            if pd.notna(e["pair_id"])
                            and e["pair_id"] in last_four_pairs
                            and e["pause_end"] >= limit
                        }
                        pair_count = len(unique_pairs)
                        if pair_count >= 3:
                            color_type = "red"
                            violation_description = "Более 2-х сокращенных еженедельных пауз в периоде"
                        elif pair_count == 2:
                            ordered = sorted(unique_pairs)
                            consecutive = (ordered[-1] - ordered[-2]).days == 7
                            color_type = "orange"
                            violation_description = (
                                "2 сокращенных еженедельных паузы в периоде, "
                                + ("подряд" if consecutive else "не подряд")
                            )
                        else:
                            color_type = "orange"
                            violation_description = "Сокращенная еженедельная пауза. Создан долг"

                else:
                    max_debt_age_days = 0
                    if active_debts:
                        oldest_dt = min(d["created_at"] for d in active_debts)
                        max_debt_age_days = (
                            control_dt.normalize() - oldest_dt.normalize()
                        ).days
                    required_h = 11.0 + total_debt_hours
                    if active_debts and h >= required_h and max_debt_age_days <= 7:
                        extra_h = h - 11.0
                        color_type = "green"
                        display_str = f"11+{extra_h:.2f}"
                        violation_description = "Компенсация всего пакета долгов"
                        compensation_dt = pd.Timestamp(p_end if pd.notna(p_end) else control_dt)
                        for debt in active_debts:
                            if compensation_dt > debt["created_at"]:
                                debt["event"]["compensated"] = True
                                debt["event"]["compensated_at"] = compensation_dt
                        active_debts.clear()

            active_debts = [d for d in active_debts if d["created_at"] >= limit]
            debt_bal = float(sum(d["debt_hours"] for d in active_debts))
            row["debt_balance"] = debt_bal
            if active_debts and debt_bal > 0:
                oldest = min(active_debts, key=lambda d: d["created_at"])
                row["debt_days_counter"] = (
                    control_dt.normalize() - oldest["created_at"].normalize()
                ).days + 1
            else:
                row["debt_days_counter"] = 0
            row["status_color"] = color_type
            row["display_text"] = display_str
            row["violation_description"] = violation_description
            row["period_comment"] = build_period_comment(
                control_dt, daily_week_counts, weekly_events_raw
            )
            processed_rows.append(row)

    res_df = pd.DataFrame(processed_rows)
    res_df["pause_start"] = pd.to_datetime(res_df["pause_start"]).dt.strftime("%Y-%m-%d %H:%M").fillna("-")
    res_df["pause_end"] = pd.to_datetime(res_df["pause_end"]).dt.strftime("%Y-%m-%d %H:%M").fillna("-")
    cols = [
        "dt_date", "date", "weekday", "group", "vehicle_country",
        "vehicle_name", "driver_name", "rest_country_before_shift",
        "rest_before_shift_hours", "display_text", "status_color",
        "violation_description", "pause_start", "pause_end",
        "debt_balance", "debt_days_counter", "period_comment",
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

    max_dt_pd = pd.to_datetime(max_date)
    max_week_start = max_dt_pd - pd.Timedelta(days=max_dt_pd.dayofweek)
    default_start_dt = max_week_start - pd.Timedelta(weeks=3)
    default_start_date = max(min_date, default_start_dt.date())
    default_end_date = max_date

    # Актуальный статус машины всегда берём из последней строки этой машины.
    latest_vehicle_state = (
        daily_df.assign(_pause_end_sort=pd.to_datetime(daily_df["pause_end"], errors="coerce"))
        .sort_values(["vehicle_name", "dt_date", "_pause_end_sort", "date"])
        .groupby("vehicle_name", as_index=False)
        .last()
    )
    latest_debts_raw = latest_vehicle_state.set_index("vehicle_name")["debt_balance"].astype(float)
    vehicles_with_debt = latest_debts_raw[latest_debts_raw > 0].index.tolist()

    # Нарушением считаем красный/критический статус, а также оранжевое предупреждение
    # по сокращённым еженедельным паузам.
    vehicles_with_violations = daily_df.loc[
        daily_df["status_color"].isin(["red", "critical", "orange"]),
        "vehicle_name",
    ].dropna().unique().tolist()

    # Навигация и единственный фильтр дат находятся над страницей.
    nav_col, date_col = st.columns([4, 1])
    with nav_col:
        nav_page = st.radio(
            "Навигация",
            options=["Main", "Calendar"],
            horizontal=True,
            label_visibility="collapsed",
        )
    with date_col:
        date_range = st.date_input(
            "Период дат",
            value=(default_start_date, default_end_date),
            min_value=min_date,
            max_value=max_date,
            key="date_range",
        )
    if isinstance(date_range, tuple) and len(date_range) == 2:
        period_start, period_end = date_range
    else:
        period_start, period_end = min_date, max_date

    vehicles_with_comment = latest_vehicle_state.loc[
        latest_vehicle_state["period_comment"].fillna("").str.strip().ne(""),
        "vehicle_name",
    ].dropna().tolist()

    st.sidebar.markdown("<div class='filter-card'>", unsafe_allow_html=True)
    only_debt_vehicles = st.sidebar.checkbox(
        "⚠️ Только машины с долгом", value=False, key="only_debt_checkbox"
    )
    only_comment_vehicles = st.sidebar.checkbox(
        "💬 Только машины с комментарием", value=False, key="only_comment_checkbox"
    )
    st.sidebar.markdown("</div>", unsafe_allow_html=True)

    def render_select_filter(label, options, key_prefix):
        sel_key = f"{key_prefix}_select"
        all_opts = sorted(list(set([
            str(x) for x in options if pd.notna(x) and str(x).strip() != ""
        ])))
        st.sidebar.markdown(
            f"<div class='filter-card'><div class='filter-card-title'>{label}</div>",
            unsafe_allow_html=True,
        )
        b1, b2 = st.sidebar.columns(2)
        b1.button("Все", key=f"{key_prefix}_all", on_click=reset_select_key, args=(sel_key, all_opts))
        b2.button("✖ Сброс", key=f"{key_prefix}_clr", on_click=reset_select_key, args=(sel_key, []))
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
        "Страна", ["CZ", "SK", "Other"], "v_countries"
    )
    selected_groups = render_select_filter("Группа", daily_df["group"].unique(), "groups")
    selected_vehicles = render_select_filter("Машина", daily_df["vehicle_name"].unique(), "vehicles")
    selected_drivers = render_select_filter("Водитель", daily_df["driver_name"].unique(), "drivers")

    def render_range_filter(label, max_val, key_prefix, step=0.5, unit="ч"):
        min_k, max_k = f"{key_prefix}_min", f"{key_prefix}_max"
        if min_k not in st.session_state:
            st.session_state[min_k] = 0.0
        if max_k not in st.session_state:
            st.session_state[max_k] = float(max_val)
        st.sidebar.markdown(
            f"<div class='filter-card'><div class='filter-card-title'>{label} ({unit})</div>",
            unsafe_allow_html=True,
        )
        c1, c2, c3 = st.sidebar.columns([2, 2, 1])
        v_min = c1.number_input("От", 0.0, float(max_val), step=step, key=min_k, label_visibility="collapsed")
        v_max = c2.number_input("До", 0.0, float(max_val), step=step, key=max_k, label_visibility="collapsed")
        c3.button("✖", key=f"{key_prefix}_rst", on_click=reset_range_key, args=(min_k, max_k, max_val))
        st.sidebar.markdown("</div>", unsafe_allow_html=True)
        return v_min, v_max

    max_rest = max(float(daily_df["rest_before_shift_hours"].max() or 24.0) + 1.0, 10.0)
    rest_min, rest_max = render_range_filter("Часов отдыха до смены", max_rest, "rest")
    selected_countries = render_select_filter(
        "Страна отдыха до смены",
        daily_df["rest_country_before_shift"].unique(),
        "countries",
    )

    st.sidebar.button(
        "🔄 Сбросить ВСЕ фильтры",
        use_container_width=True,
        on_click=reset_all_filters,
    )

    # Эти фильтры формируют множество машин и не удаляют даты/строки сами по себе.
    vehicle_pool = set(daily_df["vehicle_name"].dropna().unique())
    if only_debt_vehicles:
        vehicle_pool &= set(vehicles_with_debt)
    if only_comment_vehicles:
        vehicle_pool &= set(vehicles_with_comment)
    if selected_vehicle_countries:
        vehicle_pool &= set(daily_df.loc[
            daily_df["vehicle_country"].isin(selected_vehicle_countries), "vehicle_name"
        ])
    if selected_groups:
        vehicle_pool &= set(daily_df.loc[daily_df["group"].isin(selected_groups), "vehicle_name"])
    if selected_vehicles:
        vehicle_pool &= set(selected_vehicles)
    if selected_drivers:
        vehicle_pool &= set(daily_df.loc[
            daily_df["driver_name"].isin(selected_drivers), "vehicle_name"
        ])

    # Период, часы отдыха и страна отдыха фильтруют конкретные строки.
    mask = daily_df["vehicle_name"].isin(vehicle_pool)
    mask &= daily_df["dt_date"].between(period_start, period_end)
    mask &= (
        daily_df["rest_before_shift_hours"].between(rest_min, rest_max)
        | daily_df["rest_before_shift_hours"].isna()
    )
    if selected_countries:
        mask &= daily_df["rest_country_before_shift"].isin(selected_countries)
    filtered = daily_df[mask].drop(columns=["dt_date"]).reset_index(drop=True)

    def render_data_table(data_to_render, title_prefix="Мониторинг смен"):
        filtered_display = data_to_render.copy()

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
                "violation_description": "Комментарий",
                "pause_start": "Начало паузы",
                "pause_end": "Конец паузы",
                "debt_balance": "Компенсация (ч.)",
                "debt_days_counter": "Возраст долга",
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

            styler = df.style.apply(style_specific_row_and_cell, axis=1).format({
                "Компенсация (ч.)": "{:.2f}",
                "Возраст долга": lambda x: f"{int(x)}" if pd.notna(x) and x > 0 else "-"
            })
            
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
                f"<div class='main-header' style='margin-top: 15px;'>{title_prefix} (найдено:"
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
            styled_display = apply_table_styling(render_df.head(500), data_to_render)
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

    if nav_page == "Main":
        render_data_table(filtered, "Мониторинг смен")

    elif nav_page == "Calendar":
        st.markdown("<div class='main-header' style='margin-top: 15px;'>Актуальное состояние машин</div>", unsafe_allow_html=True)

        latest_summary = (
            daily_df.sort_values(["vehicle_name", "dt_date", "pause_end", "date"])
            .groupby("vehicle_name", as_index=False)
            .last()
        )
        summary_df = latest_summary[[
            "group", "vehicle_name", "debt_balance", "debt_days_counter", "period_comment"
        ]].copy()
        summary_df.columns = [
            "Группа", "Машина", "Актуальная компенсация",
            "Возраст актуального долга", "Комментарий"
        ]

        st.markdown("**Фильтры актуального состояния**")
        f1, f2, f3 = st.columns([1, 1, 2])
        max_comp = max(float(summary_df["Актуальная компенсация"].max() or 0.0), 0.0)
        max_age = max(int(summary_df["Возраст актуального долга"].max() or 0), 0)
        with f1:
            comp_filter = st.slider(
                "Актуальная компенсация",
                min_value=0.0,
                max_value=max(max_comp, 0.5),
                value=(0.0, max(max_comp, 0.5)),
                step=0.5,
                key="calendar_comp_filter",
            )
        with f2:
            age_filter = st.slider(
                "Возраст актуального долга",
                min_value=0,
                max_value=max(max_age, 1),
                value=(0, max(max_age, 1)),
                step=1,
                key="calendar_age_filter",
            )
        comment_categories = [
            "4 сокращенных паузы в неделю",
            "2 сокращенных еженедельных паузы в периоде",
            "Более 2-х сокращенных еженедельных пауз в периоде",
            "две паузы на одну дату",
        ]
        with f3:
            selected_summary_comments = st.multiselect(
                "Актуальный комментарий",
                options=comment_categories,
                key="calendar_comment_filter",
                placeholder="Все комментарии",
            )

        summary_mask = (
            summary_df["Актуальная компенсация"].between(comp_filter[0], comp_filter[1])
            & summary_df["Возраст актуального долга"].between(age_filter[0], age_filter[1])
        )

        # Общие фильтры применяются именно к машинам, а не к отдельным датам.
        summary_mask &= summary_df["Машина"].isin(vehicle_pool)
        if selected_summary_comments:
            summary_mask &= summary_df["Комментарий"].fillna("").apply(
                lambda text: any(category in text for category in selected_summary_comments)
            )
        filtered_summary = summary_df[summary_mask].sort_values(
            ["Группа", "Машина"]
        ).reset_index(drop=True)

        st.dataframe(
            filtered_summary.style.format({
                "Актуальная компенсация": "{:.2f}",
                "Возраст актуального долга": lambda x: f"{int(x)}" if pd.notna(x) and x > 0 else "-",
            }),
            use_container_width=True,
            hide_index=True,
            height=min(520, 70 + 35 * max(len(filtered_summary), 1)),
            column_config={
                "Группа": st.column_config.TextColumn("Группа", width="small"),
                "Машина": st.column_config.TextColumn("Машина", width="small"),
                "Актуальная компенсация": st.column_config.NumberColumn(
                    "Актуальная компенсация", width="small", format="%.2f"
                ),
                "Возраст актуального долга": st.column_config.NumberColumn(
                    "Возраст актуального долга", width="small", format="%d"
                ),
                "Комментарий": st.column_config.TextColumn("Комментарий", width="large"),
            },
        )

        st.markdown("<div class='main-header' style='margin-top: 25px;'>Календарная матрица отдыха машин</div>", unsafe_allow_html=True)

        # Календарь не фильтруется по выбранному диапазону дат.
        # В нём остаются все даты, но только для машин, прошедших фильтры
        # актуальной компенсации, возраста долга, комментария и общие чекбоксы.
        calendar_vehicle_names = filtered_summary["Машина"].dropna().unique().tolist()
        calendar_mask = daily_df["vehicle_name"].isin(calendar_vehicle_names)
        calendar_mask &= daily_df["dt_date"].between(period_start, period_end)
        calendar_mask &= (
            daily_df["rest_before_shift_hours"].between(rest_min, rest_max)
            | daily_df["rest_before_shift_hours"].isna()
        )
        if selected_countries:
            calendar_mask &= daily_df["rest_country_before_shift"].isin(selected_countries)
        calendar_filtered = daily_df[calendar_mask].drop(columns=["dt_date"]).reset_index(drop=True)

        if not calendar_filtered.empty:
            pivot_df = calendar_filtered.copy()

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

            # Набор дат календаря всегда фиксирован по полному daily_df.
            # Фильтры меняют только строки-машины и никогда не удаляют столбцы дат.
            all_calendar_dates = pd.date_range(
                period_start, period_end, freq="D"
            ).strftime("%Y-%m-%d").tolist()

            text_matrix = grouped_matrix.pivot(
                index="vehicle_name", columns="date", values="display_text"
            ).reindex(columns=all_calendar_dates).fillna("")

            hours_matrix = grouped_matrix.pivot(
                index="vehicle_name", columns="date", values="max_hours"
            ).reindex(columns=all_calendar_dates)

            display_calendar = text_matrix.copy()
            for col in hours_matrix.columns:
                for idx in hours_matrix.index:
                    h_val = hours_matrix.loc[idx, col]
                    txt_val = text_matrix.loc[idx, col]
                    if pd.notna(h_val):
                        display_calendar.loc[idx, col] = txt_val if txt_val else f"{h_val:.2f}"
                    else:
                        display_calendar.loc[idx, col] = ""

            latest_row_per_vehicle = daily_df.sort_values(["vehicle_name", "date"]).groupby("vehicle_name").last()
            latest_debts_float = latest_row_per_vehicle["debt_balance"].astype(float)
            latest_days_float = latest_row_per_vehicle["debt_days_counter"].astype(float)

            display_calendar["Компенсация (ч.)"] = display_calendar.index.map(latest_debts_float).fillna(0.0).astype(float)
            display_calendar["Возраст долга"] = display_calendar.index.map(latest_days_float).fillna(0.0).astype(int)

            vehicle_to_group = daily_df.drop_duplicates("vehicle_name").set_index("vehicle_name")["group"]
            display_calendar.insert(0, "Группа", display_calendar.index.map(vehicle_to_group).fillna("Н/Д"))

            middle_cols = [c for c in display_calendar.columns if c not in ["Группа", "Компенсация (ч.)", "Возраст долга"]]
            cols_order = ["Группа"] + middle_cols + ["Компенсация (ч.)", "Возраст долга"]
            display_calendar = display_calendar[cols_order]

            new_column_names = {}
            for col in display_calendar.columns:
                if col in ["Группа", "Компенсация (ч.)", "Возраст долга"]:
                    new_column_names[col] = col
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
            
            display_calendar = display_calendar.rename(columns=new_column_names)
            color_matrix = (
                grouped_matrix.pivot(
                    index="vehicle_name", columns="date", values="status_color"
                )
                .reindex(columns=all_calendar_dates)
                .fillna("")
                .rename(columns=new_column_names)
            )

            def style_calendar_cell(data):
                df_styles = pd.DataFrame('', index=display_calendar.index, columns=display_calendar.columns)
                
                for new_col in display_calendar.columns:
                    if new_col in ["Группа", "Компенсация (ч.)", "Возраст долга"]:
                        if new_col == "Компенсация (ч.)":
                            for idx in display_calendar.index:
                                try:
                                    val = float(latest_debts_float.get(idx, 0.0))
                                except:
                                    val = 0.0
                                if val > 0:
                                    df_styles.loc[idx, new_col] = "font-weight: bold; color: #9A3412; background-color: #FFEDD5;"
                        elif new_col == "Возраст долга":
                            for idx in display_calendar.index:
                                try:
                                    d_val = int(latest_days_float.get(idx, 0))
                                except:
                                    d_val = 0
                                if d_val > 0:
                                    df_styles.loc[idx, new_col] = "font-weight: bold; color: #1E3A8A; background-color: #DBEAFE;"
                        continue

                    for idx in display_calendar.index:
                        c_type = color_matrix.loc[idx, new_col] if new_col in color_matrix.columns and idx in display_calendar.index else ""
                        orig_col_key = [k for k, v in new_column_names.items() if v == new_col]
                        h_val = hours_matrix.loc[idx, orig_col_key[0]] if orig_col_key and idx in hours_matrix.index and orig_col_key[0] in hours_matrix.columns else np.nan
                        
                        bg_style = get_cell_style(c_type)
                        if not bg_style and pd.notna(h_val) and h_val >= 45.0:
                            bg_style = get_cell_style("blue")

                        if bg_style:
                            df_styles.loc[idx, new_col] = bg_style

                return df_styles

            styled_calendar = display_calendar.style.apply(style_calendar_cell, axis=None).format({
                "Компенсация (ч.)": "{:.2f}",
                "Возраст долга": lambda x: f"{int(x)}" if pd.notna(x) and x > 0 else "-"
            })
            
            st.dataframe(
                styled_calendar,
                use_container_width=True,
                height=750,
                column_config={
                    "Группа": st.column_config.TextColumn("Группа", pinned=True),
                }
            )
        else:
            st.info("Нет данных для отображения матрицы.")


else:
    st.markdown(
        "<div class='main-header'>Мониторинг смен</div>", unsafe_allow_html=True
    )
    st.info("Загрузите CSV или Excel файл в панели слева.")
