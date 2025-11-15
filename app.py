from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional, Dict

import pandas as pd
import plotly.express as px
import streamlit as st


# =========================
# Helpers (BCD / datetime)
# =========================

def bcd_to_int(b: int) -> int:
    return (b >> 4) * 10 + (b & 0x0F)


def parse_device_datetime(tokens: List[str], idx: int) -> datetime:
    yy = int(tokens[idx], 16)
    mm = int(tokens[idx + 1], 16)
    dd = int(tokens[idx + 2], 16)
    hh = int(tokens[idx + 3], 16)
    mi = int(tokens[idx + 4], 16)
    ss = int(tokens[idx + 5], 16)

    year = 2000 + bcd_to_int(yy)
    month = bcd_to_int(mm)
    day = bcd_to_int(dd)
    hour = bcd_to_int(hh)
    minute = bcd_to_int(mi)
    second = bcd_to_int(ss)

    return datetime(year, month, day, hour, minute, second)


# =========================
# Data classes
# =========================

@dataclass
class SleepMinute:
    t: datetime
    raw_code: int
    stage: str = "UNKNOWN"
    hr: Optional[int] = None
    hrv: Optional[int] = None


@dataclass
class HrSample:
    t: datetime
    hr: int


@dataclass
class HrvSample:
    t: datetime
    hrv: int


# =========================
# Parsing log
# =========================

def parse_log_text(text: str):
    sleep_minutes: List[SleepMinute] = []
    hr_samples: List[HrSample] = []
    hrv_samples: List[HrvSample] = []

    for line in text.splitlines():
        if "onCharacteristicChanged:" not in line:
            continue

        _, payload_str = line.split("onCharacteristicChanged:", 1)
        tokens = payload_str.strip().split()
        if not tokens:
            continue

        cmd = tokens[0]

        # 53 – sleep codes
        if cmd == "53":
            if len(tokens) < 11:
                continue
            start_dt = parse_device_datetime(tokens, 3)
            length = int(tokens[9], 16)
            stage_bytes = tokens[10:10 + length]
            for i, sb in enumerate(stage_bytes):
                code = int(sb, 16)
                minute_dt = start_dt + timedelta(minutes=i)
                sleep_minutes.append(SleepMinute(t=minute_dt, raw_code=code))

        # 55 – HR
        elif cmd == "55":
            if len(tokens) < 10:
                continue
            dt = parse_device_datetime(tokens, 3)
            hr = int(tokens[-1], 16)
            hr_samples.append(HrSample(t=dt, hr=hr))

        # 56 – HRV
        elif cmd == "56":
            if len(tokens) < 11:
                continue
            dt = parse_device_datetime(tokens, 3)
            hrv_val = int(tokens[9], 16)
            hrv_samples.append(HrvSample(t=dt, hrv=hrv_val))

        # 52 – preskačemo za sada

    return sleep_minutes, hr_samples, hrv_samples


# =========================
# Attach nearest HR / HRV
# =========================

def attach_nearest(samples, minutes: List[SleepMinute], attr: str, max_diff_min: float):
    if not samples or not minutes:
        return

    samples = sorted(samples, key=lambda s: s.t)

    def nearest_value(t: datetime):
        best = None
        best_diff = None
        for s in samples:
            diff = abs((s.t - t).total_seconds())
            if best_diff is None or diff < best_diff:
                best = s
                best_diff = diff
        if best is None:
            return None
        if best_diff <= max_diff_min * 60:
            return getattr(best, attr)
        return None

    for m in minutes:
        val = nearest_value(m.t)
        if val is not None:
            setattr(m, attr, val)


# =========================
# Split into sessions
# =========================

def split_sessions(minutes: List[SleepMinute], gap_min: float = 30.0):
    if not minutes:
        return []

    minutes = sorted(minutes, key=lambda m: m.t)
    sessions = []
    current = [minutes[0]]

    for m in minutes[1:]:
        delta = (m.t - current[-1].t).total_seconds() / 60.0
        if delta > gap_min:
            sessions.append(current)
            current = [m]
        else:
            current.append(m)

    sessions.append(current)
    return sessions


# =========================
# Stage mapping
# =========================

RAW_CODE_TO_STAGE = {
    0: "UNKNOWN",
    1: "AWAKE",
    2: "LIGHT",
    3: "DEEP",
    4: "REM",
}


def apply_stage_mapping(minutes: List[SleepMinute]):
    for m in minutes:
        m.stage = RAW_CODE_TO_STAGE.get(m.raw_code, "UNKNOWN")


# =========================
# DataFrame helpers
# =========================

def build_dataframe(session: List[SleepMinute]) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "time": [m.t for m in session],
            "stage": [m.stage for m in session],
            "raw_code": [m.raw_code for m in session],
            "hr": [m.hr for m in session],
            "hrv": [m.hrv for m in session],
        }
    ).sort_values("time")
    return df


def summarize_stages(df: pd.DataFrame) -> Dict[str, int]:
    return df["stage"].value_counts().to_dict()


def format_hm(minutes: int) -> str:
    h = minutes // 60
    m = minutes % 60
    return f"{h}h {m:02d}m"


# =========================
# Plotly hypnogram
# =========================

def build_hypnogram_figure(df: pd.DataFrame):
    if df.empty:
        return None

    stage_order = ["AWAKE", "REM", "LIGHT", "DEEP"]
    # blokovi kontinuiranih faza
    df = df.copy().sort_values("time")
    blocks = []

    if not df.empty:
        current_stage = df.iloc[0]["stage"]
        start_time = df.iloc[0]["time"]
        prev_time = start_time

        for _, row in df.iloc[1:].iterrows():
            t = row["time"]
            stg = row["stage"]
            if stg != current_stage or (t - prev_time).total_seconds() > 90:
                blocks.append(
                    {"stage": current_stage, "start": start_time, "end": prev_time + timedelta(minutes=1)}
                )
                current_stage = stg
                start_time = t
            prev_time = t
        blocks.append(
            {"stage": current_stage, "start": start_time, "end": prev_time + timedelta(minutes=1)}
        )

    blocks_df = pd.DataFrame(blocks)
    # filtriraj UNKNOWN da ne šara graf bezveze
    blocks_df = blocks_df[blocks_df["stage"] != "UNKNOWN"]

    fig = px.timeline(
        blocks_df,
        x_start="start",
        x_end="end",
        y="stage",
        color="stage",
        category_orders={"stage": stage_order},
    )

    fig.update_yaxes(autorange="reversed")  # DEEP dolje
    fig.update_layout(
        xaxis_title="Time of night",
        yaxis_title="Stage",
        showlegend=True,
        height=320,
        margin=dict(l=60, r=20, t=40, b=40),
    )

    return fig


def build_hr_figure(df: pd.DataFrame):
    if df["hr"].dropna().empty:
        return None

    fig = px.line(
        df,
        x="time",
        y="hr",
        title="Heart rate during sleep",
    )
    fig.update_layout(
        xaxis_title="Time of night",
        yaxis_title="HR (bpm)",
        height=300,
        margin=dict(l=60, r=20, t=40, b=40),
    )
    return fig


# =========================
# Streamlit UI
# =========================

def main():
    st.set_page_config(page_title="Sleep Analyzer", layout="wide")
    st.title("🛏️ Sleep Analyzer")

    st.write("Upload raw BLE log (.txt) from your band and explore your sleep in detail.")

    uploaded = st.file_uploader("Upload sleep log (.txt)", type=["txt"])

    if not uploaded:
        st.info("Waiting for file…")
        return

    text = uploaded.read().decode("utf-8", errors="ignore")
    sleep_minutes, hr_samples, hrv_samples = parse_log_text(text)

    if not sleep_minutes:
        st.error("No 53 packets (sleep data) found in this file.")
        return

    # attach HR / HRV
    attach_nearest(hr_samples, sleep_minutes, "hr", max_diff_min=10)
    attach_nearest(hrv_samples, sleep_minutes, "hrv", max_diff_min=30)

    apply_stage_mapping(sleep_minutes)

    # split into sessions i odabir sesije
    sessions = split_sessions(sleep_minutes, gap_min=30)
    sessions = sorted(sessions, key=lambda s: s[-1].t)

    session_labels = []
    for i, sess in enumerate(sessions):
        df_tmp = build_dataframe(sess)
        s = df_tmp["time"].min()
        e = df_tmp["time"].max()
        session_labels.append(f"Session {i+1}  ({s.strftime('%Y-%m-%d %H:%M')} – {e.strftime('%H:%M')})")

    st.sidebar.header("Session")
    selected_idx = st.sidebar.selectbox("Choose sleep session", range(len(sessions)), format_func=lambda i: session_labels[i])
    session = sessions[selected_idx]

    df = build_dataframe(session)

    # time filter slider
    min_t, max_t = df["time"].min(), df["time"].max()
    st.sidebar.header("Time filter")
    # --- TIME SLIDER FIX (avoid datetime tuples) ---
min_t, max_t = df["time"].min(), df["time"].max()

# pretvorba u minute od početka
total_minutes = int((max_t - min_t).total_seconds() / 60)

start_min, end_min = st.sidebar.slider(
    "Visible time range (minutes from start)",
    min_value=0,
    max_value=total_minutes,
    value=(0, total_minutes),
)

# pretvorba nazad u datetime
start_t = min_t + timedelta(minutes=start_min)
end_t = min_t + timedelta(minutes=end_min)

# primijeni filter
df = df[(df["time"] >= start_t) & (df["time"] <= end_t)]

    df = df[(df["time"] >= start_t) & (df["time"] <= end_t)]

    # stage filter
    st.sidebar.header("Stages")
    all_stages = ["AWAKE", "REM", "LIGHT", "DEEP"]
    selected_stages = st.sidebar.multiselect(
        "Show stages",
        options=all_stages,
        default=all_stages,
    )
    df = df[df["stage"].isin(selected_stages)]

    # summary
    start = df["time"].min()
    end = df["time"].max() + timedelta(minutes=1)
    total_min = int((end - start).total_seconds() / 60)
    stage_counts = summarize_stages(df)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sleep window", f"{start.strftime('%H:%M')} – {end.strftime('%H:%M')}")
    c2.metric("Duration", format_hm(total_min))
    c3.metric("REM", format_hm(stage_counts.get("REM", 0)))
    c4.metric("Deep", format_hm(stage_counts.get("DEEP", 0)))

    st.markdown("### Hypnogram")
    hypno_fig = build_hypnogram_figure(df)
    if hypno_fig is not None:
        st.plotly_chart(hypno_fig, use_container_width=True)
    else:
        st.info("Not enough data to draw hypnogram.")

    st.markdown("### Heart rate")
    hr_fig = build_hr_figure(df)
    if hr_fig is not None:
        st.plotly_chart(hr_fig, use_container_width=True)
    else:
        st.info("No HR data available for this session.")

    with st.expander("Debug table (per minute)"):
        st.dataframe(df.reset_index(drop=True))


if __name__ == "__main__":
    main()
