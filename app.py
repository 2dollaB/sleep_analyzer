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
    """Convert BCD-coded byte to int (0x24 -> 24, 0x11 -> 11)."""
    return (b >> 4) * 10 + (b & 0x0F)


def parse_device_datetime(tokens: List[str], idx: int) -> datetime:
    """
    Decode timestamp from tokens[idx:idx+6] in BCD:
    YY MM DD HH mm SS  (hex BCD)
    """
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
# Parsing log (53 / 55 / 56)
# =========================

def parse_log_text(text: str):
    sleep_minutes: List[SleepMinute] = []
    hr_samples: List[HrSample] = []
    hrv_samples: List[HrvSample] = []

    for line in text.splitlines():
        if "onCharacteristicChanged:" not in line:
            continue

        payload = line.split("onCharacteristicChanged:", 1)[1].strip()
        if not payload:
            continue
        tokens = payload.split()
        if not tokens:
            continue

        # 53 – sleep command
        # Format: 53 idx 00 YY MM DD HH mm SS LEN S1..S120
        if tokens[0] == "53":
            if len(tokens) < 11:
                continue
            try:
                start_dt = parse_device_datetime(tokens, 3)
            except Exception:
                continue

            length = int(tokens[9], 16)
            stage_bytes = tokens[10:10 + length]
            for i, sb in enumerate(stage_bytes):
                try:
                    code = int(sb, 16)
                except ValueError:
                    continue
                minute_dt = start_dt + timedelta(minutes=i)
                sleep_minutes.append(SleepMinute(t=minute_dt, raw_code=code))

        # 56 – HRV (format nije dokumentiran, ovo je samo placeholder)
        if tokens[0] == "56":
            if len(tokens) >= 10:
                try:
                    dt = parse_device_datetime(tokens, 3)
                    # uzmi neki bajt kao HRV indeks (heuristika)
                    hrv_val = int(tokens[9], 16)
                    hrv_samples.append(HrvSample(t=dt, hrv=hrv_val))
                except Exception:
                    pass

        # 55 – HR: može biti više paketa u jednoj liniji
        # Format jednog paketa: 55 idx 00 YY MM DD HH mm SS HV
        i = 0
        n = len(tokens)
        while i + 9 < n:
            if tokens[i] != "55":
                i += 1
                continue
            try:
                dt = parse_device_datetime(tokens, i + 3)
                hr_val = int(tokens[i + 9], 16)
                # sanity check
                if 30 <= hr_val <= 220:
                    hr_samples.append(HrSample(t=dt, hr=hr_val))
            except Exception:
                pass
            i += 10  # sljedeći 55 paket

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
# Split into sessions (big gaps)
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
# Stage mapping (from firmware doc)
# 01 -> deep, 02 -> light, 03 -> REM, else -> awake
# =========================

def code_to_stage(code: int) -> str:
    if code == 0x01:
        return "DEEP"
    if code == 0x02:
        return "LIGHT"
    if code == 0x03:
        return "REM"
    return "AWAKE"


def apply_stage_mapping(minutes: List[SleepMinute]):
    for m in minutes:
        m.stage = code_to_stage(m.raw_code)


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

    df = df.copy().sort_values("time")
    blocks = []

    current_stage = df.iloc[0]["stage"]
    start_time = df.iloc[0]["time"]
    prev_time = start_time

    for _, row in df.iloc[1:].iterrows():
        t = row["time"]
        stg = row["stage"]
        # prekid ako se promijeni faza ili ako je rupa > 90s
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
    blocks_df = blocks_df[blocks_df["stage"] != "UNKNOWN"]

    if blocks_df.empty:
        return None

    fig = px.timeline(
        blocks_df,
        x_start="start",
        x_end="end",
        y="stage",
        color="stage",
        category_orders={"stage": stage_order},
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(
        xaxis_title="Time of night",
        yaxis_title="Stage",
        height=320,
        margin=dict(l=60, r=20, t=40, b=40),
        legend_title="Stage",
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

    st.write("Upload raw BLE log (.txt) from your Rolla Band and explore your sleep in detail.")

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

    # split into sessions and session picker
    sessions = split_sessions(sleep_minutes, gap_min=30)
    sessions = sorted(sessions, key=lambda s: s[-1].t)

    session_labels = []
    for i, sess in enumerate(sessions):
        df_tmp = build_dataframe(sess)
        s = df_tmp["time"].min()
        e = df_tmp["time"].max()
        session_labels.append(f"Session {i + 1}  ({s.strftime('%Y-%m-%d %H:%M')} – {e.strftime('%H:%M')})")

    st.sidebar.header("Session")
    selected_idx = st.sidebar.selectbox(
        "Choose sleep session",
        range(len(sessions)),
        format_func=lambda i: session_labels[i],
    )
    session = sessions[selected_idx]

    df = build_dataframe(session)

    # -------- TIME SLIDER (int minutes, radi i na Streamlit Cloud) --------
    min_t, max_t = df["time"].min(), df["time"].max()
    total_minutes = max(1, int((max_t - min_t).total_seconds() / 60))

    st.sidebar.header("Time filter")
    start_min, end_min = st.sidebar.slider(
        "Visible time range (minutes from start)",
        min_value=0,
        max_value=total_minutes,
        value=(0, total_minutes),
    )

    start_t = min_t + timedelta(minutes=start_min)
    end_t = min_t + timedelta(minutes=end_min)

    df = df[(df["time"] >= start_t) & (df["time"] <= end_t)]

    # -------- Stage filter --------
    st.sidebar.header("Stages")
    all_stages = ["AWAKE", "REM", "LIGHT", "DEEP"]
    selected_stages = st.sidebar.multiselect(
        "Show stages",
        options=all_stages,
        default=all_stages,
    )
    df = df[df["stage"].isin(selected_stages)]

    if df.empty:
        st.warning("No data in selected time/stage range.")
        return

    # summary
    start = df["time"].min()
    end = df["time"].max() + timedelta(minutes=1)
    total_min = int((end - start).total_seconds() / 60)
    stage_counts = summarize_stages(df)

    c1, c2 =
