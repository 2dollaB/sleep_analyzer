from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional, Dict

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


# ----------------- BCD / datetime helpers ----------------- #

def bcd_to_int(b: int) -> int:
    """Convert 1 byte from BCD (e.g., 0x25 -> 25)."""
    return (b >> 4) * 10 + (b & 0x0F)


def parse_device_datetime(tokens: List[str], idx: int) -> datetime:
    """
    Decode a timestamp from tokens:
    YY MM DD hh mm ss   (each 1 byte, hex printed)
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


# ----------------- Data classes ----------------- #

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
    hrv: int   # ← OVDJE: polje se sada zove hrv


# ----------------- Parse log ----------------- #

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

        # ---- 53 packets: sleep minute codes ----
        if cmd == "53":
            # Format from your logs:
            # 53 00 00 YY MM DD hh mm ss LEN SD1 SD2 ...
            if len(tokens) < 11:
                continue

            start_dt = parse_device_datetime(tokens, 3)
            length = int(tokens[9], 16)
            stage_bytes = tokens[10:10 + length]

            for i, sb in enumerate(stage_bytes):
                code = int(sb, 16)
                minute_dt = start_dt + timedelta(minutes=i)
                sleep_minutes.append(
                    SleepMinute(t=minute_dt, raw_code=code)
                )

        # ---- 55 packets: HR ----
        elif cmd == "55":
            # 55 id1 id2 YY MM DD hh mm ss HR
            if len(tokens) < 10:
                continue
            dt = parse_device_datetime(tokens, 3)
            hr = int(tokens[-1], 16)
            hr_samples.append(HrSample(t=dt, hr=hr))

        # ---- 56 packets: HRV ----
        elif cmd == "56":
            if len(tokens) < 11:
                continue
            dt = parse_device_datetime(tokens, 3)
            # heuristic metric: uzimamo tokens[9] kao HRV indeks
            hrv_val = int(tokens[9], 16)
            hrv_samples.append(HrvSample(t=dt, hrv=hrv_val))

        # ---- 52 packets ignored for now ----

    return sleep_minutes, hr_samples, hrv_samples


# ----------------- Attach nearest HR / HRV ----------------- #

def attach_nearest(samples, minutes: List[SleepMinute], attr: str, max_diff_min: float):
    """
    samples: lista objekata koji imaju atribut 't' i npr. 'hr' ili 'hrv'
    attr: ime atributa u samples i ime polja koje želimo postaviti u SleepMinute
    """
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


# ----------------- Split into sleep sessions ----------------- #

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


# ----------------- Classification ----------------- #

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


# ----------------- DataFrame builder ----------------- #

def build_dataframe(session: List[SleepMinute]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time": [m.t for m in session],
            "stage": [m.stage for m in session],
            "raw_code": [m.raw_code for m in session],
            "hr": [m.hr for m in session],
            "hrv": [m.hrv for m in session],
        }
    ).sort_values("time")


# ----------------- Stage duration summary ----------------- #

def summarize_stages(df: pd.DataFrame) -> Dict[str, int]:
    return df["stage"].value_counts().to_dict()


def format_hm(minutes: int) -> str:
    h = minutes // 60
    m = minutes % 60
    return f"{h}h {m:02d}m"


# ----------------- Hypnogram plot ----------------- #

def plot_hypnogram(df: pd.DataFrame):
    if df.empty:
        return None

    stage_order = ["AWAKE", "REM", "LIGHT", "DEEP"]
    y_map = {s: i for i, s in enumerate(stage_order)}

    df = df.copy().sort_values("time")
    df["y"] = df["stage"].map(y_map)

    blocks = []
    if not df.empty:
        current_stage = df.iloc[0]["stage"]
        start_time = df.iloc[0]["time"]
        prev_time = start_time

        for _, row in df.iloc[1:].iterrows():
            t = row["time"]
            stg = row["stage"]
            if stg != current_stage or (t - prev_time).total_seconds() > 90:
                blocks.append((current_stage, start_time, prev_time + timedelta(minutes=1)))
                current_stage = stg
                start_time = t
            prev_time = t
        blocks.append((current_stage, start_time, prev_time + timedelta(minutes=1)))

    fig, ax = plt.subplots(figsize=(10, 3))

    colors = {
        "AWAKE": "#ffffff",
        "REM": "#4ea5ff",
        "LIGHT": "#7cc8ff",
        "DEEP": "#274b8f",
        "UNKNOWN": "#aaaaaa",
    }

    t0 = df["time"].min()

    for stg, s, e in blocks:
        if stg not in y_map:
            continue
        left = (s - t0).total_seconds() / 60.0
        width = (e - s).total_seconds() / 60.0
        ax.barh(
            y=y_map[stg],
            width=width,
            left=left,
            height=0.8,
            color=colors.get(stg),
            edgecolor="none",
        )

    ax.set_yticks(list(y_map.values()))
    ax.set_yticklabels(stage_order)
    ax.set_xlabel("Minutes from sleep start")
    ax.invert_yaxis()
    ax.grid(axis="x", linestyle="--", alpha=0.3)

    return fig


# ----------------- Streamlit UI ----------------- #

def main():
    st.set_page_config(page_title="Sleep Viewer", layout="wide")
    st.title("Sleep Viewer – Raw Log → Hypnogram")

    uploaded = st.file_uploader("Upload your sleep log (.txt)", type=["txt"])

    if not uploaded:
        st.info("Upload your TXT log file.")
        return

    text = uploaded.read().decode("utf-8", errors="ignore")

    sleep_minutes, hr_samples, hrv_samples = parse_log_text(text)

    if not sleep_minutes:
        st.error("No 53 packets found (sleep data missing).")
        return

    # HR -> SleepMinute.hr
    attach_nearest(hr_samples, sleep_minutes, "hr", max_diff_min=10)

    # HRV -> SleepMinute.hrv
    attach_nearest(hrv_samples, sleep_minutes, "hrv", max_diff_min=30)

    apply_stage_mapping(sleep_minutes)

    sessions = split_sessions(sleep_minutes, gap_min=30)
    sessions = sorted(sessions, key=lambda s: s[-1].t)
    session = sessions[-1]

    df = build_dataframe(session)

    start = df["time"].min()
    end = df["time"].max() + timedelta(minutes=1)
    total_min = int((end - start).total_seconds() / 60)
    stage_counts = summarize_stages(df)

    st.subheader("Sleep Summary")
    c1, c2 = st.columns(2)
    c1.metric("Sleep Window", f"{start.strftime('%H:%M')} – {end.strftime('%H:%M')}")
    c2.metric("Total Duration", format_hm(total_min))

    st.write("**Stages:**")
    for stage in ["AWAKE", "REM", "LIGHT", "DEEP", "UNKNOWN"]:
        mins = stage_counts.get(stage, 0)
        if mins > 0:
            st.write(f"- {stage}: {format_hm(mins)} ({mins} min)")

    st.subheader("Hypnogram")
    fig = plot_hypnogram(df)
    if fig:
        st.pyplot(fig)

    with st.expander("Debug Table"):
        st.dataframe(df)


if __name__ == "__main__":
    main()
