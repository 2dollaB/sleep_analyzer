from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional, Dict

import pandas as pd
import plotly.express as px
import streamlit as st


# =========================
# Privacy & Terms (simple)
# =========================
def render_privacy_policy():
    st.title("Privacy Policy")
    st.write(
        """
        ### Sleep Analyzer – Privacy Policy

        This project is an internal tool used for personal analysis of sleep data.
        We do not store, share or process any user data outside of this app.
        Uploaded files are processed in-memory in your session and are not persisted
        on any external server.

        **What data is processed?**
        - Sleep staging packets (cmd 53)
        - Heart rate samples (cmd 55)
        - HRV samples (cmd 56)
        - Steps (cmd 52)

        **Contact**: minarik.jan@rolla.app
        """
    )


def render_terms_of_service():
    st.title("Terms of Service")
    st.write(
        """
        ### Sleep Analyzer – Terms of Service

        This tool is intended solely for development and experimentation.
        It is **not** a medical device. Use at your own risk.

        For any questions, contact: minarik.jan@rolla.app
        """
    )


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
    stage: str = "UNKNOWN"   # mapped device stage
    hr: Optional[int] = None
    hrv: Optional[int] = None
    steps: Optional[int] = None


@dataclass
class HrSample:
    t: datetime
    hr: int


@dataclass
class HrvSample:
    t: datetime
    hrv: int


@dataclass
class StepSample:
    t: datetime
    steps: int


# =========================
# Parsing Rolla log (53 / 55 / 56 / 52)
# =========================
def parse_log_text(text: str):
    sleep_minutes: List[SleepMinute] = []
    hr_samples: List[HrSample] = []
    hrv_samples: List[HrvSample] = []
    step_samples: List[StepSample] = []

    for line in text.splitlines():
        if "onCharacteristicChanged:" not in line:
            continue

        payload = line.split("onCharacteristicChanged:", 1)[1].strip()
        if not payload:
            continue
        tokens = payload.split()
        if not tokens:
            continue

        # --- 53 – Sleep minute grid -----------------------------------------
        # Format: 53 IDX 00 YY MM DD HH mm SS LEN S1..S120
        if tokens[0] == "53":
            if len(tokens) < 11:
                continue
            try:
                start_dt = parse_device_datetime(tokens, 3)
            except Exception:
                continue

            try:
                length = int(tokens[9], 16)
            except Exception:
                continue

            stage_bytes = tokens[10:10 + length]
            for i, sb in enumerate(stage_bytes):
                try:
                    code = int(sb, 16)
                except ValueError:
                    continue
                minute_dt = start_dt + timedelta(minutes=i)
                sleep_minutes.append(SleepMinute(t=minute_dt, raw_code=code))

        # --- 56 – HRV / fatigue / BP (spec compliant) -----------------------
        # 56 ID1 ID2 YY MM DD HH mm SS D1 D2 D3 D4 [D5 D6] ...
        # D1 = HRV (1 byte)
        if tokens[0] == "56":
            if len(tokens) >= 10:
                try:
                    dt = parse_device_datetime(tokens, 3)   # YY MM DD HH mm SS (BCD)
                    hrv_val = int(tokens[9], 16)            # D1 = HRV
                    # Optional fields exist on some firmwares (not used yet):
                    # fatigue = int(tokens[12], 16) if len(tokens) > 12 else None
                    # sys_bp  = int(tokens[13], 16) if len(tokens) > 13 else None
                    # dia_bp  = int(tokens[14], 16) if len(tokens) > 14 else None
                    if 0 <= hrv_val <= 255:
                        hrv_samples.append(HrvSample(t=dt, hrv=hrv_val))
                except Exception:
                    pass

        # --- 55 – HR: može biti više paketa u jednoj liniji -----------------
        i = 0
        n = len(tokens)
        while i + 9 < n:
            if tokens[i] != "55":
                i += 1
                continue
            try:
                dt = parse_device_datetime(tokens, i + 3)
                hr_val = int(tokens[i + 9], 16)
                if 30 <= hr_val <= 220:
                    hr_samples.append(HrSample(t=dt, hr=hr_val))
            except Exception:
                pass
            i += 10

        # --- 52 – steps (heuristika) ----------------------------------------
        j = 0
        n2 = len(tokens)
        while j + 9 < n2:
            if tokens[j] != "52":
                j += 1
                continue
            try:
                dt = parse_device_datetime(tokens, j + 3)
                steps_val = int(tokens[j + 9], 16)
                step_samples.append(StepSample(t=dt, steps=steps_val))
            except Exception:
                pass
            j += 10

    return sleep_minutes, hr_samples, hrv_samples, step_samples


# =========================
# Attach nearest HR / HRV / steps to minute grid
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
# Split into sessions (gap > X min = new session)
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
# Device stage mapping (53) – for comparison
# =========================
def code_to_stage(code: int) -> str:
    if code == 0x01:
        return "DEEP"
    if code == 0x02:
        return "LIGHT"
    if code == 0x03:
        return "REM"
    return "AWAKE"


def apply_device_stage(minutes: List[SleepMinute]):
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
            "steps": [m.steps for m in session],
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
    blocks_df = blocks_df[blocks_df["stage"].isin(stage_order)]
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


# =========================
# HR line from samples
# =========================
def build_hr_figure_from_samples(samples: List[HrSample], start_t=None, end_t=None):
    if not samples:
        return None

    rows = []
    for s in samples:
        if start_t and s.t < start_t:
            continue
        if end_t and s.t > end_t:
            continue
        rows.append({"time": s.t, "hr": s.hr})

    if not rows:
        return None

    df_hr = pd.DataFrame(rows).sort_values("time")
    fig = px.line(df_hr, x="time", y="hr")
    fig.update_layout(
        xaxis_title="Time of night",
        yaxis_title="HR (bpm)",
        height=300,
        margin=dict(l=60, r=20, t=40, b=40),
    )
    return fig


# =========================
# Custom staging (v3 – same as before)
# =========================
def compute_custom_stage(df: pd.DataFrame) -> pd.Series:
    df = df.copy()

    if df["hr"].notna().sum() < 5:
        return df["stage"].fillna("LIGHT")

    hr_s = df["hr"].astype("float").interpolate(limit_direction="both")
    hr_s = hr_s.rolling(5, min_periods=1, center=True).mean()

    hrv_s = df["hrv"].astype("float")
    if hrv_s.notna().sum() > 5:
        hrv_s = hrv_s.interpolate(limit_direction="both")
        hrv_s = hrv_s.rolling(5, min_periods=1, center=True).mean()
    else:
        hrv_s = None

    steps_s = df["steps"].fillna(0)
    hr_diff = hr_s.diff().abs().rolling(3, min_periods=1).mean()

    hr_low = hr_s.quantile(0.25)
    hr_high = hr_s.quantile(0.75)
    hr_very_high = hr_s.quantile(0.90)

    if hrv_s is not None:
        hrv_low = hrv_s.quantile(0.25)
        hrv_high = hrv_s.quantile(0.75)
    else:
        hrv_low = hrv_high = None

    diff_low = hr_diff.quantile(0.25)
    diff_high = hr_diff.quantile(0.75)

    t_start = df["time"].min()
    t_end = df["time"].max()
    total_sec = max(1.0, (t_end - t_start).total_seconds())

    stages = []

    for idx, row in df.iterrows():
        t = row["time"]
        hr_val = hr_s.loc[idx]
        diff_val = hr_diff.loc[idx]
        hrv_val = hrv_s.loc[idx] if hrv_s is not None else None
        steps_val = steps_s.loc[idx]
        frac = (t - t_start).total_seconds() / total_sec

        stage = "LIGHT"

        # Awake
        if steps_val and steps_val > 0:
            stage = "AWAKE"
            stages.append(stage)
            continue

        if hr_val >= hr_very_high and diff_val >= diff_high and (frac < 0.1 or frac > 0.9):
            stage = "AWAKE"
            stages.append(stage)
            continue

        deep_score = 0
        rem_score = 0

        if hr_val <= hr_low:
            deep_score += 2
        if diff_val <= diff_low:
            deep_score += 1
        if hrv_high is not None and hrv_val is not None and hrv_val >= hrv_high:
            deep_score += 2
        if frac < 0.4:
            deep_score += 1

        if hr_val >= hr_high:
            rem_score += 2
        if diff_val >= diff_high:
            rem_score += 1
        if hrv_low is not None and hrv_val is not None and hrv_val <= hrv_low:
            rem_score += 2
        if frac > 0.3:
            rem_score += 1

        if deep_score >= 3 and deep_score > rem_score:
            stage = "DEEP"
        elif rem_score >= 3 and rem_score > deep_score:
            stage = "REM"
        else:
            stage = "LIGHT"

        stages.append(stage)

    s = pd.Series(stages, index=df.index)

    # 1-minute spike smoother
    s_smoothed = s.copy()
    for i in range(1, len(s) - 1):
        if s.iloc[i] != s.iloc[i - 1] and s.iloc[i] != s.iloc[i + 1]:
            s_smoothed.iloc[i] = s.iloc[i - 1]

    return s_smoothed


# =========================
# Debug helpers
# =========================
def debug_gaps(samples, name, start_t, end_t):
    ts = sorted([s.t for s in samples if start_t <= s.t <= end_t])
    if len(ts) < 2:
        st.info(f"Not enough {name} points in selected range.")
        return
    gaps = [(ts[i] - ts[i - 1]).total_seconds() / 60 for i in range(1, len(ts))]
    st.markdown(
        f"**{name} gaps (minutes)**: "
        f"min={min(gaps):.2f}, median={pd.Series(gaps).median():.2f}, max={max(gaps):.2f}"
    )
    st.dataframe(
        pd.Series([round(g, 1) for g in gaps], name="gap_min")
        .value_counts()
        .sort_index()
        .to_frame("count")
    )


# =========================
# Rolla app (BLE)
# =========================
def rolla_app():
    st.title("🛏️ Sleep Analyzer – Rolla band")

    uploaded = st.file_uploader("Upload Rolla BLE sleep log (.txt)", type=["txt"])
    st.caption("Supported commands in the log: 53 (sleep), 55 (HR), 56 (HRV), 52 (steps).")

    if not uploaded:
        st.info("Upload a BLE log file to analyze sleep.")
        return

    text = uploaded.read().decode("utf-8", errors="ignore")
    sleep_minutes, hr_samples, hrv_samples, step_samples = parse_log_text(text)

    if not sleep_minutes:
        st.error("No 53 packets (sleep data) found in this file.")
        return

    # attach with realistic tolerances
    attach_nearest(hr_samples, sleep_minutes, "hr", max_diff_min=5)    # HR ~ every 2 min
    attach_nearest(hrv_samples, sleep_minutes, "hrv", max_diff_min=15) # HRV ~ every 10 min
    attach_nearest(step_samples, sleep_minutes, "steps", max_diff_min=5)

    apply_device_stage(sleep_minutes)
    sessions = split_sessions(sleep_minutes, gap_min=30)
    sessions = sorted(sessions, key=lambda s: s[-1].t)

    session_labels = []
    for i, sess in enumerate(sessions):
        df_tmp = build_dataframe(sess)
        s_start = df_tmp["time"].min()
        s_end = df_tmp["time"].max()
        session_labels.append(
            f"Session {i + 1}  ({s_start.strftime('%Y-%m-%d %H:%M')} – {s_end.strftime('%H:%M')})"
        )

    st.sidebar.header("Session")
    default_index = max(0, len(sessions) - 1)  # pick the latest by default
    selected_idx = st.sidebar.selectbox(
        "Choose sleep session",
        range(len(sessions)),
        index=default_index,
        format_func=lambda i: session_labels[i],
    )
    session = sessions[selected_idx]

    df_full = build_dataframe(session)
    df_full["device_stage"] = df_full["stage"]
    df_full["custom_stage"] = compute_custom_stage(df_full)

    st.sidebar.header("Stage source")
    stage_source = st.sidebar.radio(
        "Use stages from",
        ["Device (cmd 53)", "Custom (HR+HRV+steps)"],
        index=1,
    )

    st.sidebar.caption(
        f"HR samples: {len(hr_samples)} | HRV samples: {len(hrv_samples)} | Step samples: {len(step_samples)}"
    )

    min_t, max_t = df_full["time"].min(), df_full["time"].max()
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

    df_view = df_full[(df_full["time"] >= start_t) & (df_full["time"] <= end_t)].copy()

    stage_col = "device_stage" if stage_source.startswith("Device") else "custom_stage"
    df_view["stage_used"] = df_view[stage_col]
    df_stage = df_view[["time", "stage_used"]].rename(columns={"stage_used": "stage"})

    st.sidebar.header("Stages")
    all_stages = ["AWAKE", "REM", "LIGHT", "DEEP"]
    selected_stages = st.sidebar.multiselect(
        "Show stages",
        options=all_stages,
        default=all_stages,
    )
    df_stage = df_stage[df_stage["stage"].isin(selected_stages)]

    # --- Diagnostics --------------------------------------------------------
    with st.expander("Signal sanity checks (session range)"):
        debug_gaps(hr_samples, "HR", start_t, end_t)
        debug_gaps(hrv_samples, "HRV", start_t, end_t)

    if df_stage.empty:
        st.warning("No data in selected time/stage range.")
        st.markdown("### Heart rate")
        hr_fig = build_hr_figure_from_samples(hr_samples, start_t=start_t, end_t=end_t)
        if hr_fig is not None:
            st.plotly_chart(hr_fig, use_container_width=True)
        else:
            st.info("No HR data available for this session.")
        return

    start = df_stage["time"].min()
    end = df_stage["time"].max() + timedelta(minutes=1)
    total_min = int((end - start).total_seconds() / 60)

    stage_counts = summarize_stages(df_stage)

    c1, c2 = st.columns(2)
    c1.metric("Sleep window (visible)", f"{start.strftime('%H:%M')} – {end.strftime('%H:%M')}")
    c2.metric("Duration (visible)", format_hm(total_min))

    st.markdown(f"### Stage durations (filtered) – source: **{stage_source}**")
    cols = st.columns(len(selected_stages))
    for col, stg in zip(cols, selected_stages):
        mins = stage_counts.get(stg, 0)
        col.metric(stg, format_hm(mins))

    st.markdown("### Hypnogram")
    hypno_fig = build_hypnogram_figure(df_stage)
    if hypno_fig is not None:
        st.plotly_chart(hypno_fig, use_container_width=True)
    else:
        st.info("Not enough data to draw hypnogram.")

    st.markdown("### Heart rate")
    hr_fig = build_hr_figure_from_samples(hr_samples, start_t=start_t, end_t=end_t)
    if hr_fig is not None:
        st.plotly_chart(hr_fig, use_container_width=True)
    else:
        st.info("No HR data available for this session.")

    with st.expander("Debug table (per minute)"):
        st.dataframe(df_view.reset_index(drop=True))


# =========================
# MAIN
# =========================
def main():
    st.set_page_config(page_title="Sleep Analyzer", layout="wide")

    # Optional simple routing for privacy/terms:
    params = st.query_params if hasattr(st, "query_params") else st.experimental_get_query_params()
    path = params.get("path", [None])[0] if isinstance(params.get("path", None), list) else params.get("path", None)
    if path == "privacy":
        render_privacy_policy()
        return
    if path == "terms":
        render_terms_of_service()
        return

    rolla_app()


if __name__ == "__main__":
    main()
