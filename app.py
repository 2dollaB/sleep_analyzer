from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional, Dict

import numpy as np
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
    stage: str = "UNKNOWN"   # mapped device stage (for comparison only)
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
    fatigue: Optional[int] = None
    id1: Optional[int] = None
    id2: Optional[int] = None


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

        # --- 56 – HRV / fatigue / BP ----------------------------------------
        # Spec-like frame: 56 ID1 ID2 YY MM DD HH mm SS D1 D2 D3 D4 CRC1 CRC2
        # D1 = HRV, D4 = fatigue (we only use D1, D4 for now)
        k = 0
        n = len(tokens)
        while k + 14 < n:
            if tokens[k] != "56":
                k += 1
                continue
            try:
                id1 = int(tokens[k + 1], 16)
                id2 = int(tokens[k + 2], 16)
                dt = parse_device_datetime(tokens, k + 3)
                d1 = int(tokens[k + 9], 16)   # HRV
                d4 = int(tokens[k + 12], 16)  # fatigue

                if 0 <= d1 <= 255:
                    hrv_samples.append(
                        HrvSample(t=dt, hrv=d1, fatigue=d4, id1=id1, id2=id2)
                    )
            except Exception:
                pass
            k += 15  # next potential 56 frame

        # --- 55 – HR: multiple frames per line ------------------------------
        i = 0
        n55 = len(tokens)
        while i + 9 < n55:
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

        # --- 52 – steps (heuristics) ----------------------------------------
        j = 0
        n52 = len(tokens)
        while j + 9 < n52:
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

    # Deduplicate HRV with same/close timestamps (keep last within 60s)
    hrv_samples.sort(key=lambda s: s.t)
    dedup: List[HrvSample] = []
    for s in hrv_samples:
        if dedup and abs((s.t - dedup[-1].t).total_seconds()) < 60:
            dedup[-1] = s
        else:
            dedup.append(s)
    hrv_samples = dedup

    return sleep_minutes, hr_samples, hrv_samples, step_samples


# =========================
# Attach nearest HR / HRV / steps to minute grid
# =========================
def attach_nearest(samples, minutes: List[SleepMinute], attr: str, max_diff_min: float):
    """
    Robust nearest attachment (O(N log N)) using vectorised search.
    """
    if not samples or not minutes:
        return

    # Sort both
    minutes_sorted = sorted(minutes, key=lambda m: m.t)
    s_times = np.array([s.t.timestamp() for s in samples], dtype="float64")
    s_vals = np.array([getattr(s, attr) for s in samples])

    m_times = np.array([m.t.timestamp() for m in minutes_sorted], dtype="float64")

    # For each minute, find nearest sample index via searchsorted
    idx = np.searchsorted(s_times, m_times, side="left")
    idx_right = np.clip(idx, 0, len(s_times) - 1)
    idx_left = np.clip(idx - 1, 0, len(s_times) - 1)

    dist_right = np.abs(s_times[idx_right] - m_times)
    dist_left = np.abs(s_times[idx_left] - m_times)

    take_left = dist_left <= dist_right
    nearest_idx = np.where(take_left, idx_left, idx_right)
    nearest_dist_sec = np.where(take_left, dist_left, dist_right)

    # Apply max distance threshold
    max_sec = max_diff_min * 60.0
    valid_mask = nearest_dist_sec <= max_sec

    for k, m in enumerate(minutes_sorted):
        if valid_mask[k]:
            setattr(m, attr, int(s_vals[nearest_idx[k]]))


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
# Device stage mapping (for comparison only)
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
    )
    # Ensure correct dtypes + sorted time
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)

    for col in ["hr", "hrv", "steps"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

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
        margin=dict(l=60, r=20, t=40),
    )
    return fig


# =========================
# Robust steps delta (FIX for diff TypeError)
# =========================
def safe_step_delta(df: pd.DataFrame, cap_per_min: Optional[float] = 200.0) -> pd.Series:
    """
    Returns per-minute step increments, resilient to:
    - object/str dtype
    - NaNs
    - counter resets (negative diffs -> 0)
    - unrealistic spikes (optional cap)
    """
    # Ensure sorted
    if "time" in df.columns:
        df = df.sort_values("time")

    steps = pd.to_numeric(df.get("steps", pd.Series(index=df.index)), errors="coerce")
    if steps.isna().all():
        return pd.Series(0.0, index=df.index, dtype="float64")

    delta = steps.diff()
    delta = delta.astype("float64")
    delta = delta.fillna(0.0)
    delta = delta.where(delta >= 0, 0.0)  # counter reset -> 0
    if cap_per_min is not None:
        delta = delta.clip(lower=0.0, upper=cap_per_min)
    delta = delta.fillna(0.0)
    return delta


# =========================
# Custom staging (score-based)
# =========================
def smooth_labels(labels: pd.Series) -> pd.Series:
    """1-min spike remover + 3-min majority filter."""
    s = labels.copy()

    # 1) remove single-minute spikes
    for i in range(1, len(s) - 1):
        if s.iloc[i] != s.iloc[i - 1] and s.iloc[i] != s.iloc[i + 1]:
            s.iloc[i] = s.iloc[i - 1]

    # 2) majority vote over window=3 (centered)
    win = 3
    out = s.copy()
    for i in range(1, len(s) - 1):
        block = s.iloc[i - 1 : i + 2]
        vote = block.value_counts().idxmax()
        out.iloc[i] = vote
    return out


def compute_custom_stage(df: pd.DataFrame) -> pd.Series:
    """
    Current V1 logic: score-based from HR (level + variability), HRV (every ~10 min),
    steps (movement proxy via safe_step_delta).
    """
    if df.empty:
        return pd.Series(dtype="object")

    df = df.sort_values("time").reset_index(drop=True)

    # Numeric coercion
    for col in ["hr", "hrv", "steps"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # HR signal (smooth + diff)
    if df["hr"].notna().sum() >= 3:
        hr_s = df["hr"].interpolate(limit_direction="both")
        hr_s = hr_s.rolling(5, center=True, min_periods=1).mean()
        hr_diff = hr_s.diff().abs().rolling(3, min_periods=1).mean()
    else:
        hr_s = df["hr"].fillna(method="ffill").fillna(method="bfill").fillna(60.0)
        hr_diff = pd.Series(0.0, index=df.index)

    # HRV (FWD/BWD fill + light smoothing) – present roughly every 10 min
    if df["hrv"].notna().sum() >= 1:
        hrv_s = df["hrv"].interpolate(limit_direction="both")
        hrv_s = hrv_s.rolling(5, center=True, min_periods=1).mean()
    else:
        hrv_s = None

    # Steps → per-minute movement proxy (ROBUST)
    step_delta = safe_step_delta(df)  # <— FIXED

    # Quantile thresholds (robust to flat signals)
    hr_low = float(hr_s.quantile(0.25)) if hr_s.notna().any() else 55.0
    hr_high = float(hr_s.quantile(0.75)) if hr_s.notna().any() else 75.0
    hr_very_high = float(hr_s.quantile(0.90)) if hr_s.notna().any() else 85.0
    diff_low = float(hr_diff.quantile(0.25)) if hr_diff.notna().any() else 0.5
    diff_high = float(hr_diff.quantile(0.75)) if hr_diff.notna().any() else 3.0

    if hrv_s is not None:
        hrv_low = float(hrv_s.quantile(0.25))
        hrv_high = float(hrv_s.quantile(0.75))
    else:
        hrv_low = hrv_high = None

    # Time fraction in session
    t0 = df["time"].min()
    t1 = df["time"].max()
    total_sec = max(1.0, (t1 - t0).total_seconds())

    stages = []
    for i, row in df.iterrows():
        t = row["time"]
        frac = (t - t0).total_seconds() / total_sec

        hr_val = float(hr_s.iloc[i]) if pd.notna(hr_s.iloc[i]) else 60.0
        diff_val = float(hr_diff.iloc[i]) if pd.notna(hr_diff.iloc[i]) else 0.0
        hrv_val = float(hrv_s.iloc[i]) if (hrv_s is not None and pd.notna(hrv_s.iloc[i])) else None
        move = float(step_delta.iloc[i]) if pd.notna(step_delta.iloc[i]) else 0.0

        # Default
        stage = "LIGHT"

        # AWAKE rules
        if move > 0.0:
            stage = "AWAKE"
            stages.append(stage)
            continue
        if hr_val >= hr_very_high and diff_val >= diff_high and (frac < 0.1 or frac > 0.9):
            stage = "AWAKE"
            stages.append(stage)
            continue

        # Score-based
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

    s = pd.Series(stages, index=df.index, dtype="object")
    s = smooth_labels(s)
    return s


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

    # Attach with realistic tolerances
    attach_nearest(hr_samples, sleep_minutes, "hr",  max_diff_min=5)     # HR ~ svake 2 min
    attach_nearest(hrv_samples, sleep_minutes, "hrv", max_diff_min=15)   # HRV ~ svakih 10 min
    attach_nearest(step_samples, sleep_minutes, "steps", max_diff_min=5)

    # (Optional) show device-mapped stage for comparison only
    apply_device_stage(sleep_minutes)

    # Sessions by gaps
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

    # *** CUSTOM stages only (ignore device staging) ***
    df_full["stage"] = compute_custom_stage(df_full)

    # Diagnostics
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

    st.sidebar.header("Stages")
    all_stages = ["AWAKE", "REM", "LIGHT", "DEEP"]
    selected_stages = st.sidebar.multiselect(
        "Show stages", options=all_stages, default=all_stages
    )
    df_stage = df_view[df_view["stage"].isin(selected_stages)]

    # --- Quick sanity checks -------------------------------------------------
    with st.expander("Signal sanity checks (session range)"):
        # HR gaps
        ts_hr = sorted([s.t for s in hr_samples if start_t <= s.t <= end_t])
        if len(ts_hr) >= 2:
            gaps_hr = pd.Series(ts_hr).diff().dt.total_seconds().div(60.0).iloc[1:]
            st.write(
                f"**HR gaps (min)** — min={gaps_hr.min():.2f}, median={gaps_hr.median():.2f}, max={gaps_hr.max():.2f}"
            )
        else:
            st.info("Not enough HR points in selected range.")

        # HRV gaps
        ts_hrv = sorted([s.t for s in hrv_samples if start_t <= s.t <= end_t])
        if len(ts_hrv) >= 2:
            gaps_hrv = pd.Series(ts_hrv).diff().dt.total_seconds().div(60.0).iloc[1:]
            st.write(
                f"**HRV gaps (min)** — min={gaps_hrv.min():.2f}, median={gaps_hrv.median():.2f}, max={gaps_hrv.max():.2f}"
            )
        else:
            st.info("Not enough HRV points in selected range.")

        # Steps sanity
        if "steps" in df_view.columns:
            # just show last few values and delta
            step_delta = safe_step_delta(df_view)
            st.write("Steps delta preview (last 20 rows):")
            st.dataframe(
                pd.DataFrame({"time": df_view["time"], "steps": df_view["steps"], "step_delta": step_delta}).tail(20),
                use_container_width=True,
            )

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

    st.markdown("### Stage durations (filtered) – source: **Custom (HR+HRV+steps)**")
    cols = st.columns(len(selected_stages))
    for col, stg in zip(cols, selected_stages):
        mins = stage_counts.get(stg, 0)
        col.metric(stg, format_hm(mins))

    st.markdown("### Hypnogram")
    hypno_fig = build_hypnogram_figure(df_stage[["time", "stage"]])
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
        st.dataframe(df_view.reset_index(drop=True), use_container_width=True)


# =========================
# MAIN
# =========================
def main():
    st.set_page_config(page_title="Sleep Analyzer", layout="wide")

    # Optional simple routing for privacy/terms:
    params = getattr(st, "query_params", None)
    if params is None:
        params = st.experimental_get_query_params()
        path = params.get("path", [None])[0] if isinstance(params.get("path", None), list) else params.get("path", None)
    else:
        path = params.get("path", None)

    if path == "privacy":
        render_privacy_policy()
        return
    if path == "terms":
        render_terms_of_service()
        return

    rolla_app()


if __name__ == "__main__":
    main()
