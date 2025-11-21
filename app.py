from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional, Dict

import pandas as pd
import plotly.express as px
import streamlit as st
import re


# =========================
# Tunables (easy to tweak)
# =========================
HR_ATTACH_TOL_MIN = 5          # HR ~ every 2 min
HRV_ATTACH_TOL_MIN = 12        # HRV ~ every 10 min, allow ±12 min nearest attach
STEPS_ATTACH_TOL_MIN = 5

HRV_FILL_LIMIT_MIN = 6         # per direction limit for ffill/bfill on the minute grid
AWAKE_STEP_DELTA = 1           # >=1 step in the minute → awake
AWAKE_HR_Z = 1.2               # HR z-score threshold for awake spikes
SMOOTH_SPIKE = True            # 1-min ABA spike smoothing


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
FRAME56 = re.compile(r"\b56(?: [0-9A-F]{2}){14}\b")  # 56 + 14 bytes


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

        # --- 56 – HRV / fatigue / BP (regex-robust) -------------------------
        if "56" in tokens:
            # Search frames in the raw payload string to be resilient to odd spacing
            for m in FRAME56.finditer(payload):
                parts = m.group(0).split()
                # parts[0] == "56"
                try:
                    id1 = int(parts[1], 16)
                    id2 = int(parts[2], 16)
                    # YY MM DD HH mm SS
                    yy = int(parts[3], 16); mm = int(parts[4], 16); dd = int(parts[5], 16)
                    hh = int(parts[6], 16); mi = int(parts[7], 16); ss = int(parts[8], 16)
                    d1 = int(parts[9], 16)   # HRV
                    # d2 = int(parts[10],16) # unused
                    # d3 = int(parts[11],16) # unused
                    d4 = int(parts[12], 16)  # fatigue
                except Exception:
                    continue

                try:
                    dt = datetime(2000 + bcd_to_int(yy),
                                  bcd_to_int(mm),
                                  bcd_to_int(dd),
                                  bcd_to_int(hh),
                                  bcd_to_int(mi),
                                  bcd_to_int(ss))
                except Exception:
                    continue

                if 0 <= d1 <= 255:
                    hrv_samples.append(HrvSample(t=dt, hrv=d1, fatigue=d4, id1=id1, id2=id2))

        # --- 55 – HR: više paketa u jednoj liniji ---------------------------
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

        # --- 52 – steps (heuristika) ----------------------------------------
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

    # Dedup HRV zapisa (ako je više frameova s istim TS-om, zadrži zadnji)
    hrv_samples.sort(key=lambda s: s.t)
    dedup = []
    for s in hrv_samples:
        if dedup and abs((s.t - dedup[-1].t).total_seconds()) < 60:
            dedup[-1] = s
        else:
            dedup.append(s)
    hrv_samples = dedup

    return sleep_minutes, hr_samples, hrv_samples, step_samples


# =========================
# Fast attach via merge_asof
# =========================
def attach_nearest_fast(samples, minutes: List[SleepMinute], attr: str, max_diff_min: float):
    if not samples or not minutes:
        return
    s_times = [s.t for s in samples]
    s_vals = [getattr(s, attr) for s in samples]
    s_df = pd.DataFrame({"time": s_times, attr: s_vals}).sort_values("time")
    m_df = pd.DataFrame({"time": [m.t for m in minutes]}).sort_values("time")

    merged = pd.merge_asof(
        m_df, s_df, on="time",
        direction="nearest",
        tolerance=pd.Timedelta(minutes=max_diff_min)
    )
    # Write back
    for m, val in zip(minutes, merged[attr].tolist()):
        if pd.notna(val):
            setattr(m, attr, int(val))


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
# Custom staging (v3 with must-fix changes)
# =========================
def compute_custom_stage(df: pd.DataFrame) -> pd.Series:
    df = df.copy()

    # HR series
    if df["hr"].notna().sum() < 5:
        return df["stage"].fillna("LIGHT")

    hr_s = pd.to_numeric(df["hr"], errors="coerce").interpolate(limit_direction="both")
    hr_s = hr_s.rolling(5, min_periods=1, center=True).mean()
    hr_diff = hr_s.diff().abs().rolling(3, min_periods=1).mean()

    # HR z-score for awake spikes (global)
    sigma = float(hr_s.std(ddof=0)) or 1.0
    mu = float(hr_s.mean())
    hr_z = (hr_s - mu) / sigma

    # HRV with limited fill (avoid bridging long gaps)
    hrv_raw = pd.to_numeric(df["hrv"], errors="coerce")
    hrv_ffill = hrv_raw.fillna(method="ffill", limit=HRV_FILL_LIMIT_MIN)
    hrv_bfill = hrv_raw.fillna(method="bfill", limit=HRV_FILL_LIMIT_MIN)
    hrv_s = hrv_ffill.where(hrv_raw.notna(), hrv_bfill)
    hrv_s = hrv_s.rolling(5, min_periods=1, center=True).mean()

    # Steps → use delta (cumulative → movement in this minute)
    steps_raw = pd.to_numeric(df["steps"], errors="coerce").fillna(0)
    step_delta = steps_raw.diff().clip(lower=0).fillna(0)
    moved = step_delta >= AWAKE_STEP_DELTA

    # Adaptive thresholds (quantiles) – computed on coarse-sleep mask to avoid awake contamination
    coarse_sleep_mask = ~(moved | (hr_z >= (AWAKE_HR_Z + 0.3)))
    hr_for_q = hr_s[coarse_sleep_mask] if coarse_sleep_mask.any() else hr_s

    hr_low = float(hr_for_q.quantile(0.25))
    hr_high = float(hr_for_q.quantile(0.75))
    hr_very_high = float(hr_for_q.quantile(0.90))

    diff_low = float(hr_diff.quantile(0.25))
    diff_high = float(hr_diff.quantile(0.75))

    # HRV quantiles on available samples only
    if hrv_s.notna().sum() >= 5:
        hrv_low = float(hrv_s.quantile(0.25))
        hrv_high = float(hrv_s.quantile(0.75))
    else:
        hrv_low = hrv_high = None

    # Time fraction in the night
    t_start = df["time"].min()
    t_end = df["time"].max()
    total_sec = max(1.0, (t_end - t_start).total_seconds())

    stages = []

    for idx, row in df.iterrows():
        t = row["time"]
        hr_val = hr_s.loc[idx]
        diff_val = hr_diff.loc[idx]
        hrv_val = hrv_s.loc[idx] if pd.notna(hrv_s.loc[idx]) else None
        frac = (t - t_start).total_seconds() / total_sec

        # 1) Awake rules
        if moved.loc[idx]:
            stages.append("AWAKE")
            continue
        # z-score awake spikes anywhere in the night (not only edges)
        if (hr_z.loc[idx] >= AWAKE_HR_Z) and (diff_val >= diff_high):
            stages.append("AWAKE")
            continue
        # additional edge-boost (keep)
        if (hr_val >= hr_very_high) and (diff_val >= diff_high) and (frac < 0.1 or frac > 0.9):
            stages.append("AWAKE")
            continue

        # 2) Sleep scoring
        deep_score = 0
        rem_score = 0

        # Deep features
        if hr_val <= hr_low:
            deep_score += 2
        if diff_val <= diff_low:
            deep_score += 1
        if (hrv_high is not None) and (hrv_val is not None) and (hrv_val >= hrv_high):
            deep_score += 2
        if frac < 0.40:
            deep_score += 1

        # REM features
        if hr_val >= hr_high:
            rem_score += 2
        if diff_val >= diff_high:
            rem_score += 1
        if (hrv_low is not None) and (hrv_val is not None) and (hrv_val <= hrv_low):
            rem_score += 2
        if frac > 0.30:
            rem_score += 1

        if deep_score >= 3 and deep_score > rem_score:
            stages.append("DEEP")
        elif rem_score >= 3 and rem_score > deep_score:
            stages.append("REM")
        else:
            stages.append("LIGHT")

    s = pd.Series(stages, index=df.index)

    # Optional 1-minute spike smoother (A–B–A -> A)
    if SMOOTH_SPIKE and len(s) >= 3:
        s_smoothed = s.copy()
        for i in range(1, len(s) - 1):
            if s.iloc[i] != s.iloc[i - 1] and s.iloc[i] != s.iloc[i + 1]:
                s_smoothed.iloc[i] = s.iloc[i - 1]
        return s_smoothed

    return s


# =========================
# HRV diagnostics helpers
# =========================
def hrv_diagnostics_from_samples(samples: List[HrvSample], title: str):
    st.markdown(f"#### {title}")
    if not samples:
        st.info("No HRV samples found.")
        return
    df = pd.DataFrame({"time": [s.t for s in samples], "hrv": [s.hrv for s in samples]}).sort_values("time")
    # Gaps
    gaps = df["time"].diff().dt.total_seconds().div(60.0).iloc[1:]
    if len(gaps) > 0:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("HRV points", f"{len(df)}")
        c2.metric("Min gap", f"{gaps.min():.2f} min")
        c3.metric("Median gap", f"{gaps.median():.2f} min")
        c4.metric("Max gap", f"{gaps.max():.2f} min")
        st.caption(f"Range: {df['time'].min()} → {df['time'].max()}")
        # Gap distribution table (rounded to 0.1)
        dist = (
            gaps.round(1)
            .value_counts()
            .sort_index()
            .rename_axis("gap_min_rounded")
            .reset_index(name="count")
        )
        st.write("Gap distribution (rounded to 0.1 min):")
        st.dataframe(dist, use_container_width=True)
    else:
        st.info("Not enough HRV points to compute gaps.")

    # Runs of identical consecutive HRV values
    df["prev"] = df["hrv"].shift(1)
    df["same"] = df["hrv"] == df["prev"]
    run_id = (~df["same"]).cumsum()
    runs = (
        df.groupby(run_id)
        .agg(start=("time", "min"), end=("time", "max"), value=("hrv", "first"), length=("hrv", "size"))
        .reset_index(drop=True)
    )
    runs2 = runs[runs["length"] >= 2].sort_values(["length", "start"], ascending=[False, True])
    st.write("Consecutive identical HRV runs (length ≥ 2):")
    st.dataframe(runs2, use_container_width=True)

    # Preview first 50 HRV rows
    st.write("First 50 HRV rows:")
    st.dataframe(df[["time", "hrv"]].head(50), use_container_width=True)


def hrv_diagnostics_from_csv(uploaded_csv):
    st.markdown("#### CSV HRV diagnostics (optional)")
    try:
        df = pd.read_csv(uploaded_csv)
    except Exception as e:
        st.error(f"CSV read error: {e}")
        return

    st.caption("Detected columns:")
    st.code(", ".join(df.columns.astype(str)))

    # Heuristic column pickers
    def pick_time_column(df):
        candidates = [c for c in df.columns if any(k in c.lower() for k in ["time", "timestamp", "datetime", "date", "stamp"])]
        best_col, best_valid, best_parsed = None, -1, None
        for c in (candidates or list(df.columns)):
            parsed = pd.to_datetime(df[c], errors="coerce")
            valid = parsed.notna().sum()
            if valid > best_valid:
                best_col, best_valid, best_parsed = c, valid, parsed
        return best_col, best_parsed

    def pick_numeric_column(df, keywords):
        for c in df.columns:
            lc = c.lower()
            if any(k in lc for k in keywords):
                s = pd.to_numeric(df[c], errors="coerce")
                if s.notna().sum() > 0:
                    return c, s
        return None, None

    time_col, time_parsed = pick_time_column(df)
    hr_col, hr_series = pick_numeric_column(df, ["hr", "heart", "bpm"])
    hrv_col, hrv_series = pick_numeric_column(df, ["hrv", "rmssd", "sdnn"])
    steps_col, steps_series = pick_numeric_column(df, ["step"])

    st.write(
        f"- time: **{time_col}** | hr: **{hr_col}** | hrv: **{hrv_col}** | steps: **{steps_col}**"
    )

    tidy = pd.DataFrame(
        {
            "time": time_parsed if time_parsed is not None else pd.NaT,
            "hr": hr_series if hr_series is not None else pd.NA,
            "hrv": hrv_series if hrv_series is not None else pd.NA,
            "steps": steps_series if steps_series is not None else pd.NA,
        }
    ).dropna(subset=["time"]).sort_values("time").reset_index(drop=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("rows_total", len(df))
    c2.metric("rows_with_time", len(tidy))
    c3.metric("hr_non_null", int(tidy["hr"].notna().sum()))
    c4.metric("hrv_non_null", int(tidy["hrv"].notna().sum()))

    hrv_present = tidy.dropna(subset=["hrv"]).copy()
    if hrv_present.empty:
        st.warning("No HRV in CSV (all empty after time parsing).")
        return

    gaps = hrv_present["time"].diff().dt.total_seconds().div(60.0).iloc[1:]
    cc1, cc2, cc3, cc4 = st.columns(4)
    cc1.metric("HRV points", f"{len(hrv_present)}")
    if len(gaps) > 0:
        cc2.metric("Min gap", f"{gaps.min():.2f} min")
        cc3.metric("Median gap", f"{gaps.median():.2f} min")
        cc4.metric("Max gap", f"{gaps.max():.2f} min")
    st.caption(f"Range: {hrv_present['time'].min()} → {hrv_present['time'].max()}")

    dist = (
        gaps.round(1)
        .value_counts()
        .sort_index()
        .rename_axis("gap_min_rounded")
        .reset_index(name="count")
    )
    st.write("Gap distribution (rounded to 0.1 min):")
    st.dataframe(dist, use_container_width=True)

    # Runs of identical values
    hrv_present["prev"] = hrv_present["hrv"].shift(1)
    hrv_present["same"] = hrv_present["hrv"] == hrv_present["prev"]
    run_id = (~hrv_present["same"]).cumsum()
    runs = (
        hrv_present.groupby(run_id)
        .agg(start=("time", "min"), end=("time", "max"), value=("hrv", "first"), length=("hrv", "size"))
        .reset_index(drop=True)
    )
    runs2 = runs[runs["length"] >= 2].sort_values(["length", "start"], ascending=[False, True])

    st.write("Consecutive identical HRV runs (length ≥ 2):")
    st.dataframe(runs2, use_container_width=True)

    st.write("First 50 CSV HRV rows:")
    st.dataframe(hrv_present[["time", "hrv"]].head(50), use_container_width=True)


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

    # Fast nearest attach with tolerances
    attach_nearest_fast(hr_samples,  sleep_minutes, "hr",    max_diff_min=HR_ATTACH_TOL_MIN)
    attach_nearest_fast(hrv_samples, sleep_minutes, "hrv",   max_diff_min=HRV_ATTACH_TOL_MIN)
    attach_nearest_fast(step_samples, sleep_minutes, "steps", max_diff_min=STEPS_ATTACH_TOL_MIN)

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

    with st.expander("HRV detailed diagnostics (BLE log)"):
        hrv_diagnostics_from_samples(hrv_samples, "BLE HRV samples")

    with st.expander("CSV diagnostics (optional)"):
        csv_up = st.file_uploader("Upload CSV export (optional)", type=["csv"], key="csv_diag")
        if csv_up is not None:
            hrv_diagnostics_from_csv(csv_up)

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
