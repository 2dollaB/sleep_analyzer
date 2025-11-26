from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional, Dict

import pandas as pd
import plotly.express as px
import streamlit as st


# =========================
# Tunables (easy to tweak)
# =========================
HR_ATTACH_TOL_MIN = 5       # nearest HR sample tolerance
HRV_ATTACH_TOL_MIN = 12     # nearest HRV sample tolerance
STEPS_ATTACH_TOL_MIN = 5    # nearest Steps sample tolerance

HRV_FILL_LIMIT_MIN = 6      # limited ffill/bfill window for HRV (in minutes)

AWAKE_STEP_DELTA = 1        # per-minute steps delta to mark Awake
SMOOTH_SPIKE = True         # 1-min spike smoothing for stages


# =========================
# Tunables for v2 staging (UNCHANGED)
# =========================
DEEP_MIN_BOUT_MIN = 5       # minimum continuous minutes to keep as DEEP
REM_MIN_BOUT_MIN = 5        # minimum continuous minutes to keep as REM

DEEP_EARLY_FRAC = 0.5       # DEEP more likely in first 50% of night
REM_LATE_FRAC = 0.3         # REM more likely after first 30% of night

AWAKE_HR_Z_V2 = 1.5         # stricter HR spike threshold for AWAKE (v2)


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
        - Sleep minute frames (cmd 53) – used only for start/end & minute grid
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
    # We keep raw_code from 53 only to create the grid, but we NEVER use it for staging.
    raw_code: int
    hr: Optional[int] = None
    hrv: Optional[int] = None
    steps: Optional[int] = None  # cumulative if provided


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
    steps: int  # cumulative


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
                # We ignore the meaning of sb (stage code). Only build the minute timeline.
                try:
                    code = int(sb, 16)
                except ValueError:
                    code = 0x00
                minute_dt = start_dt + timedelta(minutes=i)
                sleep_minutes.append(SleepMinute(t=minute_dt, raw_code=code))

        # --- 56 – HRV / fatigue / BP (spec-robust) --------------------------
        # 56 ID1 ID2 YY MM DD HH mm SS D1 D2 D3 D4 CRC1 CRC2  (15 tokens)
        # D1 = HRV, D4 = fatigue
        k = 0
        n = len(tokens)
        while k + 14 < n:
            if tokens[k] != "56":
                k += 1
                continue
            try:
                id1 = int(tokens[k + 1], 16)
                id2 = int(tokens[k + 2], 16)
                dt = parse_device_datetime(tokens, k + 3)  # YY..SS in BCD

                d1 = int(tokens[k + 9], 16)   # HRV
                d4 = int(tokens[k + 12], 16)  # fatigue

                if 0 <= d1 <= 255:
                    hrv_samples.append(
                        HrvSample(t=dt, hrv=d1, fatigue=d4, id1=id1, id2=id2)
                    )
            except Exception:
                pass
            k += 15  # jump to next potential 56 frame

        # --- 55 – HR: possibly multiple frames in one line ------------------
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

        # --- 52 – steps (cumulative) ----------------------------------------
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

    # Dedup HRV (if multiple with same TS, keep last within 60s)
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
            return getattr(s, attr) if (s := best) else None
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
# DataFrame helpers
# =========================
def build_dataframe(session: List[SleepMinute]) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "time": [m.t for m in session],
            "raw_code": [m.raw_code for m in session],  # not used for staging
            "hr": [m.hr for m in session],
            "hrv": [m.hrv for m in session],
            "steps": [m.steps for m in session],        # cumulative if present
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
# Custom staging (v2, experimental) — UNCHANGED
# =========================
def compute_custom_stage_v2(df_in: pd.DataFrame) -> pd.Series:
    """
    Experimental v2 staging:
    - Uses HR relative to a slow baseline (~30 min)
    - Uses HR volatility and HRV as slower trends
    - Adds time-of-night bias (early = more DEEP, late = more REM)
    - Enforces minimum DEEP/REM bout length
    """
    df = df_in.copy()

    # If we have almost no HR, bail out
    if df["hr"].notna().sum() < 5:
        return pd.Series(["LIGHT"] * len(df), index=df.index)

    # ----- HR preprocessing -----
    hr = df["hr"].astype(float).interpolate(limit_direction="both")
    # Short-term smoothing
    hr_short = hr.rolling(5, min_periods=1, center=True).mean()

    # Slow baseline over ~30 minutes (assuming 1-min grid)
    hr_baseline = hr.rolling(31, min_periods=15, center=True).median()
    if hr_baseline.isna().all():
        hr_baseline = pd.Series(hr.median(), index=df.index)
    else:
        hr_baseline = hr_baseline.fillna(hr.median())

    # HR relative to baseline
    hr_rel = hr_short - hr_baseline

    # Z-scores for HR relative change
    hr_rel_mu = hr_rel.mean()
    hr_rel_sigma = hr_rel.std(ddof=0) or 1.0
    hr_rel_z = (hr_rel - hr_rel_mu) / hr_rel_sigma

    # HR volatility (short-term variability)
    hr_diff = hr_short.diff().abs().rolling(5, min_periods=1).mean()

    # ----- HRV preprocessing (slow trend) -----
    hrv = df["hrv"].astype(float)
    if hrv.notna().sum() > 5:
        # conservative filling
        hrv = hrv.ffill(limit=HRV_FILL_LIMIT_MIN).bfill(limit=HRV_FILL_LIMIT_MIN)
        # slow trend over ~30 minutes
        hrv_slow = hrv.rolling(31, min_periods=5, center=True).mean()
        if hrv_slow.notna().sum() > 5:
            hrv_mu = hrv_slow.mean()
            hrv_sigma = hrv_slow.std(ddof=0) or 1.0
            hrv_z = (hrv_slow - hrv_mu) / hrv_sigma
        else:
            hrv_slow = None
            hrv_z = None
    else:
        hrv_slow = None
        hrv_z = None

    # ----- Steps: cumulative → per-minute delta -----
    steps_cum = df["steps"].copy()
    steps_cum = steps_cum.ffill()
    step_delta = steps_cum.diff().fillna(0).clip(lower=0)

    # ----- Global thresholds (quantiles) -----
    def safe_q(s, qv):
        s_valid = s.dropna()
        if s_valid.empty:
            return None
        return float(s_valid.quantile(qv))

    hr_diff_q50 = safe_q(hr_diff, 0.50)
    hr_diff_q75 = safe_q(hr_diff, 0.75)
    hr_diff_q25 = safe_q(hr_diff, 0.25)

    hr_rel_z_q25 = safe_q(hr_rel_z, 0.25)
    hr_rel_z_q75 = safe_q(hr_rel_z, 0.75)

    if hrv_z is not None:
        hrv_z_q25 = safe_q(hrv_z, 0.25)
        hrv_z_q75 = safe_q(hrv_z, 0.75)
    else:
        hrv_z_q25 = hrv_z_q75 = None

    # ----- Time-of-night fraction -----
    t_start = df["time"].min()
    t_end = df["time"].max()
    total_sec = max(1.0, (t_end - t_start).total_seconds())

    # ----- First pass: assign raw stages -----
    raw_stages = []

    for idx in df.index:
        t = df.loc[idx, "time"]
        frac = (t - t_start).total_seconds() / total_sec

        hrz_val = hr_rel_z.loc[idx]
        hrz = 0.0 if pd.isna(hrz_val) else hrz_val

        diff_val = hr_diff.loc[idx]
        diff_val = 0.0 if pd.isna(diff_val) else diff_val

        steps_d = step_delta.loc[idx]
        steps_d = 0.0 if pd.isna(steps_d) else steps_d

        if hrv_z is not None and idx in hrv_z.index:
            hrvz_val = hrv_z.loc[idx]
            hrvz = None if pd.isna(hrvz_val) else hrvz_val
        else:
            hrvz = None

        # -------- AWAKE detection (more conservative than v1) --------
        is_awake = False

        # Obvious movement
        if steps_d >= AWAKE_STEP_DELTA:
            is_awake = True

        # HR spike + high volatility
        if (
            not is_awake
            and hr_diff_q75 is not None
            and hrz >= AWAKE_HR_Z_V2
            and diff_val >= hr_diff_q75
        ):
            is_awake = True

        # Edge boost: start/end 10% of night
        if (
            not is_awake
            and hr_diff_q75 is not None
            and hr_rel_z_q75 is not None
            and (frac < 0.10 or frac > 0.90)
            and hrz >= hr_rel_z_q75
            and diff_val >= hr_diff_q75
        ):
            is_awake = True

        if is_awake:
            raw_stages.append("AWAKE")
            continue

        # -------- Sleep stages: DEEP vs REM vs LIGHT --------
        deep_score = 0
        rem_score = 0

        # DEEP: low HR relative baseline, low volatility, high HRV
        if hr_rel_z_q25 is not None and hrz <= hr_rel_z_q25:
            deep_score += 2
        if hr_diff_q25 is not None and diff_val <= hr_diff_q25:
            deep_score += 1
        if hrv_z_q75 is not None and hrvz is not None and hrvz >= hrv_z_q75:
            deep_score += 2
        # Early-night bias
        if frac <= DEEP_EARLY_FRAC:
            deep_score += 1

        # REM: higher HR rel baseline, moderate volatility, lower HRV, later in night
        if hr_rel_z_q75 is not None and hrz >= hr_rel_z_q75:
            rem_score += 2
        if hr_diff_q50 is not None and diff_val >= hr_diff_q50:
            rem_score += 1
        if hrv_z_q25 is not None and hrvz is not None and hrvz <= hrv_z_q25:
            rem_score += 2
        # Late-night bias
        if frac >= REM_LATE_FRAC:
            rem_score += 1

        if deep_score >= 3 and deep_score > rem_score:
            raw_stages.append("DEEP")
        elif rem_score >= 3 and rem_score > deep_score:
            raw_stages.append("REM")
        else:
            raw_stages.append("LIGHT")

    s = pd.Series(raw_stages, index=df.index)

    # ----- Smoothing step 1: remove single-minute spikes -----
    if SMOOTH_SPIKE and len(s) >= 3:
        s2 = s.copy()
        for i in range(1, len(s) - 1):
            if s.iloc[i] != s.iloc[i - 1] and s.iloc[i] != s.iloc[i + 1]:
                s2.iloc[i] = s.iloc[i - 1]
        s = s2

    # ----- Smoothing step 2: enforce minimum bout duration for DEEP/REM -----
    def enforce_min_bout(series: pd.Series, stage: str, min_len: int) -> pd.Series:
        arr = series.values.copy()
        n = len(arr)
        i = 0
        while i < n:
            if arr[i] != stage:
                i += 1
                continue
            start = i
            while i < n and arr[i] == stage:
                i += 1
            end = i  # [start, end)
            length = end - start
            if length < min_len:
                # convert short bouts to LIGHT
                for j in range(start, end):
                    arr[j] = "LIGHT"
        return pd.Series(arr, index=series.index)

    s = enforce_min_bout(s, "DEEP", DEEP_MIN_BOUT_MIN)
    s = enforce_min_bout(s, "REM", REM_MIN_BOUT_MIN)

    return s


# =========================
# Diagnostics
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


# =========================
# Rolla app (BLE) – CUSTOM V2 ONLY
# =========================
def rolla_app():
    st.title("🛏️ Sleep Analyzer — Custom Stages (V2 only)")

    uploaded = st.file_uploader("Upload Rolla BLE sleep log (.txt)", type=["txt"])
    st.caption("We use 53 only for time grid (start/end). Staging is 100% custom from 55/56/52 (V2).")

    if not uploaded:
        st.info("Upload a BLE log file to analyze sleep.")
    else:
        text = uploaded.read().decode("utf-8", errors="ignore")
        sleep_minutes, hr_samples, hrv_samples, step_samples = parse_log_text(text)

        if not sleep_minutes:
            st.error("No 53 packets (sleep time grid) found in this file.")
            return

        # Attach signals to minute grid
        attach_nearest(hr_samples,   sleep_minutes, "hr",    max_diff_min=HR_ATTACH_TOL_MIN)
        attach_nearest(hrv_samples,  sleep_minutes, "hrv",   max_diff_min=HRV_ATTACH_TOL_MIN)
        attach_nearest(step_samples, sleep_minutes, "steps", max_diff_min=STEPS_ATTACH_TOL_MIN)

        # Sessions (by gaps in the minute grid)
        sessions = split_sessions(sleep_minutes, gap_min=30)
        sessions = sorted(sessions, key=lambda s: s[-1].t)  # latest last for labels; we default to latest

        # Sidebar: choose session (default last)
        session_labels = []
        for i, sess in enumerate(sessions):
            df_tmp = build_dataframe(sess)
            s_start = df_tmp["time"].min()
            s_end = df_tmp["time"].max()
            session_labels.append(
                f"Session {i + 1}  ({s_start.strftime('%Y-%m-%d %H:%M')} – {s_end.strftime('%H:%M')})"
            )

        st.sidebar.header("Session")
        default_index = max(0, len(sessions) - 1)  # default to latest
        selected_idx = st.sidebar.selectbox(
            "Choose sleep session",
            range(len(sessions)),
            index=default_index,
            format_func=lambda i: session_labels[i],
        )
        session = sessions[selected_idx]

        # Build DataFrame
        df_full = build_dataframe(session)

        # ---- V2 STAGING ONLY ----
        df_full["stage"] = compute_custom_stage_v2(df_full)

        # Sidebar counts
        st.sidebar.caption(
            f"HR samples: {len(hr_samples)} | HRV samples: {len(hrv_samples)} | Step samples: {len(step_samples)}"
        )

        # Time filter
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

        # Stage filter
        st.sidebar.header("Stages (custom V2)")
        all_stages = ["AWAKE", "REM", "LIGHT", "DEEP"]
        selected_stages = st.sidebar.multiselect(
            "Show stages",
            options=all_stages,
            default=all_stages,
        )
        df_stage = df_view[df_view["stage"].isin(selected_stages)]

        # Diagnostics
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

        # If no stage rows visible, still show HR graph
        if df_stage.empty:
            st.warning("No data in selected time/stage range.")
            st.markdown("### Heart rate")
            hr_rows = [{"time": s.t, "hr": s.hr} for s in hr_samples if start_t <= s.t <= end_t]
            if hr_rows:
                st.plotly_chart(px.line(pd.DataFrame(hr_rows).sort_values("time"), x="time", y="hr"),
                                use_container_width=True)
            else:
                st.info("No HR data available for this session.")
            return

        # Top metrics
        start = df_stage["time"].min()
        end = df_stage["time"].max() + timedelta(minutes=1)
        total_min = int((end - start).total_seconds() / 60)

        stage_counts = summarize_stages(df_stage)

        c1, c2 = st.columns(2)
        c1.metric("Sleep window (visible)", f"{start.strftime('%H:%M')} – {end.strftime('%H:%M')}")
        c2.metric("Duration (visible)", format_hm(total_min))

        st.markdown("### Stage durations (filtered) — Custom V2")
        cols = st.columns(len(selected_stages))
        for col, stg in zip(cols, selected_stages):
            mins = stage_counts.get(stg, 0)
            col.metric(stg, format_hm(mins))

        st.markdown("### Hypnogram (Custom V2)")
        hypno_fig = build_hypnogram_figure(df_stage[["time", "stage"]])
        if hypno_fig is not None:
            st.plotly_chart(hypno_fig, use_container_width=True)
        else:
            st.info("Not enough data to draw hypnogram.")

        st.markdown("### Heart rate")
        hr_rows = [{"time": s.t, "hr": s.hr} for s in hr_samples if start_t <= s.t <= end_t]
        if hr_rows:
            st.plotly_chart(px.line(pd.DataFrame(hr_rows).sort_values("time"), x="time", y="hr"),
                            use_container_width=True)
        else:
            st.info("No HR data available for this session.")

        with st.expander("Debug table (per minute)"):
            st.dataframe(df_view.reset_index(drop=True))


# =========================
# MAIN
# =========================
def main():
    st.set_page_config(page_title="Sleep Analyzer (V2 only)", layout="wide")

    # Optional routing for privacy/terms:
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
