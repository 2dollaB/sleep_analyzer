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
    stage: str = "UNKNOWN"   # "device" stage
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
# Parsing log (53 / 55 / 56 / 52)
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

        # 56 – HRV (format nije skroz jasan, koristimo ga kao "index")
        if tokens[0] == "56":
            if len(tokens) >= 10:
                try:
                    dt = parse_device_datetime(tokens, 3)
                    hrv_val = int(tokens[9], 16)
                    hrv_samples.append(HrvSample(t=dt, hrv=hrv_val))
                except Exception:
                    pass

        # 55 – HR: više paketa u jednoj liniji
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
                if 30 <= hr_val <= 220:
                    hr_samples.append(HrSample(t=dt, hr=hr_val))
            except Exception:
                pass
            i += 10  # sljedeći 55 paket

        # 52 – steps: pretpostavimo isti pattern kao 55 (heuristika)
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
# Split into sessions (prekid veći od X minuta = nova sesija)
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
# Device stage mapping (53) – SAMO za usporedbu
# 01 -> deep, 02 -> light, 03 -> REM, ostalo -> awake
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
            "stage": [m.stage for m in session],  # "device" stage
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
    """Očekuje df s kolonom 'stage' = AWAKE/REM/LIGHT/DEEP i 'time'."""
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
# HR graf direktno iz HR sampleova
# =========================

def build_hr_figure_from_samples(samples: List[HrSample], start_t=None, end_t=None):
    """Crta HR graf direktno iz liste HrSample, ne iz df['hr']."""
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

    fig = px.line(
        df_hr,
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
# CUSTOM ALGORITAM v2.0
# =========================

def compute_custom_stage(df: pd.DataFrame) -> pd.Series:
    """
    v3:
      - nema globalnog forsiranja postotaka (nema max DEEP / min REM)
      - samo lokalni signal:
          * HR (razina)
          * dHR (promjena HR)
          * HRV (ako postoji)
          * pozicija u noći (frac)
          * steps za budnost
    """
    df = df.copy()

    # fallback: ako nemamo HR, koristi device stage
    if df["hr"].notna().sum() < 5:
        return df["stage"].fillna("LIGHT")

    # HR smoothing
    hr_s = df["hr"].astype("float").interpolate(limit_direction="both")
    hr_s = hr_s.rolling(5, min_periods=1, center=True).mean()

    # HRV smoothing (ako ima)
    hrv_s = df["hrv"].astype("float")
    if hrv_s.notna().sum() > 5:
        hrv_s = hrv_s.interpolate(limit_direction="both")
        hrv_s = hrv_s.rolling(5, min_periods=1, center=True).mean()
    else:
        hrv_s = None

    # steps (NaN -> 0)
    steps_s = df["steps"].fillna(0)

    # HR derivative (REM i Awake imaju veće oscilacije)
    hr_diff = hr_s.diff().abs().rolling(3, min_periods=1).mean()

    # pragovi (malo konzervativniji)
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
        frac = (t - t_start).total_seconds() / total_sec  # 0.0 početak, 1.0 kraj

        # default je sleep (LIGHT) – prvo provjerimo budnost
        stage = "LIGHT"

        # --- Awake ---
        if steps_val and steps_val > 0:
            stage = "AWAKE"
            stages.append(stage)
            continue

        if hr_val >= hr_very_high and diff_val >= diff_high and (frac < 0.1 or frac > 0.9):
            stage = "AWAKE"
            stages.append(stage)
            continue

        # --- Scoring za DEEP i REM ---
        deep_score = 0
        rem_score = 0

        # Deep: nizak HR, stabilan HR, (po mogućnosti) visok HRV, raniji dio noći
        if hr_val <= hr_low:
            deep_score += 2
        if diff_val <= diff_low:
            deep_score += 1
        if hrv_high is not None and hrv_val is not None and hrv_val >= hrv_high:
            deep_score += 2
        if frac < 0.4:
            deep_score += 1

        # REM: viši HR, veća varijacija, niži HRV, kasnije u noći
        if hr_val >= hr_high:
            rem_score += 2
        if diff_val >= diff_high:
            rem_score += 1
        if hrv_low is not None and hrv_val is not None and hrv_val <= hrv_low:
            rem_score += 2
        if frac > 0.3:
            rem_score += 1

        # odlučivanje – treba veći score i REM/DEEP da bude barem "jači" od LIGHT-a
        if deep_score >= 3 and deep_score > rem_score:
            stage = "DEEP"
        elif rem_score >= 3 and rem_score > deep_score:
            stage = "REM"
        else:
            stage = "LIGHT"

        stages.append(stage)

    s = pd.Series(stages, index=df.index)

    # Smoothing: makni izolirane single-minute skokove
    s_smoothed = s.copy()
    for i in range(1, len(s) - 1):
        if s.iloc[i] != s.iloc[i - 1] and s.iloc[i] != s.iloc[i + 1]:
            s_smoothed.iloc[i] = s.iloc[i - 1]

    return s_smoothed



# =========================
# Streamlit UI
# =========================

def main():
    st.set_page_config(page_title="Sleep Analyzer", layout="wide")
    st.title("🛏️ Sleep Analyzer – Device vs Custom v2.0")

    st.write(
        "Upload raw BLE log (.txt) from your band and compare device staging (cmd 53) "
        "vs our custom HR+HRV algorithm."
    )

    uploaded = st.file_uploader("Upload sleep log (.txt)", type=["txt"])

    if not uploaded:
        st.info("Waiting for file…")
        return

    text = uploaded.read().decode("utf-8", errors="ignore")
    sleep_minutes, hr_samples, hrv_samples, step_samples = parse_log_text(text)

    if not sleep_minutes:
        st.error("No 53 packets (sleep data) found in this file.")
        return

    # zalijepi HR / HRV / steps na minute
    attach_nearest(hr_samples, sleep_minutes, "hr", max_diff_min=10)
    attach_nearest(hrv_samples, sleep_minutes, "hrv", max_diff_min=30)
    attach_nearest(step_samples, sleep_minutes, "steps", max_diff_min=5)

    # device staging (samo za usporedbu)
    apply_device_stage(sleep_minutes)

    # split into sessions (na temelju 53)
    sessions = split_sessions(sleep_minutes, gap_min=30)
    sessions = sorted(sessions, key=lambda s: s[-1].t)

    session_labels = []
    for i, sess in enumerate(sessions):
        df_tmp = build_dataframe(sess)
        s_start = df_tmp["time"].min()
        s_end = df_tmp["time"].max()
        session_labels.append(f"Session {i + 1}  ({s_start.strftime('%Y-%m-%d %H:%M')} – {s_end.strftime('%H:%M')})")

    st.sidebar.header("Session")
    # ✅ default = zadnji session
    default_index = max(0, len(sessions) - 1)
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

    # Stage source toggle
    st.sidebar.header("Stage source")
    stage_source = st.sidebar.radio(
        "Use stages from",
        ["Device (cmd 53)", "Custom v2 (HR+HRV)"],
        index=1,  # default na naš algoritam :)
    )

    # Info o broju uzoraka
    st.sidebar.caption(
        f"HR samples: {len(hr_samples)} | HRV samples: {len(hrv_samples)} | Step samples: {len(step_samples)}"
    )

    # -------- TIME SLIDER (minute) --------
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

    # Koju kolonu koristimo kao "stage"?
    stage_col = "device_stage" if stage_source.startswith("Device") else "custom_stage"
    df_view["stage_used"] = df_view[stage_col]

    # za summary i hypnogram uzimamo samo time + stage_used
    df_stage = df_view[["time", "stage_used"]].rename(columns={"stage_used": "stage"})

    # -------- Stage filter --------
    st.sidebar.header("Stages")
    all_stages = ["AWAKE", "REM", "LIGHT", "DEEP"]
    selected_stages = st.sidebar.multiselect(
        "Show stages",
        options=all_stages,
        default=all_stages,
    )
    df_stage = df_stage[df_stage["stage"].isin(selected_stages)]

    if df_stage.empty:
        st.warning("No data in selected time/stage range.")
        st.markdown("### Heart rate")
        hr_fig = build_hr_figure_from_samples(hr_samples, start_t=start_t, end_t=end_t)
        if hr_fig is not None:
            st.plotly_chart(hr_fig, use_container_width=True)
        else:
            st.info("No HR data available for this session.")
        return

    # summary
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


if __name__ == "__main__":
    main()
