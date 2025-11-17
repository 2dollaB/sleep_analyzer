from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, date
from typing import List, Optional, Dict
from requests.auth import HTTPBasicAuth

import pandas as pd
import plotly.express as px
import streamlit as st
import requests
import urllib.parse



# =========================
# Privacy & Terms
# =========================

def render_privacy_policy():
    st.title("Privacy Policy")
    st.write("""
    ### Sleep Analyzer – Privacy Policy

    This project is a development / testing tool used for personal analysis of sleep data.
    We do not store, share or process any user data outside of this application.
    Your Oura API access token is used only to fetch data directly to your browser session
    and is never saved on any external server.

    **What data is processed?**
    - Sleep stages
    - Heart rate data
    - HRV values

    **Where is the data stored?**  
    - Only temporarily inside your browser session.
    - No data is sent to third-party servers.

    **Contact**: minarik.jan@rolla.app  
    """)


def render_terms_of_service():
    st.title("Terms of Service")
    st.write("""
    ### Sleep Analyzer – Terms of Service

    This tool is intended solely for personal experimentation and development.
    By using this application, you agree that:

    - You are providing your own Oura account access voluntarily
    - The authors of this tool are not responsible for incorrect or incomplete data
    - The tool is not a medical device
    - All data remains your property and is not stored outside your usage session

    For any questions, contact: minarik.jan@rolla.app
    """)


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

        # 56 – HRV (heuristički)
        if tokens[0] == "56":
            if len(tokens) >= 10:
                try:
                    dt = parse_device_datetime(tokens, 3)
                    hrv_val = int(tokens[9], 16)
                    hrv_samples.append(HrvSample(t=dt, hrv=hrv_val))
                except Exception:
                    pass

        # 55 – HR: više paketa u jednoj liniji
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

        # 52 – steps (heuristika)
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
# Split into sessions (prekid > X min = nova sesija)
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
# Device stage mapping (53) – samo za usporedbu
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
# HR graf iz HR sampleova
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
# Custom algoritam v3
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

    s_smoothed = s.copy()
    for i in range(1, len(s) - 1):
        if s.iloc[i] != s.iloc[i - 1] and s.iloc[i] != s.iloc[i + 1]:
            s_smoothed.iloc[i] = s.iloc[i - 1]

    return s_smoothed


# =========================
# Rolla app (BLE)
# =========================

def rolla_app():
    st.title("🛏️ Sleep Analyzer – Rolla band")

    uploaded = st.file_uploader("Upload Rolla BLE sleep log (.txt)", type=["txt"])

    if not uploaded:
        st.info("Upload a BLE log file to analyze sleep.")
        return

    text = uploaded.read().decode("utf-8", errors="ignore")
    sleep_minutes, hr_samples, hrv_samples, step_samples = parse_log_text(text)

    if not sleep_minutes:
        st.error("No 53 packets (sleep data) found in this file.")
        return

    attach_nearest(hr_samples, sleep_minutes, "hr", max_diff_min=10)
    attach_nearest(hrv_samples, sleep_minutes, "hrv", max_diff_min=30)
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
# Oura helpers
# =========================

OURA_BASE = "https://api.ouraring.com/v2/usercollection"


def get_oura_config():
    client_id = st.secrets["OURA_CLIENT_ID"]
    client_secret = st.secrets["OURA_CLIENT_SECRET"]
    redirect_uri = st.secrets.get(
        "OURA_REDIRECT_URI",
        "https://sleepanalyzer-kglbkkvkpvkq9unhaatrpb.streamlit.app/oauth",
    )
    return client_id, client_secret, redirect_uri


def fetch_oura_sleep(token: str, day: str):
    """
    Fetch Oura sleep for given calendar day (debug version).
    Shows raw HTTP response if something is weird.
    """
    headers = {"Authorization": f"Bearer {token}"}
    params = {"start_date": day, "end_date": day}

    # ---- 1) /sleep ----
    r = requests.get(f"{OURA_BASE}/sleep", headers=headers, params=params)

    # Debug: pokaži status + raw tekst
    st.write("📡 /sleep status:", r.status_code)
    try:
        st.write("📡 /sleep raw body:", r.text[:2000])  # skraćeno, da ne bude predugačko
    except Exception:
        pass

    if r.status_code != 200:
        st.error(f"Oura /sleep HTTP {r.status_code}: {r.text}")
        return None

    try:
        json_data = r.json()
    except Exception as e:
        st.error(f"Error parsing /sleep JSON: {e}")
        return None

    sleeps = json_data.get("data", [])
    if sleeps:
        # uzmi prvi zapis za taj dan
        return sleeps[0]

    # ---- 2) Ako nema ničega u /sleep, probaj /daily_sleep ----
    r2 = requests.get(f"{OURA_BASE}/daily_sleep", headers=headers, params=params)

    st.write("📡 /daily_sleep status:", r2.status_code)
    try:
        st.write("📡 /daily_sleep raw body:", r2.text[:2000])
    except Exception:
        pass

    if r2.status_code != 200:
        # ako je i ovo prazno – vratit ćemo None i gore će se ispisati upozorenje
        return None

    try:
        json_data2 = r2.json()
    except Exception:
        return None

    daily = json_data2.get("data", [])
    if not daily:
        return None

    # /daily_sleep je već agregat za dan, pa ga samo vratimo
    return daily[0]


def fetch_oura_heartrate(token: str, start_iso: str, end_iso: str):
    headers = {"Authorization": f"Bearer {token}"}
    params = {"start_datetime": start_iso, "end_datetime": end_iso}
    r = requests.get(f"{OURA_BASE}/heartrate", headers=headers, params=params)
    r.raise_for_status()
    data = r.json().get("data", [])
    if not data:
        return pd.DataFrame(columns=["time", "hr"])

    rows = []
    for row in data:
        ts = row.get("timestamp")
        hr = row.get("bpm")
        if ts is None or hr is None:
            continue
        rows.append({"time": pd.to_datetime(ts), "hr": hr})

    df = pd.DataFrame(rows).sort_values("time")
    return df


def summarize_oura_sleep(sleep_json: dict):
    if sleep_json is None:
        return None

    start = pd.to_datetime(sleep_json.get("bedtime_start"))
    end = pd.to_datetime(sleep_json.get("bedtime_end"))

    total_sleep_sec = (
        sleep_json.get("total_sleep_duration")
        or sleep_json.get("duration")
    )
    rem_sec = sleep_json.get("rem_sleep_duration")
    deep_sec = sleep_json.get("deep_sleep_duration")
    light_sec = sleep_json.get("light_sleep_duration")
    awake_sec = sleep_json.get("awake_time")

    summary = {
        "start": start,
        "end": end,
        "total_sleep_min": int(total_sleep_sec / 60) if total_sleep_sec else None,
        "rem_min": int(rem_sec / 60) if rem_sec else None,
        "deep_min": int(deep_sec / 60) if deep_sec else None,
        "light_min": int(light_sec / 60) if light_sec else None,
        "awake_min": int(awake_sec / 60) if awake_sec else None,
    }

    summary["hypnogram_raw"] = (
        sleep_json.get("hypnogram_5min") or sleep_json.get("sleep_phase_5min")
    )

    return summary


# =========================
# Oura app (OAuth)
# =========================

def oura_app():
    st.title("🛏️ Sleep Analyzer – Oura API (OAuth)")

    # 1) Credentials
    try:
        client_id, client_secret, redirect_uri = get_oura_config()
    except Exception:
        st.error("Set OURA_CLIENT_ID, OURA_CLIENT_SECRET and OURA_REDIRECT_URI in Streamlit secrets.")
        st.stop()

    AUTH_URL = "https://cloud.ouraring.com/oauth/authorize"
    TOKEN_URL = "https://api.ouraring.com/oauth/token"

    # 2) Query parametri (code nakon povratka iz Oure)
    #    (možeš kasnije prebaciti na st.query_params, ovo sada radi)
    params = st.experimental_get_query_params()
    code = params.get("code", [None])[0]

    # Uzmemo eventualni token iz session_state (bez KeyError-a)
    token = st.session_state.get("oura_token", None)

    # 3) Ako NEMAMO token → moramo odraditi login / exchange
    if not token:
        # 3a) Imamo code u URL-u → manualni exchange
        if code:
            st.warning("Authorization code detected in URL. Manual exchange required.")

            st.write("**Code:**", code)
            st.write("**Redirect URI (from secrets):**", redirect_uri)
            st.write("**Client ID (from secrets):**", client_id)
            st.write(
                "If this is an old/used code, it will fail. Use 'Connect Oura account' again to generate a new code."
            )

            if st.button("Exchange code for token now"):
                data = {
                    "grant_type": "authorization_code",
                    "code": code,
                    "redirect_uri": redirect_uri,
                }
                try:
                    r = requests.post(
                        TOKEN_URL,
                        data=data,
                        auth=HTTPBasicAuth(client_id, client_secret),
                        headers={"Content-Type": "application/x-www-form-urlencoded"},
                        timeout=15,
                    )
                except Exception as e:
                    st.error(f"Network error calling token endpoint: {e}")
                    st.stop()

                st.write(f"HTTP {r.status_code} — response text:")
                st.code(r.text)

                if r.status_code != 200:
                    st.error("Token exchange failed. See response above.")
                    st.stop()

                try:
                    token_json = r.json()
                except Exception:
                    st.error(f"Cannot parse token JSON: {r.text}")
                    st.stop()

                access_token = token_json.get("access_token")
                if not access_token:
                    st.error(f"Token response missing access_token: {token_json}")
                    st.stop()

                # Spremimo token i očistimo ?code= iz URL-a
                st.session_state["oura_token"] = access_token
                st.experimental_set_query_params()
                st.success("Oura account connected. You can now load sleep data.")
            else:
                st.info("Press the button to try exchanging code for token.")

            # Zaustavi tu – čekamo da user napravi exchange
            st.stop()

        # 3b) NEMAMO ni token ni code → pokaži link za login
        else:
            scope = "email personal daily sleep heartrate"
            # (you can keep "session" too if želiš)
            # scope = "email personal daily sleep session heartrate"
            auth_params = {
                "response_type": "code",
                "client_id": client_id,
                "redirect_uri": redirect_uri,
                "scope": scope,
            }
            auth_url = AUTH_URL + "?" + urllib.parse.urlencode(auth_params)

            st.markdown(
                """
                1. Click the button below (opens in a new tab)  
                2. Log into your Oura account and approve access  
                3. You will be redirected back to this app (same URL)  
                """
            )

            st.markdown(
                f'<a href="{auth_url}" target="_blank"><button>🔗 Connect Oura account</button></a>',
                unsafe_allow_html=True,
            )
            st.stop()

    # 4) Ovdje SIGURNO imamo token u session_state
    token = st.session_state["oura_token"]
    st.info("Oura account connected. Pick a date to inspect sleep.")

    day = st.date_input("Sleep date", value=date.today())

    if st.button("Load Oura sleep"):
        day_str = day.strftime("%Y-%m-%d")

        # Sleep blok
        try:
            sleep_json = fetch_oura_sleep(token, day_str)
        except Exception as e:
            st.error(f"Error fetching sleep: {e}")
            return

        if sleep_json is None:
            st.warning("No Oura sleep data found for that date.")
            return

        summary = summarize_oura_sleep(sleep_json)

        st.subheader("Sleep summary (Oura)")
        st.json(summary)

        # Heart rate tijekom spavanja
        start = summary["start"]
        end = summary["end"]

        start_iso = start.strftime("%Y-%m-%dT%H:%M:%SZ")
        end_iso = end.strftime("%Y-%m-%dT%H:%M:%SZ")

        try:
            df_hr = fetch_oura_heartrate(token, start_iso, end_iso)
        except Exception as e:
            st.error(f"Error fetching heartrate: {e}")
            return

        if not df_hr.empty:
            st.subheader("Heart rate during sleep (Oura)")
            st.line_chart(df_hr.set_index("time")["hr"])
        else:
            st.info("No HR data returned from Oura for that interval.")


# =========================
# MAIN
# =========================

def main():
    st.set_page_config(page_title="Sleep Analyzer", layout="wide")

    # routing za privacy/terms
    params = st.experimental_get_query_params()
    path = params.get("path", [None])[0]
    if path == "privacy":
        render_privacy_policy()
        return
    if path == "terms":
        render_terms_of_service()
        return

    app_mode = st.sidebar.selectbox(
        "Data source",
        ["Rolla band (BLE log)", "Oura API"],
    )

    if app_mode.startswith("Rolla"):
        rolla_app()
    else:
        oura_app()


if __name__ == "__main__":
    main()
