"""
april_app.py — April Weeks 16–20 | Sensor Data Science Modules
Run: streamlit run april_app.py

Week 16 — Kaggle Sensor Data Cleaning       (sensor_readings_4.csv / sensor_readings_24.csv)
Week 17 — Warehouse Collision Avoidance Sim  (sensor_readings_2.csv)
Week 18 — Dataset Visualization              (real_time_data.csv)
Week 19 — Social Interaction Tracker Sim     (BLE RSSI simulation)
Week 20 — Predictive Maintenance ML          (data.csv)
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay,
                             precision_score, recall_score, f1_score)
from sklearn.preprocessing import StandardScaler
import io, os

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="April — Sensor Data Modules",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.stApp { background: #0d1117; color: #e6edf3; }
.header-card {
    background: linear-gradient(135deg,#0d2137,#0d3b5e);
    border:1px solid #1f4068; border-radius:16px;
    padding:22px 28px; text-align:center; margin-bottom:20px;
}
.header-card h1 { color:#58a6ff; margin:0 0 6px; font-size:1.6rem; }
.header-card p  { color:#8b949e; margin:0; font-size:.88rem; }
.week-badge {
    display:inline-block; background:#1f4068; color:#58a6ff;
    border:1px solid #2d5986; border-radius:20px;
    padding:4px 14px; font-size:.78rem; font-weight:600; margin-bottom:12px;
}
.metric-box {
    background:#161b22; border:1px solid #30363d;
    border-radius:12px; padding:16px; text-align:center;
}
.metric-box .val { font-size:1.8rem; font-weight:700; color:#58a6ff; }
.metric-box .lbl { font-size:.78rem; color:#8b949e; margin-top:4px; }
.info-box {
    background:#161b22; border-left:3px solid #58a6ff;
    border-radius:8px; padding:12px 16px;
    margin:10px 0; font-size:.88rem; color:#c9d1d9;
}
.alert-danger { background:#3d0014;border:1px solid #da3633;border-radius:10px;padding:12px 16px;color:#f78166;font-weight:600; }
.alert-safe   { background:#0d4429;border:1px solid #26a641;border-radius:10px;padding:12px 16px;color:#56d364;font-weight:600; }
.alert-warn   { background:#3d2b00;border:1px solid #bb8009;border-radius:10px;padding:12px 16px;color:#e3b341;font-weight:600; }
section[data-testid="stSidebar"] > div { background:#0d1117 !important; }
#MainMenu, footer, header { visibility:hidden; }
div[data-testid="stDecoration"] { display:none; }
</style>
""", unsafe_allow_html=True)

# ── File paths ─────────────────────────────────────────────────────────────────
BASE  = os.path.dirname(os.path.abspath(__file__))
F_S2  = os.path.join(BASE, "sensor_readings_2.csv")
F_S4  = os.path.join(BASE, "sensor_readings_4.csv")
F_S24 = os.path.join(BASE, "sensor_readings_24.csv")
F_RT  = os.path.join(BASE, "real_time_data.csv")
F_PM  = os.path.join(BASE, "data.csv")

COLS_2  = ['SD_front','SD_left','Class']
COLS_4  = ['SD_front','SD_left','SD_right','SD_back','Class']
COLS_24 = [f'US{i}' for i in range(1,25)] + ['Class']

@st.cache_data
def load_s2():  return pd.read_csv(F_S2,  header=None, names=COLS_2)
@st.cache_data
def load_s4():  return pd.read_csv(F_S4,  header=None, names=COLS_4)
@st.cache_data
def load_s24(): return pd.read_csv(F_S24, header=None, names=COLS_24)
@st.cache_data
def load_rt():  return pd.read_csv(F_RT,  parse_dates=['timestamp'])
@st.cache_data
def load_pm():  return pd.read_csv(F_PM)

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.markdown("## 📅 April — Weeks 16–20")
st.sidebar.markdown("---")
PAGES = {
    "🧹 Week 16 — Sensor Data Cleaning":   "w16",
    "🏭 Week 17 — Collision Avoidance":     "w17",
    "📊 Week 18 — Dataset Visualization":   "w18",
    "📡 Week 19 — Interaction Tracker":     "w19",
    "🤖 Week 20 — Predictive Maintenance":  "w20",
}
page     = st.sidebar.radio("Select Week", list(PAGES.keys()), label_visibility="collapsed")
selected = PAGES[page]
st.sidebar.markdown("---")
st.sidebar.markdown('<div style="color:#484f58;font-size:.74rem;line-height:1.7">📁 Files needed:<br>sensor_readings_2.csv<br>sensor_readings_4.csv<br>sensor_readings_24.csv<br>real_time_data.csv<br>data.csv</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# WEEK 16 — Sensor Data Cleaning
# ═══════════════════════════════════════════════════════════════════════════════
if selected == "w16":
    st.markdown('<div class="header-card"><h1>🧹 Week 16 — Kaggle Sensor Data Cleaning</h1><p>Wall-Following Robot | Ultrasonic Sensors | Missing Values & Outlier Removal</p></div>', unsafe_allow_html=True)
    st.markdown('<div class="week-badge">WEEK 16 · DATA CLEANING</div>', unsafe_allow_html=True)

    ds_choice = st.radio("Select Dataset", ["4-Sensor (Front/Left/Right/Back)", "24-Sensor (Full Ring)"], horizontal=True)
    df_raw    = load_s4() if "4-Sensor" in ds_choice else load_s24()
    s_cols    = [c for c in df_raw.columns if c != 'Class']

    st.markdown("### 📋 Raw Dataset")
    c1,c2,c3,c4 = st.columns(4)
    c1.markdown(f'<div class="metric-box"><div class="val">{len(df_raw):,}</div><div class="lbl">Rows</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-box"><div class="val">{len(s_cols)}</div><div class="lbl">Sensors</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="metric-box"><div class="val">{df_raw[s_cols].isna().sum().sum()}</div><div class="lbl">Missing Values</div></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="metric-box"><div class="val">{df_raw["Class"].nunique()}</div><div class="lbl">Classes</div></div>', unsafe_allow_html=True)
    st.dataframe(df_raw.head(10), use_container_width=True)

    st.markdown("### 🔧 Inject Dirty Data & Clean")
    st.markdown('<div class="info-box">📌 The original dataset has no missing values. We inject them here to demonstrate the full cleaning pipeline as required by Week 16.</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1: miss_pct = st.slider("Missing % to Inject", 1, 20, 5)
    with col2: out_pct  = st.slider("Outlier % to Inject", 1, 10, 2)

    if st.button("🚀 Run Cleaning Pipeline", use_container_width=True):
        rng      = np.random.default_rng(42)
        df_dirty = df_raw.copy()

        for col in s_cols:
            mask = rng.random(len(df_dirty)) < (miss_pct/100)
            df_dirty.loc[mask, col] = np.nan
            mask2 = rng.random(len(df_dirty)) < (out_pct/100)
            df_dirty.loc[mask2, col] = df_dirty[col].max() * 10

        # Clean
        df_clean      = df_dirty.copy()
        null_filled   = 0
        out_fixed     = 0
        for col in s_cols:
            mu = df_clean[col].mean(skipna=True)
            n  = int(df_clean[col].isna().sum())
            df_clean[col].fillna(mu, inplace=True)
            null_filled += n
            std = df_clean[col].std()
            if std > 0:
                z    = (df_clean[col] - df_clean[col].mean()) / std
                omask = z.abs() > 3.0
                out_fixed += int(omask.sum())
                df_clean.loc[omask, col] = df_clean[col].mean()

        st.markdown("### ✅ Results")
        c1,c2,c3,c4 = st.columns(4)
        c1.markdown(f'<div class="metric-box"><div class="val">{null_filled}</div><div class="lbl">Nulls Filled</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="metric-box"><div class="val">{out_fixed}</div><div class="lbl">Outliers Fixed</div></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="metric-box"><div class="val">{df_clean[s_cols].isna().sum().sum()}</div><div class="lbl">Remaining Nulls</div></div>', unsafe_allow_html=True)
        c4.markdown(f'<div class="metric-box"><div class="val">{round((null_filled+out_fixed)/(len(df_raw)*len(s_cols))*100,2)}%</div><div class="lbl">Data Repaired</div></div>', unsafe_allow_html=True)

        tab1, tab2 = st.tabs(["Before Cleaning", "After Cleaning"])
        show_cols  = s_cols[:6]
        with tab1: st.dataframe(df_dirty[show_cols].describe().round(4), use_container_width=True)
        with tab2: st.dataframe(df_clean[show_cols].describe().round(4), use_container_width=True)

        fig, axes = plt.subplots(1,2, figsize=(12,4), facecolor='#161b22')
        for ax, data, title in zip(axes,
            [df_dirty[show_cols[0]], df_clean[show_cols[0]]],
            ['Before Cleaning', 'After Cleaning']):
            ax.hist(data.dropna(), bins=40, color='#58a6ff', edgecolor='#30363d', alpha=.85)
            ax.set_facecolor('#0d1117'); ax.set_title(title, color='#e6edf3')
            ax.set_xlabel(show_cols[0], color='#8b949e')
            ax.tick_params(colors='#8b949e')
            for sp in ax.spines.values(): sp.set_edgecolor('#30363d')
        plt.tight_layout(); st.pyplot(fig); plt.close()

        # Class distribution
        st.markdown("### 🏷️ Robot Navigation Class Distribution")
        cc = df_raw['Class'].value_counts()
        fig2,ax2 = plt.subplots(figsize=(8,3), facecolor='#161b22')
        ax2.barh(cc.index, cc.values, color=['#58a6ff','#3fb950','#f78166','#e3b341'])
        ax2.set_facecolor('#0d1117'); ax2.set_title('Navigation Commands', color='#e6edf3')
        ax2.tick_params(colors='#8b949e')
        for sp in ax2.spines.values(): sp.set_edgecolor('#30363d')
        for i,v in enumerate(cc.values): ax2.text(v+10, i, str(v), color='#e6edf3', va='center', fontsize=9)
        plt.tight_layout(); st.pyplot(fig2); plt.close()

        buf = io.StringIO(); df_clean.to_csv(buf, index=False)
        st.download_button("📥 Download Cleaned Data", buf.getvalue(), "cleaned_sensor.csv", "text/csv", use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# WEEK 17 — Collision Avoidance
# ═══════════════════════════════════════════════════════════════════════════════
elif selected == "w17":
    st.markdown('<div class="header-card"><h1>🏭 Week 17 — Warehouse Collision Avoidance</h1><p>Laser Proximity Geofence | AGV Speed Control | Incident Logger</p></div>', unsafe_allow_html=True)
    st.markdown('<div class="week-badge">WEEK 17 · COLLISION AVOIDANCE</div>', unsafe_allow_html=True)

    df = load_s2()
    st.markdown("### 📋 Front & Left Proximity Sensor Data")
    st.markdown('<div class="info-box">📌 Using sensor_readings_2.csv — front and left ultrasonic distances (meters) from SCITOS-G5 robot. Simulates an AGV geofence system: if object detected within danger zone, motor stops and incident is logged.</div>', unsafe_allow_html=True)

    c1,c2,c3 = st.columns(3)
    c1.markdown(f'<div class="metric-box"><div class="val">{len(df):,}</div><div class="lbl">Readings</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-box"><div class="val">{df["SD_front"].min():.3f}m</div><div class="lbl">Min Front Dist</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="metric-box"><div class="val">{df["SD_front"].mean():.3f}m</div><div class="lbl">Avg Front Dist</div></div>', unsafe_allow_html=True)
    st.dataframe(df.head(8), use_container_width=True)

    st.markdown("### ⚙️ Geofence Configuration")
    col1,col2,col3 = st.columns(3)
    with col1: danger  = st.slider("🔴 Danger Zone — STOP (m)",  0.3, 1.5, 0.6, 0.05)
    with col2: caution = st.slider("🟡 Caution Zone — SLOW (m)", 0.5, 2.5, 1.2, 0.05)
    with col3: speed   = st.slider("AGV Normal Speed (m/s)",     0.5, 3.0, 1.5, 0.1)

    def agv_decide(front, left, d, c, s):
        dist = min(front, left)
        if   dist <= d: return "STOP",         0.0,                  "🔴"
        elif dist <= c: return "SLOW DOWN",    round(s*(dist/c), 2), "🟡"
        else:           return "MOVE FORWARD", s,                    "🟢"

    if st.button("▶️ Run on All Sensor Readings", use_container_width=True):
        rows = []
        for _, r in df.iterrows():
            action, spd, icon = agv_decide(r['SD_front'], r['SD_left'], danger, caution, speed)
            rows.append({'SD_front': round(r['SD_front'],3), 'SD_left': round(r['SD_left'],3),
                         'Class': r['Class'], 'AGV_Action': action, 'AGV_Speed': spd})
        df_res = pd.DataFrame(rows)
        counts = df_res['AGV_Action'].value_counts()

        st.markdown("### 📊 AGV Decision Summary")
        icons_m = {"STOP":"🔴","SLOW DOWN":"🟡","MOVE FORWARD":"🟢"}
        cols = st.columns(len(counts))
        for i,(act,cnt) in enumerate(counts.items()):
            cols[i].markdown(f'<div class="metric-box"><div class="val">{icons_m.get(act,"")} {cnt:,}</div><div class="lbl">{act} ({cnt/len(df_res)*100:.1f}%)</div></div>', unsafe_allow_html=True)

        st.markdown("### 🚨 Incident Log (STOP Events)")
        incidents = df_res[df_res['AGV_Action']=='STOP'].head(20)
        if not incidents.empty:
            st.dataframe(incidents, use_container_width=True)
            st.markdown(f'<div class="alert-danger">⚠️ {len(df_res[df_res["AGV_Action"]=="STOP"])} STOP events detected — AGV halted to prevent collision!</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="alert-safe">✅ No STOP events with current thresholds.</div>', unsafe_allow_html=True)

        sample = df_res.head(300)
        fig, ax = plt.subplots(figsize=(12,4), facecolor='#161b22')
        ax.plot(sample['SD_front'].values, color='#58a6ff', linewidth=0.8, label='Front Distance')
        ax.axhline(y=danger,  color='#f78166', linestyle='--', linewidth=1.2, label=f'Danger ({danger}m)')
        ax.axhline(y=caution, color='#e3b341', linestyle='--', linewidth=1.2, label=f'Caution ({caution}m)')
        ax.fill_between(range(len(sample)), 0, danger,  alpha=.15, color='#f78166')
        ax.fill_between(range(len(sample)), danger, caution, alpha=.08, color='#e3b341')
        ax.set_facecolor('#0d1117'); ax.set_title('Front Proximity — Geofence Zones (300 readings)', color='#e6edf3')
        ax.set_xlabel('Reading #', color='#8b949e'); ax.set_ylabel('Distance (m)', color='#8b949e')
        ax.tick_params(colors='#8b949e'); ax.legend(facecolor='#161b22', labelcolor='#e6edf3', fontsize=9)
        for sp in ax.spines.values(): sp.set_edgecolor('#30363d')
        plt.tight_layout(); st.pyplot(fig); plt.close()

        buf = io.StringIO()
        df_res[df_res['AGV_Action']!='MOVE FORWARD'].to_csv(buf, index=False)
        st.download_button("📥 Download Incident Log", buf.getvalue(), "incident_log.csv", "text/csv", use_container_width=True)

    # Live test
    st.markdown("---")
    st.markdown("### 🔴 Live Sensor Test")
    c1,c2 = st.columns(2)
    with c1: sf = st.slider("Front Sensor (m)", 0.3, 5.0, 1.5, 0.05)
    with c2: sl = st.slider("Left Sensor (m)",  0.3, 5.0, 1.5, 0.05)
    action, spd, icon = agv_decide(sf, sl, danger, caution, speed)
    css = "alert-danger" if action=="STOP" else "alert-warn" if action=="SLOW DOWN" else "alert-safe"
    st.markdown(f'<div class="{css}">{icon} AGV: <strong>{action}</strong> &nbsp;|&nbsp; Speed: <strong>{spd} m/s</strong> &nbsp;|&nbsp; Min dist: <strong>{min(sf,sl):.2f}m</strong></div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# WEEK 18 — Dataset Visualization
# ═══════════════════════════════════════════════════════════════════════════════
elif selected == "w18":
    st.markdown('<div class="header-card"><h1>📊 Week 18 — Kaggle Dataset Visualization</h1><p>IoT Real-Time Sensor Data | 1,000 Readings | Time-Series Analysis</p></div>', unsafe_allow_html=True)
    st.markdown('<div class="week-badge">WEEK 18 · VISUALIZATION</div>', unsafe_allow_html=True)

    df = load_rt()
    num_cols = ['temperature','humidity','pressure','light','sound','battery']

    st.markdown("### 📋 Dataset Overview")
    c1,c2,c3,c4 = st.columns(4)
    c1.markdown(f'<div class="metric-box"><div class="val">{len(df):,}</div><div class="lbl">Readings</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-box"><div class="val">{df["device_id"].nunique()}</div><div class="lbl">Devices</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="metric-box"><div class="val">{df["location"].nunique()}</div><div class="lbl">Locations</div></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="metric-box"><div class="val">{df[num_cols].isna().sum().sum()}</div><div class="lbl">Missing Values</div></div>', unsafe_allow_html=True)
    st.dataframe(df.head(10), use_container_width=True)

    st.markdown("### 📐 Sensor Statistics")
    st.dataframe(df[num_cols].describe().round(3), use_container_width=True)

    # Time-series
    st.markdown("### 📈 Time-Series Plot")
    col1, col2 = st.columns([2,1])
    with col1: sensor_sel = st.selectbox("Sensor", num_cols)
    with col2: loc_sel = st.multiselect("Locations", df['location'].unique().tolist(), default=df['location'].unique().tolist())

    df_f = df[df['location'].isin(loc_sel)].sort_values('timestamp') if loc_sel else df.sort_values('timestamp')
    fig, ax = plt.subplots(figsize=(13,4), facecolor='#161b22')
    pal = ['#58a6ff','#3fb950','#f78166','#e3b341','#bc8cff']
    for i,loc in enumerate(df_f['location'].unique()):
        sub = df_f[df_f['location']==loc]
        ax.plot(sub['timestamp'], sub[sensor_sel], label=loc, color=pal[i%len(pal)], linewidth=0.9, alpha=.85)
    ax.set_facecolor('#0d1117'); ax.set_title(f'{sensor_sel.title()} Over Time by Location', color='#e6edf3', fontsize=12)
    ax.set_xlabel('Timestamp', color='#8b949e'); ax.set_ylabel(sensor_sel.title(), color='#8b949e')
    ax.tick_params(colors='#8b949e', labelsize=7); ax.legend(facecolor='#161b22', labelcolor='#e6edf3', fontsize=9)
    for sp in ax.spines.values(): sp.set_edgecolor('#30363d')
    plt.tight_layout(); st.pyplot(fig); plt.close()

    # Averages per location
    st.markdown("### 📏 Average Sensor Values by Location")
    avg = df.groupby('location')[num_cols].mean().round(3)
    st.dataframe(avg, use_container_width=True)

    fig2, axes = plt.subplots(1, len(num_cols), figsize=(16,4), facecolor='#161b22')
    for ax, col in zip(axes, num_cols):
        means = avg[col]
        ax.bar(means.index, means.values, color=pal[:len(means)], alpha=.85)
        ax.set_facecolor('#0d1117'); ax.set_title(col.title(), color='#e6edf3', fontsize=9)
        ax.tick_params(colors='#8b949e', labelsize=7)
        for sp in ax.spines.values(): sp.set_edgecolor('#30363d')
    plt.suptitle('Average Sensor Values by Location', color='#e6edf3', fontsize=11, y=1.01)
    plt.tight_layout(); st.pyplot(fig2); plt.close()

    # Correlation heatmap
    st.markdown("### 🔥 Correlation Heatmap")
    corr = df[num_cols].corr()
    fig3, ax3 = plt.subplots(figsize=(8,6), facecolor='#161b22')
    im = ax3.imshow(corr, cmap='RdYlBu', vmin=-1, vmax=1)
    ax3.set_xticks(range(len(num_cols))); ax3.set_yticks(range(len(num_cols)))
    ax3.set_xticklabels(num_cols, rotation=45, ha='right', color='#e6edf3', fontsize=9)
    ax3.set_yticklabels(num_cols, color='#e6edf3', fontsize=9)
    ax3.set_facecolor('#0d1117'); ax3.set_title('Sensor Correlation Matrix', color='#e6edf3', fontsize=12, pad=12)
    plt.colorbar(im, ax=ax3)
    for i in range(len(num_cols)):
        for j in range(len(num_cols)):
            ax3.text(j, i, f'{corr.iloc[i,j]:.2f}', ha='center', va='center', color='#0d1117', fontsize=8, fontweight='bold')
    plt.tight_layout(); st.pyplot(fig3); plt.close()

    # Motion summary
    st.markdown("### 🏃 Motion Detection by Location")
    motion = df.groupby(['location','motion']).size().unstack(fill_value=0)
    motion.columns = ['No Motion','Motion Detected']
    st.dataframe(motion, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# WEEK 19 — Social Interaction Tracker
# ═══════════════════════════════════════════════════════════════════════════════
elif selected == "w19":
    st.markdown('<div class="header-card"><h1>📡 Week 19 — Social Interaction Tracker</h1><p>BLE Proximity Sensors | RSSI Signal Strength | Interaction Classification & Storage</p></div>', unsafe_allow_html=True)
    st.markdown('<div class="week-badge">WEEK 19 · BLE INTERACTION SIMULATION</div>', unsafe_allow_html=True)

    st.markdown('<div class="info-box">📌 BLE wearables detect nearby devices using RSSI (Received Signal Strength Indicator). Stronger RSSI = closer device. Values: -30 dBm (very close) → -100 dBm (out of range). Interaction strength is classified and stored in an array.</div>', unsafe_allow_html=True)

    st.markdown("### ⚙️ Configuration")
    col1,col2,col3 = st.columns(3)
    with col1: n_devices  = st.slider("Devices (People)", 3, 15, 8)
    with col2: duration   = st.slider("Duration (seconds)", 10, 60, 20)
    with col3: interval   = st.slider("Scan Interval (s)", 1, 5, 2)

    st.markdown("### 📶 RSSI Thresholds")
    c1,c2 = st.columns(2)
    with c1: strong_t = st.number_input("Strong Interaction (dBm)", -90, -30, -60)
    with c2: medium_t = st.number_input("Medium Interaction (dBm)", -100, -50, -75)

    def classify(rssi):
        if rssi >= strong_t:  return "Strong", "🔴"
        elif rssi >= medium_t: return "Medium", "🟡"
        elif rssi >= -90:     return "Weak",   "🟢"
        else:                 return "None",   "⚫"

    if st.button("▶️ Run Interaction Simulation", use_container_width=True):
        devices = [f"Device_{i+1:02d}" for i in range(n_devices)]
        rng     = np.random.default_rng(42)
        rows    = []
        t = 0
        while t < duration:
            for i in range(n_devices):
                for j in range(i+1, n_devices):
                    rssi = int(rng.integers(-100, -30))
                    strength, icon = classify(rssi)
                    if strength != "None":
                        rows.append({'Time(s)': t, 'Device A': devices[i],
                                     'Device B': devices[j], 'RSSI(dBm)': rssi,
                                     'Strength': strength, 'Icon': icon})
            t += interval

        df_int = pd.DataFrame(rows)
        if df_int.empty:
            st.warning("No interactions detected. Adjust thresholds.")
        else:
            counts = df_int['Strength'].value_counts()
            st.markdown("### 📊 Summary")
            cols = st.columns(len(counts)+1)
            cols[0].markdown(f'<div class="metric-box"><div class="val">{len(df_int)}</div><div class="lbl">Total Interactions</div></div>', unsafe_allow_html=True)
            im = {"Strong":"🔴","Medium":"🟡","Weak":"🟢"}
            for i,(s,c) in enumerate(counts.items()):
                cols[i+1].markdown(f'<div class="metric-box"><div class="val">{im.get(s,"")} {c}</div><div class="lbl">{s}</div></div>', unsafe_allow_html=True)

            st.markdown("### 📋 Interaction Log (stored in array)")
            st.dataframe(df_int[['Time(s)','Device A','Device B','RSSI(dBm)','Strength']],
                         use_container_width=True, height=280)

            # RSSI distribution
            fig,ax = plt.subplots(figsize=(10,4), facecolor='#161b22')
            cs = {"Strong":"#f78166","Medium":"#e3b341","Weak":"#3fb950"}
            for s,grp in df_int.groupby('Strength'):
                ax.hist(grp['RSSI(dBm)'], bins=20, alpha=.75, label=s, color=cs.get(s,'#58a6ff'))
            ax.axvline(x=strong_t, color='#f78166', linestyle='--', linewidth=1.2, label=f'Strong ({strong_t} dBm)')
            ax.axvline(x=medium_t, color='#e3b341', linestyle='--', linewidth=1.2, label=f'Medium ({medium_t} dBm)')
            ax.set_facecolor('#0d1117'); ax.set_title('RSSI Distribution by Strength', color='#e6edf3', fontsize=12)
            ax.set_xlabel('RSSI (dBm)', color='#8b949e'); ax.set_ylabel('Count', color='#8b949e')
            ax.tick_params(colors='#8b949e'); ax.legend(facecolor='#161b22', labelcolor='#e6edf3', fontsize=8)
            for sp in ax.spines.values(): sp.set_edgecolor('#30363d')
            plt.tight_layout(); st.pyplot(fig); plt.close()

            # Per-device count
            st.markdown("### 👥 Most Active Devices")
            all_d = pd.concat([df_int['Device A'].rename('Device'), df_int['Device B'].rename('Device')])
            dc    = all_d.value_counts().reset_index(); dc.columns = ['Device','Interactions']
            fig2,ax2 = plt.subplots(figsize=(10,4), facecolor='#161b22')
            ax2.bar(dc['Device'], dc['Interactions'], color='#58a6ff', alpha=.85)
            ax2.set_facecolor('#0d1117'); ax2.set_title('Interactions per Device', color='#e6edf3', fontsize=12)
            ax2.set_xlabel('Device', color='#8b949e'); ax2.set_ylabel('Count', color='#8b949e')
            ax2.tick_params(colors='#8b949e', labelsize=8)
            plt.xticks(rotation=45, ha='right')
            for sp in ax2.spines.values(): sp.set_edgecolor('#30363d')
            plt.tight_layout(); st.pyplot(fig2); plt.close()

            buf = io.StringIO(); df_int.to_csv(buf, index=False)
            st.download_button("📥 Download Interaction Log", buf.getvalue(), "interaction_log.csv", "text/csv", use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# WEEK 20 — Predictive Maintenance
# ═══════════════════════════════════════════════════════════════════════════════
elif selected == "w20":
    st.markdown('<div class="header-card"><h1>🤖 Week 20 — Kaggle Predictive Maintenance</h1><p>Machine Failure Prediction | Feature Engineering | Logistic Regression</p></div>', unsafe_allow_html=True)
    st.markdown('<div class="week-badge">WEEK 20 · PREDICTIVE MAINTENANCE ML</div>', unsafe_allow_html=True)

    df = load_pm()
    feat_cols = [c for c in df.columns if c != 'fail']

    st.markdown("### 📋 Dataset Overview")
    c1,c2,c3,c4 = st.columns(4)
    c1.markdown(f'<div class="metric-box"><div class="val">{len(df):,}</div><div class="lbl">Samples</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-box"><div class="val">{len(feat_cols)}</div><div class="lbl">Features</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="metric-box"><div class="val">{int(df["fail"].sum())}</div><div class="lbl">Failures</div></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="metric-box"><div class="val">{round(df["fail"].mean()*100,1)}%</div><div class="lbl">Failure Rate</div></div>', unsafe_allow_html=True)

    tab1,tab2 = st.tabs(["Raw Data","Statistics"])
    with tab1: st.dataframe(df.head(15), use_container_width=True)
    with tab2: st.dataframe(df.describe().round(3), use_container_width=True)

    # Feature engineering
    st.markdown("### 🔧 Feature Engineering")
    st.markdown('<div class="info-box">📌 Engineering new features from raw sensor proximity readings to improve model performance.</div>', unsafe_allow_html=True)

    df_f = df.copy()
    df_f['USS_AQ_ratio'] = df_f['USS'] / (df_f['AQ'] + 1e-5)
    df_f['sensor_mean']  = df_f[['AQ','USS','CS','VOC','RP','IP']].mean(axis=1)
    df_f['sensor_std']   = df_f[['AQ','USS','CS','VOC','RP','IP']].std(axis=1)
    df_f['temp_risk']    = (df_f['Temperature'] > df_f['Temperature'].quantile(0.75)).astype(int)
    new_f = ['USS_AQ_ratio','sensor_mean','sensor_std','temp_risk']
    all_f = feat_cols + new_f

    st.success(f"Original features: {len(feat_cols)} → After engineering: {len(all_f)}")
    st.dataframe(df_f[new_f].head(8), use_container_width=True)

    # Model config
    st.markdown("### 🤖 Model Configuration")
    col1,col2 = st.columns(2)
    with col1: test_pct = st.slider("Test Split %", 10, 40, 20)
    with col2: use_eng  = st.checkbox("Use Engineered Features", value=True)

    features = all_f if use_eng else feat_cols
    X = df_f[features]; y = df_f['fail']

    if st.button("🚀 Train Model & Evaluate", use_container_width=True):
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_pct/100, random_state=42, stratify=y)
        scaler   = StandardScaler()
        X_tr_s   = scaler.fit_transform(X_tr)
        X_te_s   = scaler.transform(X_te)
        model    = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_tr_s, y_tr)
        y_pred   = model.predict(X_te_s)
        y_prob   = model.predict_proba(X_te_s)[:,1]

        st.markdown("### ✅ Model Performance")
        c1,c2,c3,c4 = st.columns(4)
        c1.markdown(f'<div class="metric-box"><div class="val">{accuracy_score(y_te,y_pred)*100:.1f}%</div><div class="lbl">Accuracy</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="metric-box"><div class="val">{precision_score(y_te,y_pred)*100:.1f}%</div><div class="lbl">Precision</div></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="metric-box"><div class="val">{recall_score(y_te,y_pred)*100:.1f}%</div><div class="lbl">Recall</div></div>', unsafe_allow_html=True)
        c4.markdown(f'<div class="metric-box"><div class="val">{f1_score(y_te,y_pred)*100:.1f}%</div><div class="lbl">F1-Score</div></div>', unsafe_allow_html=True)

        st.markdown("### 📋 Classification Report")
        rpt = classification_report(y_te, y_pred, target_names=['No Failure','Failure'], output_dict=True)
        st.dataframe(pd.DataFrame(rpt).transpose().round(3), use_container_width=True)

        cola, colb = st.columns(2)

        with cola:
            st.markdown("#### Confusion Matrix")
            fig,ax = plt.subplots(figsize=(5,4), facecolor='#161b22')
            ConfusionMatrixDisplay(confusion_matrix(y_te, y_pred),
                                   display_labels=['No Failure','Failure']).plot(ax=ax, colorbar=False, cmap='Blues')
            ax.set_facecolor('#0d1117'); ax.set_title('Confusion Matrix', color='#e6edf3', fontsize=10)
            ax.tick_params(colors='#8b949e', labelsize=8)
            ax.xaxis.label.set_color('#8b949e'); ax.yaxis.label.set_color('#8b949e')
            for sp in ax.spines.values(): sp.set_edgecolor('#30363d')
            plt.tight_layout(); st.pyplot(fig); plt.close()

        with colb:
            st.markdown("#### Feature Importance")
            coef_df = pd.DataFrame({'Feature': features, 'Importance': np.abs(model.coef_[0])})
            coef_df = coef_df.sort_values('Importance', ascending=True).tail(10)
            fig2,ax2 = plt.subplots(figsize=(5,4), facecolor='#161b22')
            ax2.barh(coef_df['Feature'], coef_df['Importance'], color='#58a6ff', alpha=.85)
            ax2.set_facecolor('#0d1117'); ax2.set_title('Top Feature Importances', color='#e6edf3', fontsize=10)
            ax2.tick_params(colors='#8b949e', labelsize=8)
            for sp in ax2.spines.values(): sp.set_edgecolor('#30363d')
            plt.tight_layout(); st.pyplot(fig2); plt.close()

        # Probability distribution
        st.markdown("#### Failure Probability Distribution")
        fig3,ax3 = plt.subplots(figsize=(10,3), facecolor='#161b22')
        ax3.hist(y_prob[y_te==0], bins=30, alpha=.7, color='#3fb950', label='No Failure (actual)')
        ax3.hist(y_prob[y_te==1], bins=30, alpha=.7, color='#f78166', label='Failure (actual)')
        ax3.axvline(x=0.5, color='#e3b341', linestyle='--', label='Decision boundary (0.5)')
        ax3.set_facecolor('#0d1117'); ax3.set_title('Predicted Failure Probability', color='#e6edf3', fontsize=11)
        ax3.set_xlabel('Probability', color='#8b949e'); ax3.set_ylabel('Count', color='#8b949e')
        ax3.tick_params(colors='#8b949e'); ax3.legend(facecolor='#161b22', labelcolor='#e6edf3', fontsize=9)
        for sp in ax3.spines.values(): sp.set_edgecolor('#30363d')
        plt.tight_layout(); st.pyplot(fig3); plt.close()

        # Live prediction
        st.markdown("### 🔮 Live Failure Prediction")
        st.markdown('<div class="info-box">Enter sensor readings to predict machine failure.</div>', unsafe_allow_html=True)
        input_cols = st.columns(len(feat_cols))
        user_vals  = {}
        for i, col in enumerate(feat_cols):
            mn = float(df[col].min()); mx = float(df[col].max()); med = float(df[col].median())
            user_vals[col] = input_cols[i].number_input(col, min_value=mn, max_value=mx, value=med)

        if st.button("🔮 Predict", use_container_width=True):
            inp = pd.DataFrame([user_vals])
            if use_eng:
                inp['USS_AQ_ratio'] = inp['USS']/(inp['AQ']+1e-5)
                inp['sensor_mean']  = inp[['AQ','USS','CS','VOC','RP','IP']].mean(axis=1)
                inp['sensor_std']   = inp[['AQ','USS','CS','VOC','RP','IP']].std(axis=1)
                inp['temp_risk']    = (inp['Temperature'] > df['Temperature'].quantile(0.75)).astype(int)
            inp_s = scaler.transform(inp[features])
            pred  = model.predict(inp_s)[0]
            prob  = model.predict_proba(inp_s)[0][1]
            if pred == 1:
                st.markdown(f'<div class="alert-danger">🚨 FAILURE PREDICTED — Probability: {prob*100:.1f}%<br>Schedule maintenance immediately!</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="alert-safe">✅ NO FAILURE — Machine is healthy. Failure probability: {prob*100:.1f}%</div>', unsafe_allow_html=True)
