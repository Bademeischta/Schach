import streamlit as st
import pandas as pd
import json
import time
from pathlib import Path
import plotly.express as px
from prometheus_tqfd.dashboard.heatmaps import render_heatmap

st.set_page_config(page_title="PROMETHEUS-TQFD Dashboard", layout="wide")

st.title("🔥 PROMETHEUS-TQFD Chess Training")

# Sidebar
st.sidebar.header("Controls")
if st.sidebar.button("Checkpoint Now"):
    pass # Send signal to supervisor
if st.sidebar.button("Pause/Resume"):
    pass

# Load Metrics
# We should find the latest run_id
def get_latest_run_dir():
    base = Path("/content/prometheus_runs")
    if not base.exists(): return None
    runs = sorted([d for d in base.iterdir() if d.is_dir()], key=lambda x: x.name, reverse=True)
    return runs[0] if runs else None

run_dir = get_latest_run_dir()
metrics_file = run_dir / "metrics" / "metrics.jsonl" if run_dir else None

def load_metrics():
    if not metrics_file or not metrics_file.exists(): return []
    data = []
    with open(metrics_file, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

metrics = load_metrics()

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Learning Curves", "Heatmaps", "Live Game", "Physics Tuning"])

with tab1:
    col1, col2, col3 = st.columns(3)
    # Placeholder for real values
    col1.metric("ATLAS ELO", "1000")
    col2.metric("ENTROPY ELO", "1000")
    col3.metric("Games Played", len([m for m in metrics if m.get('type') == 'duel']))

with tab2:
    st.subheader("Loss Curves")
    atlas_metrics = [m for m in metrics if m.get('type') == 'atlas_metrics']
    if atlas_metrics:
        df_atlas = pd.DataFrame(atlas_metrics)
        st.line_chart(df_atlas.set_index('step')[['loss', 'policy_loss', 'value_loss']])

    entropy_metrics = [m for m in metrics if m.get('type') == 'entropy_metrics']
    if entropy_metrics:
        df_entropy = pd.DataFrame(entropy_metrics)
        st.line_chart(df_entropy.set_index('step')[['loss', 'l_entropy', 'l_conservation', 'l_td']])

with tab5:
    st.subheader("Thermodynamic Constants")
    # This tab would allow adjusting physics via a config file that supervisor watches
    st.slider("Diffusion Sigma (σ)", 0.1, 5.0, 2.5)
    st.slider("Energy King", 100.0, 2000.0, 1000.0)

st.sidebar.text(f"Uptime: {time.time():.0f}")
