import streamlit as st
import plotly.graph_objects as go
import json
import os
import time
from pathlib import Path
from typing import List, Dict

st.set_page_config(
    page_title="PROMETHEUS-TQFD Dashboard",
    page_icon="♟️",
    layout="wide"
)

def find_latest_metrics_file(base_dir: Path):
    # This is a simplified version, in reality we'd look into the run_id/metrics/ folder
    metrics_files = list(base_dir.glob("**/metrics/*.jsonl"))
    if not metrics_files:
        return None
    return max(metrics_files, key=os.path.getmtime)

@st.cache_data(ttl=2)
def load_metrics(metrics_file):
    if not metrics_file or not os.path.exists(metrics_file):
        return []

    metrics = []
    with open(metrics_file, 'r') as f:
        for line in f:
            try:
                metrics.append(json.loads(line))
            except:
                pass
    return metrics[-10000:]

def get_elo(name: str, metrics: List[Dict]) -> float:
    for m in reversed(metrics):
        if m.get('type') == 'evaluation' and 'elo' in m:
            return m['elo'].get(name, 1000.0)
    return 1000.0

def get_games(name: str, metrics: List[Dict]) -> int:
    # Estimate from training steps if games not explicit
    count = 0
    for m in metrics:
        if m.get('type') == f'{name}_train':
            count += 1
    return count

def main():
    st.title("♟️ PROMETHEUS-TQFD")
    st.markdown("### Dual-AI Tabula Rasa Chess Training")

    # In a real setup, base_dir comes from environment or config
    base_dir = Path("./prometheus_runs")
    metrics_file = find_latest_metrics_file(base_dir)
    metrics = load_metrics(metrics_file)

    # Sidebar
    with st.sidebar:
        st.markdown("### System Status")
        st.metric("ATLAS ELO", f"{get_elo('atlas', metrics):.0f}")
        st.metric("ENTROPY ELO", f"{get_elo('entropy', metrics):.0f}")
        st.markdown("---")
        st.metric("ATLAS Progress", f"{get_games('atlas', metrics)} steps")
        st.metric("ENTROPY Progress", f"{get_games('entropy', metrics)} steps")

        if st.button("Refresh"):
            st.rerun()

    # Tabs
    tab1, tab2, tab3 = st.tabs(["📈 Lernkurven", "🔥 Heatmaps", "🎮 Live-Spiel"])

    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("ATLAS Training")
            atlas_metrics = [m for m in metrics if m.get('type') == 'atlas_train']
            if atlas_metrics:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    y=[m['loss'] for m in atlas_metrics[-1000:]],
                    name='Total Loss'
                ))
                fig.update_layout(yaxis_type="log", title="Atlas Total Loss")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No ATLAS metrics yet.")

        with col2:
            st.subheader("ENTROPY Training")
            entropy_metrics = [m for m in metrics if m.get('type') == 'entropy_train']
            if entropy_metrics:
                fig = go.Figure()
                for key in ['outcome', 'mobility', 'pressure', 'stability', 'novelty']:
                    if key in entropy_metrics[0]:
                        fig.add_trace(go.Scatter(
                            y=[m.get(key, 0) for m in entropy_metrics[-1000:]],
                            name=key
                        ))
                fig.update_layout(yaxis_type="log", title="Entropy Hybrid Losses")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No ENTROPY metrics yet.")

    with tab2:
        st.info("Heatmaps will be implemented soon.")

    with tab3:
        st.info("Live-game view coming soon.")

if __name__ == '__main__':
    main()
