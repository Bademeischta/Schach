import streamlit as st
import plotly.graph_objects as go
import json
import os
import time
import pandas as pd
from pathlib import Path
from typing import List, Dict
import numpy as np
from prometheus_tqfd.dashboard.heatmaps import render_heatmap, get_latest_heatmap_data

st.set_page_config(
    page_title="PROMETHEUS-TQFD Dashboard",
    page_icon="♟️",
    layout="wide"
)

def find_latest_metrics_file(base_dir: Path):
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
    for m in reversed(metrics):
        if m.get('type') == f'{name}_metrics':
            return m.get('games', 0)
    return 0

def get_latest_board_fen(metrics):
    for m in reversed(metrics):
        if 'fen' in m:
            return m['fen']
    return None

def main():
    st.title("♟️ PROMETHEUS-TQFD")
    st.markdown("### Dual-AI Tabula Rasa Chess Training")

    # Path to metrics
    base_dir = Path("./prometheus_runs")
    if os.path.exists("/content"):
        base_dir = Path("/content/prometheus_runs")

    metrics_file = find_latest_metrics_file(base_dir)
    metrics = load_metrics(metrics_file)

    # Sidebar
    with st.sidebar:
        st.markdown("### System Status")
        col_elo1, col_elo2 = st.columns(2)
        col_elo1.metric("ATLAS ELO", f"{get_elo('atlas', metrics):.0f}")
        col_elo2.metric("ENTROPY ELO", f"{get_elo('entropy', metrics):.0f}")

        st.markdown("---")
        st.metric("ATLAS Games", get_games('atlas', metrics))
        st.metric("ENTROPY Games", get_games('entropy', metrics))

        if st.button("Refresh Now"):
            st.rerun()

        st.info(f"Last updated: {time.strftime('%H:%M:%S')}")

    # Tabs
    tab1, tab2, tab3 = st.tabs(["📈 Lernkurven", "🔥 Heatmaps", "🎮 Live-Spiel"])

    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("ATLAS Training")
            atlas_metrics = [m for m in metrics if m.get('type') == 'atlas_metrics']
            if atlas_metrics:
                df = pd.DataFrame(atlas_metrics)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df['step'], y=df['loss'], name='Total Loss'))
                fig.add_trace(go.Scatter(x=df['step'], y=df['policy_loss'], name='Policy Loss'))
                fig.add_trace(go.Scatter(x=df['step'], y=df['value_loss'], name='Value Loss'))
                fig.update_layout(yaxis_type="log", title="Atlas Losses")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No ATLAS metrics yet.")

        with col2:
            st.subheader("ENTROPY Training")
            entropy_metrics = [m for m in metrics if m.get('type') == 'entropy_metrics']
            if entropy_metrics:
                df = pd.DataFrame(entropy_metrics)
                fig = go.Figure()
                for key in ['loss', 'l_entropy', 'l_conservation', 'l_td', 'l_novelty']:
                    if key in df.columns:
                        fig.add_trace(go.Scatter(x=df['step'], y=df[key], name=key))
                fig.update_layout(yaxis_type="log", title="Entropy Hybrid Losses")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No ENTROPY metrics yet.")

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("ATLAS Value Heatmap")
            atlas_hm = get_latest_heatmap_data(metrics, 'atlas_metrics')
            if atlas_hm is not None:
                st.plotly_chart(render_heatmap(atlas_hm, "MCTS Visit Counts", "Viridis"), use_container_width=True)
            else:
                st.info("Waiting for Atlas heatmap data...")

        with col2:
            st.subheader("ENTROPY Energy Field")
            entropy_hm = get_latest_heatmap_data(metrics, 'entropy_metrics')
            if entropy_hm is not None:
                st.plotly_chart(render_heatmap(entropy_hm, "Energy Field", "Magma"), use_container_width=True)
            else:
                st.info("Waiting for Entropy heatmap data...")

    with tab3:
        st.subheader("Live-Spiel Monitoring")
        fen = get_latest_board_fen(metrics)
        if fen:
            st.write(f"**Current FEN:** `{fen}`")
            # Display board using a simple text representation or an image if we had a generator
            # For now, let's use a markdown representation if possible or just the FEN.
            st.code(fen)

            # Show game log
            game_events = [m for m in metrics if 'event' in m and m['event'] == 'move']
            if game_events:
                st.write("Recent Moves:")
                for e in game_events[-5:]:
                    st.write(f"- {e.get('player')}: {e.get('move')}")
        else:
            st.info("No live game data available yet.")

if __name__ == '__main__':
    main()
