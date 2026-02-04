import plotly.graph_objects as go
import numpy as np

def render_heatmap(data, title, colorscale='RdBu'):
    """
    Renders an 8x8 heatmap using Plotly.
    """
    # Flip vertically for correct chess board orientation in plotly
    # Chess board is typically (row 0 = rank 1, row 7 = rank 8)
    # Numpy array might have row 0 at the top.
    # In my encoding, rank 0 is row 0. So we want rank 7 (row 7) at the top.
    display_data = np.flipud(data)

    fig = go.Figure(data=go.Heatmap(
        z=display_data,
        colorscale=colorscale,
        zmid=0 if colorscale == 'RdBu' else None,
        showscale=True
    ))

    fig.update_layout(
        title=title,
        xaxis=dict(ticktext=list('abcdefgh'), tickvals=list(range(8))),
        yaxis=dict(ticktext=list('87654321'), tickvals=list(range(8))),
        width=400,
        height=400,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig

def get_latest_heatmap_data(metrics, type_key):
    """
    Extracts the latest heatmap data from metrics list.
    """
    for m in reversed(metrics):
        if m.get('type') == type_key and 'heatmap' in m:
            return np.array(m['heatmap'])
    return None

def get_atlas_heatmap(root):
    heatmap = np.zeros((8, 8))
    for move, child in root.children.items():
        row, col = divmod(move.to_square, 8)
        heatmap[row, col] += child.visit_count
    if heatmap.max() > 0:
        heatmap /= heatmap.max()
    return heatmap

def get_entropy_heatmap(field_tensor):
    # Use the pressure field (channel 2)
    return field_tensor[2].cpu().numpy()
