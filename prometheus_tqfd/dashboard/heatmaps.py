import plotly.graph_objects as go
import numpy as np

def render_heatmap(data, title, colorscale='RdBu'):
    fig = go.Figure(data=go.Heatmap(
        z=data,
        colorscale=colorscale,
        zmid=0 if colorscale == 'RdBu' else None,
        showscale=True
    ))
    fig.update_layout(
        title=title,
        xaxis=dict(ticktext=list('abcdefgh'), tickvals=list(range(8))),
        yaxis=dict(ticktext=list('12345678'), tickvals=list(range(8))),
        width=400, height=400
    )
    return fig

def get_atlas_heatmap(mcts_root):
    heatmap = np.zeros((8, 8))
    for move, child in mcts_root.children.items():
        to_square = move.to_square
        row, col = divmod(to_square, 8)
        heatmap[row, col] += child.visit_count
    if heatmap.max() > 0:
        heatmap /= heatmap.max()
    return heatmap

def get_entropy_heatmap(phi_tensor):
    # phi_tensor: [8, 8]
    return phi_tensor.numpy()
