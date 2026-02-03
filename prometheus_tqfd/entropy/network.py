import torch
import torch.nn as nn
import torch.nn.functional as F
import chess

try:
    from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool
    from torch_geometric.data import Data, Batch
    HAS_PYG = True
except ImportError:
    HAS_PYG = False

class FieldEncoder(nn.Module):
    def __init__(self, in_channels, out_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=5, padding=2, padding_mode='circular')
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, padding=2, padding_mode='circular')
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, out_dim)

    def forward(self, x):
        x = F.gelu(self.bn1(self.conv1(x)))
        x = F.gelu(self.bn2(self.conv2(x)))
        x = F.gelu(self.bn3(self.conv3(x)))
        x = F.gelu(self.bn4(self.conv4(x)))
        x = self.avgpool(x).view(x.size(0), -1)
        return self.fc(x)

class GNNEncoder(nn.Module):
    def __init__(self, in_features, out_dim):
        super().__init__()
        if not HAS_PYG:
            return
        self.conv1 = GATConv(in_features, 64, heads=4)
        self.conv2 = GATConv(256, 64, heads=4)
        self.conv3 = GATConv(256, 64, heads=4)
        self.conv4 = GATConv(256, 128, heads=4)
        self.fc = nn.Linear(1024, out_dim)

    def forward(self, data):
        if not HAS_PYG:
            return torch.zeros((1, 512)).to(data.x.device)
        x, edge_index = data.x, data.edge_index
        x = F.elu(self.conv1(x, edge_index))
        x = F.elu(self.conv2(x, edge_index))
        x = F.elu(self.conv3(x, edge_index))
        x = F.elu(self.conv4(x, edge_index))

        # Pooling
        x_mean = global_mean_pool(x, data.batch)
        x_max = global_max_pool(x, data.batch)
        x = torch.cat([x_mean, x_max], dim=1) # 1024D
        return self.fc(x)

class QuantumPolicyHead(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.GELU(),
            nn.Linear(256, 4672 * 2)
        )

    def forward(self, x):
        out = self.fc(x)
        real = out[:, :4672]
        imag = out[:, 4672:]
        return real, imag

    def get_probabilities(self, real, imag, legal_mask):
        amplitudes_squared = real**2 + imag**2
        amplitudes_squared = amplitudes_squared * legal_mask.float()

        # Avoid division by zero
        sum_amp = amplitudes_squared.sum(dim=-1, keepdim=True)
        probs = amplitudes_squared / (sum_amp + 1e-8)
        return probs

class EntropyNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.has_pyg = HAS_PYG

        # Encoder A: Field
        self.field_encoder = FieldEncoder(4, 512)

        # Encoder B: GNN or Simplified CNN
        if self.has_pyg:
            self.gnn_encoder = GNNEncoder(10, 512)
        else:
            # Simplified mode: use a standard CNN encoder for the board
            self.simplified_encoder = nn.Sequential(
                nn.Conv2d(19, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(128, 512)
            )

        self.fusion = nn.Sequential(
            nn.Linear(1024, 768),
            nn.LayerNorm(768),
            nn.GELU(),
            nn.Linear(768, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.GELU()
        )

        self.quantum_policy = QuantumPolicyHead(512)
        self.energy_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 1)
        )
        self.flow_field_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 128) # 8x8x2
        )

    def forward(self, field_tensor, board_data, legal_mask):
        # field_tensor: [B, 4, 8, 8]
        # board_data: either PyG Batch or [B, 19, 8, 8] tensor
        # legal_mask: [B, 4672]

        field_out = self.field_encoder(field_tensor)

        if self.has_pyg:
            gnn_out = self.gnn_encoder(board_data)
        else:
            gnn_out = self.simplified_encoder(board_data)

        fused = self.fusion(torch.cat([field_out, gnn_out], dim=1))

        real, imag = self.quantum_policy(fused)
        probs = self.quantum_policy.get_probabilities(real, imag, legal_mask)

        energy = self.energy_head(fused)
        flow = self.flow_field_head(fused).view(-1, 2, 8, 8)

        return probs, energy, flow, (real, imag), fused

def build_graph_data(board: chess.Board):
    if not HAS_PYG:
        return None

    # nodes: each piece
    nodes = []
    piece_map = board.piece_map()
    squares = list(piece_map.keys())

    for sq in squares:
        piece = piece_map[sq]
        rank, file = divmod(sq, 8)
        # Features: [type(6), rank, file, mobility]
        ptype = [0]*6
        ptype[piece.piece_type-1] = 1.0

        # mobility
        mobility = 0
        for move in board.pseudo_legal_moves:
            if move.from_square == sq:
                mobility += 1

        feat = ptype + [rank/7.0, file/7.0, mobility/20.0, 1.0 if piece.color == chess.WHITE else -1.0]
        nodes.append(feat)

    x = torch.tensor(nodes, dtype=torch.float32)

    # edges: if pieces attack each other
    edge_index = []
    for i, sq1 in enumerate(squares):
        for j, sq2 in enumerate(squares):
            if i == j: continue
            if board.is_attacking(sq1, sq2) or board.is_attacking(sq2, sq1):
                edge_index.append([i, j])

    if not edge_index:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    return Data(x=x, edge_index=edge_index)
