import math
import torch
import torch.nn.functional as F
import numpy as np
import chess
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple
from prometheus_tqfd.config import PrometheusConfig
from prometheus_tqfd.atlas.network import AtlasNetwork
from prometheus_tqfd.encoding import BoardEncoder, MoveEncoder

@dataclass
class MCTSNode:
    state: chess.Board
    parent: Optional['MCTSNode'] = None
    parent_action: Optional[chess.Move] = None
    children: Dict[chess.Move, 'MCTSNode'] = field(default_factory=dict)
    visit_count: int = 0
    total_value: float = 0.0
    prior: float = 0.0

    @property
    def q_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

class MCTS:
    """
    Monte Carlo Tree Search mit PUCT.
    """

    def __init__(self, config: PrometheusConfig, network: AtlasNetwork, device: str):
        self.config = config
        self.network = network
        self.device = device
        self.encoder = BoardEncoder()
        self.move_encoder = MoveEncoder()

    def search(self, root_board: chess.Board, num_simulations: int = None) -> MCTSNode:
        if num_simulations is None:
            num_simulations = self.config.atlas_mcts_simulations

        root = MCTSNode(state=root_board.copy())

        # Initial expansion of root
        self._evaluate_and_expand(root)
        self._add_dirichlet_noise(root)

        for _ in range(num_simulations):
            node = self._select(root)
            value = self._evaluate_and_expand(node)
            self._backpropagate(node, value)

        return root

    def _ucb_score(self, parent: MCTSNode, child: MCTSNode) -> float:
        """PUCT-Formel"""
        c = self.config.atlas_mcts_cpuct
        q = child.q_value
        u = c * child.prior * math.sqrt(parent.visit_count) / (1 + child.visit_count)
        return q + u

    def _select(self, node: MCTSNode) -> MCTSNode:
        """Traversiere bis Blatt mit max UCB"""
        while node.children and not node.state.is_game_over():
            node = max(node.children.values(), key=lambda c: self._ucb_score(node, c))
        return node

    def _evaluate_and_expand(self, node: MCTSNode) -> float:
        """Netzwerk-Inference und Expansion"""
        if node.state.is_game_over():
            result = node.state.result()
            if result == "1-0":
                v = 1.0 if node.state.turn == chess.BLACK else -1.0
            elif result == "0-1":
                v = -1.0 if node.state.turn == chess.BLACK else 1.0
            else:
                v = 0.0
            return v

        # Netzwerk-Inference
        tensor = self.encoder.encode(node.state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            policy_logits, value = self.network(tensor)

        # Legal-Mask und Softmax
        legal_mask = self.move_encoder.get_legal_mask(node.state).to(self.device)
        policy_logits[~legal_mask.unsqueeze(0)] = float('-inf')
        policy = F.softmax(policy_logits, dim=1).squeeze(0)

        # Kinder erstellen
        for move in node.state.legal_moves:
            idx = self.move_encoder.move_to_index(move)
            new_board = node.state.copy()
            new_board.push(move)
            child = MCTSNode(
                state=new_board,
                parent=node,
                parent_action=move,
                prior=policy[idx].item()
            )
            node.children[move] = child

        return value.item()

    def _backpropagate(self, node: MCTSNode, value: float):
        """Value entlang Pfad propagieren mit Vorzeichenwechsel"""
        while node is not None:
            node.visit_count += 1
            node.total_value += value
            value = -value  # Perspektivwechsel
            node = node.parent

    def _add_dirichlet_noise(self, root: MCTSNode):
        """Exploration-Noise am Wurzelknoten"""
        if not root.children:
            return
        alpha = self.config.atlas_dirichlet_alpha
        epsilon = self.config.atlas_dirichlet_epsilon
        noise = np.random.dirichlet([alpha] * len(root.children))
        for i, child in enumerate(root.children.values()):
            child.prior = (1 - epsilon) * child.prior + epsilon * noise[i]

    def get_policy_target(self, root: MCTSNode, temperature: float) -> torch.Tensor:
        """Normalisierte Visit-Counts als Policy-Target"""
        policy = torch.zeros(4672)
        visits = []
        indices = []

        for move, child in root.children.items():
            idx = self.move_encoder.move_to_index(move)
            visits.append(child.visit_count)
            indices.append(idx)

        visits = torch.tensor(visits, dtype=torch.float32)

        if temperature <= 0.01:
            # Greedy
            best_idx = visits.argmax()
            policy[indices[best_idx]] = 1.0
        else:
            # Temperature-Sampling
            probs = (visits ** (1 / temperature))
            probs = probs / probs.sum()
            for i, idx in enumerate(indices):
                policy[idx] = probs[i]

        return policy

    def select_move(self, root: MCTSNode, temperature: float) -> chess.Move:
        """Wähle Zug basierend auf Visit-Counts"""
        moves = list(root.children.keys())
        visits = torch.tensor([root.children[m].visit_count for m in moves], dtype=torch.float32)

        if temperature <= 0.01:
            return moves[visits.argmax()]

        probs = (visits ** (1 / temperature))
        probs = probs / (probs.sum() + 1e-8)
        idx = torch.multinomial(probs, 1).item()
        return moves[idx]
