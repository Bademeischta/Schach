import math
import numpy as np
import torch
import chess
from prometheus_tqfd.encoding import MoveEncoder, BoardEncoder

class MCTSNode:
    def __init__(self, board: chess.Board, parent=None, prior=0.0):
        self.board = board
        self.parent = parent
        self.children = {}
        self.visit_count = 0
        self.total_value = 0.0
        self.prior = prior
        self.is_expanded = False

    @property
    def q_value(self):
        return self.total_value / self.visit_count if self.visit_count > 0 else 0.0

class MCTS:
    def __init__(self, config):
        self.config = config
        self.move_encoder = MoveEncoder()
        self.board_encoder = BoardEncoder()

    def search(self, board: chess.Board, network, device='cpu'):
        root = MCTSNode(board.copy())

        # Initial expansion for root
        self._expand_node(root, network, device)
        self._add_dirichlet_noise(root)

        for _ in range(self.config.atlas_mcts_simulations):
            node = root
            search_path = [node]

            # Selection
            while node.is_expanded and not node.board.is_game_over():
                action, node = self._select_child(node)
                search_path.append(node)

            # Expansion & Evaluation
            value = self._evaluate(node, network, device)

            # Backpropagation
            self._backpropagate(search_path, value)

        return root

    def _select_child(self, node):
        best_score = -float('inf')
        best_action = None
        best_child = None

        total_sqrt_n = math.sqrt(sum(child.visit_count for child in node.children.values()))

        for action, child in node.children.items():
            u_score = self.config.atlas_mcts_cpuct_init * child.prior * total_sqrt_n / (1 + child.visit_count)
            # Standard AlphaZero PUCT
            # c_puct = log((1 + N_parent + c_base) / c_base) + c_init
            # For simplicity using fixed C for now, or can implement the full formula

            score = child.q_value + u_score
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child

    def _expand_node(self, node, network, device):
        if node.board.is_game_over():
            return

        with torch.no_grad():
            board_tensor = self.board_encoder.encode(node.board).unsqueeze(0).to(device)
            policy_logits, _ = network(board_tensor)

            mask = self.move_encoder.get_legal_mask(node.board).to(device)
            probs = torch.softmax(policy_logits[0][mask], dim=0).cpu().numpy()

            legal_moves = list(node.board.legal_moves)
            for move, prob in zip(legal_moves, probs):
                new_board = node.board.copy()
                new_board.push(move)
                node.children[move] = MCTSNode(new_board, parent=node, prior=prob)

        node.is_expanded = True

    def _evaluate(self, node, network, device):
        if node.board.is_game_over():
            res = node.board.result()
            if res == "1-0": return 1.0 if node.board.turn == chess.BLACK else -1.0
            if res == "0-1": return 1.0 if node.board.turn == chess.WHITE else -1.0
            return 0.0

        self._expand_node(node, network, device)
        with torch.no_grad():
            board_tensor = self.board_encoder.encode(node.board).unsqueeze(0).to(device)
            _, value = network(board_tensor)
            return value.item()

    def _backpropagate(self, search_path, value):
        for node in reversed(search_path):
            node.visit_count += 1
            node.total_value += value
            value = -value

    def _add_dirichlet_noise(self, node):
        moves = list(node.children.keys())
        noise = np.random.dirichlet([self.config.atlas_dirichlet_alpha] * len(moves))
        for i, move in enumerate(moves):
            node.children[move].prior = (1 - self.config.atlas_dirichlet_epsilon) * node.children[move].prior + \
                                       self.config.atlas_dirichlet_epsilon * noise[i]

    def select_action(self, root, temperature=1.0):
        moves = list(root.children.keys())
        visit_counts = [child.visit_count for child in root.children.values()]

        if temperature == 0:
            return moves[np.argmax(visit_counts)]

        visit_counts = np.array(visit_counts) ** (1.0 / temperature)
        probs = visit_counts / visit_counts.sum()
        return np.random.choice(moves, p=probs)
