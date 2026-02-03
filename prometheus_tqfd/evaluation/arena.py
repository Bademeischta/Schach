import chess
import time
import torch
import numpy as np
from prometheus_tqfd.evaluation.baselines import RandomPlayer, HeuristicPlayer
from prometheus_tqfd.encoding import BoardEncoder, MoveEncoder
from prometheus_tqfd.entropy.fields import FieldCalculator

class Arena:
    def __init__(self, config, metrics_queue, shared_values):
        self.config = config
        self.metrics_queue = metrics_queue
        self.shared_values = shared_values

        self.board_encoder = BoardEncoder()
        self.move_encoder = MoveEncoder()
        self.field_calc = FieldCalculator(config.physics)

    def play_game(self, player_white, player_black):
        board = chess.Board()
        while not board.is_game_over():
            if board.turn == chess.WHITE:
                move = player_white.select_move(board)
            else:
                move = player_black.select_move(board)
            board.push(move)
        return board.result()

    def update_elo(self, results, k_factor=32):
        # results: {'atlas_wins': x, 'entropy_wins': y, 'draws': z}
        r_a = self.shared_values['atlas_elo']
        r_b = self.shared_values['entropy_elo']

        e_a = 1 / (1 + 10 ** ((r_b - r_a) / 400))

        total = results['atlas_wins'] + results['entropy_wins'] + results['draws']
        if total == 0: return

        s_a = (results['atlas_wins'] + 0.5 * results['draws']) / total

        self.shared_values['atlas_elo'] = r_a + k_factor * (s_a - e_a)
        self.shared_values['entropy_elo'] = r_b + k_factor * ((1 - s_a) - (1 - e_a))

class NNPlayer:
    def __init__(self, model, config, move_encoder, board_encoder, field_calc=None, is_mcts=False):
        self.model = model
        self.config = config
        self.move_encoder = move_encoder
        self.board_encoder = board_encoder
        self.field_calc = field_calc
        self.is_mcts = is_mcts

        if is_mcts:
            from prometheus_tqfd.atlas.mcts import MCTS
            self.mcts = MCTS(config)

    def select_move(self, board):
        self.model.eval()
        with torch.no_grad():
            if self.is_mcts:
                root = self.mcts.search(board, self.model, self.config.actor_device)
                return self.mcts.select_action(root, temperature=0)
            else:
                field_tensor = self.field_calc.compute_fields(board).unsqueeze(0).to(self.config.actor_device)
                if hasattr(self.model, 'has_pyg') and self.model.has_pyg:
                    from prometheus_tqfd.entropy.network import build_graph_data
                    board_data = build_graph_data(board).to(self.config.actor_device)
                else:
                    board_data = self.board_encoder.encode(board).unsqueeze(0).to(self.config.actor_device)

                mask = self.move_encoder.get_legal_mask(board).unsqueeze(0).to(self.config.actor_device)
                probs, _, _, _, _ = self.model(field_tensor, board_data, mask)

                probs = probs[0].cpu().numpy()
                legal_moves = list(board.legal_moves)
                legal_indices = [self.move_encoder.move_to_index(m) for m in legal_moves]

                p_legal = probs[legal_indices]
                move_idx = np.argmax(p_legal)
                return legal_moves[move_idx]

def calculate_elo_update(r_a, r_b, score_a, k=32):
    e_a = 1 / (1 + 10 ** ((r_b - r_a) / 400))
    return r_a + k * (score_a - e_a)
