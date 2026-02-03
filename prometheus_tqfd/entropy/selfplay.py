import torch
import chess
import random
import numpy as np
from prometheus_tqfd.entropy.fields import FieldCalculator
from prometheus_tqfd.entropy.network import build_graph_data, HAS_PYG
from prometheus_tqfd.encoding import BoardEncoder, MoveEncoder

class EntropySelfPlayWorker:
    def __init__(self, config, weights_queue, data_queue, stop_event, worker_id):
        self.config = config
        self.weights_queue = weights_queue
        self.data_queue = data_queue
        self.stop_event = stop_event
        self.worker_id = worker_id

        from prometheus_tqfd.entropy.network import EntropyNetwork
        self.network = EntropyNetwork(config).to(config.actor_device)
        self.network.eval()

        self.field_calc = FieldCalculator(config.physics)
        self.board_encoder = BoardEncoder()
        self.move_encoder = MoveEncoder()

        self.temperature = config.entropy_temp_start

    def run(self):
        print(f"Entropy Worker {self.worker_id} started.")
        while not self.stop_event.is_set():
            if not self.weights_queue.empty():
                try:
                    self.network.load_state_dict(self.weights_queue.get_nowait())
                except:
                    pass

            trajectory = self._play_game()
            self.data_queue.put(trajectory)

            # Anneal temperature
            self.temperature = max(self.config.entropy_temp_end, self.temperature * self.config.entropy_temp_decay)

    def _play_game(self):
        board = chess.Board()
        trajectory = []

        while not board.is_game_over() and not self.stop_event.is_set():
            field_tensor = self.field_calc.compute_fields(board).unsqueeze(0).to(self.config.actor_device)
            if self.network.has_pyg:
                board_data = build_graph_data(board)
                if board_data: board_data = board_data.to(self.config.actor_device)
            else:
                board_data = self.board_encoder.encode(board).unsqueeze(0).to(self.config.actor_device)

            mask = self.move_encoder.get_legal_mask(board).unsqueeze(0).to(self.config.actor_device)

            with torch.no_grad():
                probs, energy, _, _ = self.network(field_tensor, board_data, mask)

            # Boltzmann sampling
            probs = probs[0].cpu().numpy()
            legal_moves = list(board.legal_moves)
            legal_indices = [self.move_encoder.move_to_index(m) for m in legal_moves]

            p_legal = probs[legal_indices]
            p_legal = np.maximum(p_legal, 1e-8)

            # Apply temperature
            logits = np.log(p_legal) / self.temperature
            exp_logits = np.exp(logits - np.max(logits))
            p_final = exp_logits / exp_logits.sum()

            move_idx = np.random.choice(len(legal_moves), p=p_final)
            move = legal_moves[move_idx]

            # Opponent legal count for entropy loss
            temp_board = board.copy()
            temp_board.push(move)
            opp_legal_count = temp_board.legal_moves.count()

            # Energy captured
            captured_piece = board.piece_at(move.to_square)
            e_captured = self.field_calc.piece_energies[captured_piece.piece_type] if captured_piece else 0.0

            step_data = {
                'field': field_tensor.cpu().squeeze(0),
                'board': board_data.x.cpu() if self.network.has_pyg else board_data.cpu().squeeze(0),
                'mask': mask.cpu().squeeze(0),
                'opp_legal_count': opp_legal_count,
                'e_captured': e_captured,
                'energy_before': energy.item(),
                'board_obj': board.copy() # For later building pairs
            }

            trajectory.append(step_data)
            board.push(move)

        return trajectory
