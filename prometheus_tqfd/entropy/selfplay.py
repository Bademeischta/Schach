import time
import torch
import chess
from typing import List, Dict, Tuple
from multiprocessing import Queue, Event
from prometheus_tqfd.config import PrometheusConfig
from prometheus_tqfd.entropy.network import EntropyNetworkV2
from prometheus_tqfd.entropy.rollout import MiniRolloutSelector
from prometheus_tqfd.tactics import TacticsDetector
from prometheus_tqfd.encoding import BoardEncoder, MoveEncoder
from prometheus_tqfd.physics import PhysicsFieldCalculator

class EntropySelfPlayWorker:
    """
    Generiert Self-Play-Spiele mit Mini-Rollouts.
    """

    def __init__(self, config: PrometheusConfig, weights_queue: Queue,
                 data_queue: Queue, device: str, worker_id: int):
        self.config = config
        self.weights_queue = weights_queue
        self.data_queue = data_queue
        self.device = device
        self.worker_id = worker_id

        self.network = EntropyNetworkV2(config).to(device)
        self.network.eval()
        self.tactics = TacticsDetector(config)
        self.selector = MiniRolloutSelector(config, self.network, self.tactics, device)

        self.encoder = BoardEncoder()
        self.field_calc = PhysicsFieldCalculator(config)
        self.move_encoder = MoveEncoder()

        self.temperature = config.entropy_temperature_start
        self.games_played = 0
        self.weights_version = 0

    def run(self, stop_event: Event, heartbeat_dict: Dict, metrics_queue: Queue = None):
        """Hauptschleife"""
        while not stop_event.is_set():
            heartbeat_dict[f'entropy_selfplay_{self.worker_id}'] = time.time()

            self._maybe_update_weights()
            trajectory = self._play_game(heartbeat_dict, metrics_queue)
            self.data_queue.put(trajectory)

            self.games_played += 1
            self._decay_temperature()

    def _play_game(self, heartbeat_dict: Dict, metrics_queue: Queue = None) -> List[Dict]:
        """Spiele ein komplettes Spiel"""
        trajectory = []
        board = chess.Board()

        while not board.is_game_over() and len(trajectory) < 400:
            # Heartbeat inside loop for long games
            heartbeat_dict[f'entropy_selfplay_{self.worker_id}'] = time.time()

            # Periodic dashboard update
            if metrics_queue and len(trajectory) % 10 == 0:
                self._send_dashboard_update(metrics_queue, board)

            # Daten für diesen Zug
            step_data = self._collect_step_data(board)

            # Zug wählen
            move, energy_before = self.selector.select_move(board, self.temperature)
            if move is None: break

            step_data['move_idx'] = self.move_encoder.move_to_index(move)
            step_data['energy_before'] = energy_before

            # Zug ausführen
            board.push(move)

            # Energie nach Zug
            step_data['energy_after'] = self._get_energy(board)
            step_data['legal_count_opponent'] = len(list(board.legal_moves))

            trajectory.append(step_data)

        # Spielergebnis hinzufügen
        result = self._get_result(board)
        for i, step in enumerate(trajectory):
            perspective = 1 if i % 2 == 0 else -1
            step['game_result'] = result * perspective

        return trajectory

    def _collect_step_data(self, board: chess.Board) -> Dict:
        """Sammle alle Daten für einen Zug"""
        board_tensor = self.encoder.encode(board)
        field_tensor = self.field_calc.compute(board)

        # Features for RND
        with torch.no_grad():
            features = self.network.get_features(
                board_tensor.unsqueeze(0).to(self.device),
                field_tensor.unsqueeze(0).to(self.device)
            ).squeeze(0).cpu()

            # Also need policy logits for mobility loss
            policy_logits, _ = self.network(
                board_tensor.unsqueeze(0).to(self.device),
                field_tensor.unsqueeze(0).to(self.device)
            )
            policy_logits = policy_logits.squeeze(0).cpu()

        return {
            'board_tensor': board_tensor,
            'field_tensor': field_tensor,
            'features': features,
            'policy_logits': policy_logits,
            'legal_count_self': len(list(board.legal_moves)),
        }

    def _get_energy(self, board: chess.Board) -> float:
        board_tensor = self.encoder.encode(board).unsqueeze(0).to(self.device)
        field_tensor = torch.tensor(self.field_calc.compute(board)).unsqueeze(0).float().to(self.device)

        with torch.no_grad():
            _, energy = self.network(board_tensor, field_tensor)
        return energy.item()

    def _send_dashboard_update(self, metrics_queue, board):
        from prometheus_tqfd.dashboard.heatmaps import get_entropy_heatmap
        field_tensor = self.field_calc.compute(board)
        heatmap = get_entropy_heatmap(field_tensor)
        metrics_queue.put({
            'type': 'entropy_update',
            'heatmap': heatmap.tolist(),
            'fen': board.fen(),
            'event': 'move',
            'player': 'ENTROPY',
            'move': board.move_stack[-1].uci() if board.move_stack else 'None'
        })

    def _get_result(self, board: chess.Board) -> float:
        result = board.result()
        if result == "1-0":
            return 1.0
        elif result == "0-1":
            return -1.0
        return 0.0

    def _decay_temperature(self):
        self.temperature = max(
            self.config.entropy_temperature_end,
            self.temperature * self.config.entropy_temperature_decay
        )

    def _maybe_update_weights(self):
        try:
            latest = None
            while not self.weights_queue.empty():
                latest = self.weights_queue.get_nowait()
            if latest:
                weights, version = latest
                if version > self.weights_version:
                    self.network.load_state_dict(weights)
                    self.weights_version = version
        except:
            pass
