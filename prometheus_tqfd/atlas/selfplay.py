import time
import torch
import chess
from typing import List, Tuple, Dict
from multiprocessing import Queue, Event
from prometheus_tqfd.config import PrometheusConfig
from prometheus_tqfd.atlas.network import AtlasNetwork
from prometheus_tqfd.atlas.mcts import MCTS
from prometheus_tqfd.encoding import BoardEncoder, MoveEncoder

class AtlasSelfPlayWorker:
    """
    Generiert Self-Play-Spiele mit MCTS.
    """

    def __init__(self, config: PrometheusConfig, weights_queue: Queue,
                 data_queue: Queue, device: str, worker_id: int):
        self.config = config
        self.weights_queue = weights_queue
        self.data_queue = data_queue
        self.device = device
        self.worker_id = worker_id

        self.network = AtlasNetwork(config).to(device)
        self.network.eval()
        self.mcts = MCTS(config, self.network, device)
        self.encoder = BoardEncoder()
        self.move_encoder = MoveEncoder()

        self.games_played = 0
        self.weights_version = 0

    def run(self, stop_event: Event, heartbeat_dict: Dict):
        """Hauptschleife des Workers"""
        while not stop_event.is_set():
            # Heartbeat
            heartbeat_dict[f'atlas_selfplay_{self.worker_id}'] = time.time()

            # Gewichte updaten
            self._maybe_update_weights()

            # Spiel spielen
            trajectory = self._play_game()

            # In Queue schieben
            self.data_queue.put(trajectory)
            self.games_played += 1

    def _maybe_update_weights(self):
        """Lade neue Gewichte wenn verfügbar"""
        try:
            # Get latest weights from queue
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

    def _play_game(self) -> List[Tuple[torch.Tensor, torch.Tensor, float]]:
        """Spiele ein komplettes Spiel"""
        trajectory = []
        board = chess.Board()
        move_count = 0

        while not board.is_game_over():
            # Temperatur-Schedule
            if move_count < self.config.atlas_temperature_moves:
                temperature = self.config.atlas_temperature_init
            else:
                temperature = self.config.atlas_temperature_final

            # MCTS
            root = self.mcts.search(board)

            # Daten sammeln
            state_tensor = self.encoder.encode(board)
            policy_target = self.mcts.get_policy_target(root, temperature)

            trajectory.append((state_tensor, policy_target, None))  # Value später

            # Zug ausführen
            move = self.mcts.select_move(root, temperature)
            board.push(move)
            move_count += 1

            # Spiellänge begrenzen
            if move_count > 400: # Slightly more than spec's 300 for safety
                break

        # Value-Targets mit Spielergebnis füllen
        result = self._get_result(board)
        final_trajectory = []
        for i in range(len(trajectory)):
            # Perspektive: Weiß bei geraden Zügen (0, 2...), Schwarz bei ungeraden (1, 3...)
            # Wenn i=0, turn=White, perspective=1, value_target = result * 1
            # Wenn result=1 (White wins), i=0 gets 1.
            # Wenn i=1, turn=Black, perspective=-1, value_target = result * -1 = -1. Correct.
            perspective = 1 if i % 2 == 0 else -1
            value_target = result * perspective
            final_trajectory.append((trajectory[i][0], trajectory[i][1], value_target))

        return final_trajectory

    def _get_result(self, board: chess.Board) -> float:
        """Spielergebnis aus Weiß-Perspektive"""
        result = board.result()
        if result == "1-0":
            return 1.0
        elif result == "0-1":
            return -1.0
        return 0.0
