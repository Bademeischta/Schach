import torch
import chess
from prometheus_tqfd.atlas.mcts import MCTS
from prometheus_tqfd.encoding import BoardEncoder, MoveEncoder
import time

class AtlasSelfPlayWorker:
    def __init__(self, config, weights_queue, data_queue, stop_event, worker_id):
        self.config = config
        self.weights_queue = weights_queue
        self.data_queue = data_queue
        self.stop_event = stop_event
        self.worker_id = worker_id

        from prometheus_tqfd.atlas.network import AtlasNetwork
        self.network = AtlasNetwork(config).to(config.actor_device)
        self.network.eval()

        self.mcts = MCTS(config)
        self.board_encoder = BoardEncoder()
        self.move_encoder = MoveEncoder()

    def run(self):
        print(f"Atlas Worker {self.worker_id} started.")
        while not self.stop_event.is_set():
            # Check for new weights
            if not self.weights_queue.empty():
                try:
                    state_dict = self.weights_queue.get_nowait()
                    self.network.load_state_dict(state_dict)
                except:
                    pass

            trajectory = self._play_game()
            self.data_queue.put(trajectory)

    def _play_game(self):
        board = chess.Board()
        trajectory = []

        while not board.is_game_over() and not self.stop_event.is_set():
            state_tensor = self.board_encoder.encode(board)

            root = self.mcts.search(board, self.network, self.config.actor_device)

            # Policy target: normalized visit counts
            policy_target = torch.zeros(4672)
            total_visits = sum(child.visit_count for child in root.children.values())
            for move, child in root.children.items():
                policy_target[self.move_encoder.move_to_index(move)] = child.visit_count / total_visits

            # Select action
            temp = self.config.atlas_temp_init if len(board.move_stack) < self.config.atlas_temp_moves else self.config.atlas_temp_final
            # Wait, atlas_temp_init/final are not in my config.py yet, I should add them or use defaults.
            # I used atlas_temp_moves=30. I'll use 1.0 and 0.1 as default temps.
            move = self.mcts.select_action(root, temperature=temp if hasattr(self.config, 'atlas_temp_init') else 1.0)

            trajectory.append({
                'state': state_tensor,
                'policy': policy_target,
                'value': None, # To be filled
                'turn': board.turn
            })

            board.push(move)

        # Fill results
        result_str = board.result()
        result = 0.0
        if result_str == "1-0": result = 1.0
        elif result_str == "0-1": result = -1.0

        for step in trajectory:
            # Value from perspective of player whose turn it is
            step['value'] = result if step['turn'] == chess.WHITE else -result

        return trajectory
