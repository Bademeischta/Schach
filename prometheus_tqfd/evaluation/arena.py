import chess
import torch
import random
from typing import Dict, Tuple, List, Optional
from prometheus_tqfd.config import PrometheusConfig
from prometheus_tqfd.atlas.network import AtlasNetwork
from prometheus_tqfd.atlas.mcts import MCTS
from prometheus_tqfd.entropy.network import EntropyNetworkV2
from prometheus_tqfd.entropy.rollout import MiniRolloutSelector
from prometheus_tqfd.tactics import TacticsDetector
from prometheus_tqfd.evaluation.baselines import RandomPlayer, HeuristicPlayer

class AtlasArenaPlayer:
    def __init__(self, config, network, device):
        self.mcts = MCTS(config, network, device)
    def select_move(self, board):
        root = self.mcts.search(board, num_simulations=50) # Reduced for faster eval
        return self.mcts.select_move(root, temperature=0)

class EntropyArenaPlayer:
    def __init__(self, config, network, device):
        self.tactics = TacticsDetector(config)
        self.selector = MiniRolloutSelector(config, network, self.tactics, device)
    def select_move(self, board):
        move, _ = self.selector.select_move(board, temperature=0)
        return move

class Arena:
    """
    Veranstaltet Duelle zwischen Spielern und berechnet ELO.
    """

    def __init__(self, config: PrometheusConfig):
        self.config = config
        self.elo_ratings = {
            'atlas': config.elo_initial,
            'entropy': config.elo_initial,
            'random': 400.0,
            'heuristic': 800.0,
        }
        self.match_history = []

    def run_evaluation(self, atlas_network: AtlasNetwork,
                       entropy_network: EntropyNetworkV2,
                       device: str) -> Dict:
        """Komplette Evaluationsrunde"""
        atlas_network.eval()
        entropy_network.eval()

        results = {}

        # Players
        atlas_player = AtlasArenaPlayer(self.config, atlas_network, device)
        entropy_player = EntropyArenaPlayer(self.config, entropy_network, device)
        random_player = RandomPlayer()
        heuristic_player = HeuristicPlayer()

        # 1. ATLAS vs ENTROPY
        wins_a, wins_e, draws = self._play_match(atlas_player, entropy_player, self.config.eval_games_atlas_entropy)
        results['atlas_vs_entropy'] = {'atlas': wins_a, 'entropy': wins_e, 'draws': draws}
        self._update_elo('atlas', 'entropy', wins_a, wins_e, draws)

        # 2. vs Baselines
        for name, player in [('atlas', atlas_player), ('entropy', entropy_player)]:
            # vs Random
            w, l, d = self._play_match(player, random_player, self.config.eval_games_vs_random)
            results[f'{name}_vs_random'] = {'wins': w, 'losses': l, 'draws': d}
            self._update_elo(name, 'random', w, l, d, update_b=False)

            # vs Heuristic
            w, l, d = self._play_match(player, heuristic_player, self.config.eval_games_vs_heuristic)
            results[f'{name}_vs_heuristic'] = {'wins': w, 'losses': l, 'draws': d}
            self._update_elo(name, 'heuristic', w, l, d, update_b=False)

        results['elo'] = self.elo_ratings.copy()
        return results

    def _play_match(self, p1, p2, num_games: int) -> Tuple[int, int, int]:
        wins1, wins2, draws = 0, 0, 0
        for i in range(num_games):
            # Alternate colors
            if i % 2 == 0:
                res = self._play_game(p1, p2)
                if res == 1.0: wins1 += 1
                elif res == -1.0: wins2 += 1
                else: draws += 1
            else:
                res = self._play_game(p2, p1)
                if res == 1.0: wins2 += 1
                elif res == -1.0: wins1 += 1
                else: draws += 1
        return wins1, wins2, draws

    def _play_game(self, white_player, black_player, max_moves: int = 200) -> float:
        board = chess.Board()
        while not board.is_game_over() and board.fullmove_number <= max_moves:
            player = white_player if board.turn == chess.WHITE else black_player
            move = player.select_move(board)
            if move is None or move not in board.legal_moves:
                # Fallback to random if player fails
                move = random.choice(list(board.legal_moves))
            board.push(move)

        res = board.result()
        if res == "1-0": return 1.0
        if res == "0-1": return -1.0
        return 0.0

    def _update_elo(self, p_a: str, p_b: str, wins_a: int, wins_b: int, draws: int, update_b: bool = True):
        total = wins_a + wins_b + draws
        if total == 0: return

        ra = self.elo_ratings[p_a]
        rb = self.elo_ratings[p_b]

        ea = 1 / (1 + 10 ** ((rb - ra) / 400))
        sa = (wins_a + 0.5 * draws) / total

        k = self.config.elo_k_factor
        self.elo_ratings[p_a] = ra + k * (sa - ea)
        if update_b:
            eb = 1 - ea
            sb = 1 - sa
            self.elo_ratings[p_b] = rb + k * (sb - eb)
