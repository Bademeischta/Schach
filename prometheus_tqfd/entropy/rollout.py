import torch
import torch.nn.functional as F
import numpy as np
import chess
from typing import Tuple, Dict
from prometheus_tqfd.config import PrometheusConfig
from prometheus_tqfd.entropy.network import EntropyNetworkV2
from prometheus_tqfd.tactics import TacticsDetector
from prometheus_tqfd.encoding import BoardEncoder, MoveEncoder
from prometheus_tqfd.physics import PhysicsFieldCalculator

class MiniRolloutSelector:
    """
    Schaut 3-5 Züge voraus mit policy-guided Rollouts.
    Nutzt Energie als Evaluation statt klassischem Value.
    """

    def __init__(self, config: PrometheusConfig, network: EntropyNetworkV2,
                 tactics: TacticsDetector, device: str):
        self.config = config
        self.network = network
        self.tactics = tactics
        self.device = device

        self.encoder = BoardEncoder()
        self.field_calc = PhysicsFieldCalculator(config)
        self.move_encoder = MoveEncoder()

        self.depth = config.entropy_rollout_depth
        self.num_rollouts = config.entropy_rollout_count

    def select_move(self, board: chess.Board, temperature: float) -> Tuple[chess.Move, float]:
        """
        Wählt Zug basierend auf Mini-Rollouts.
        """
        # Taktik-Check: Matt in 1 sofort spielen
        threats = self.tactics.detect(board)
        if threats['mate_in_1']:
            return threats['mate_in_1'], 100.0

        legal_moves = list(board.legal_moves)
        if not legal_moves:
             return None, 0.0
        if len(legal_moves) == 1:
            return legal_moves[0], 0.0

        # Rollout-Scores für jeden Zug
        move_scores = {}
        for move in legal_moves:
            scores = []
            for _ in range(self.num_rollouts):
                score = self._rollout(board, move, self.depth)
                scores.append(score)
            move_scores[move] = np.mean(scores)

        # Taktik-Boost addieren
        tactic_boost = self.tactics.get_tactical_boost(board)
        for move in legal_moves:
            idx = self.move_encoder.move_to_index(move)
            move_scores[move] += tactic_boost[idx].item() * 0.1

        # Boltzmann-Sampling
        moves = list(move_scores.keys())
        scores = torch.tensor([move_scores[m] for m in moves])

        if temperature <= 0.01:
            chosen = moves[scores.argmax()]
        else:
            probs = F.softmax(scores / temperature, dim=0)
            idx = torch.multinomial(probs, 1).item()
            chosen = moves[idx]

        return chosen, move_scores[chosen]

    def _rollout(self, board: chess.Board, first_move: chess.Move, depth: int) -> float:
        """
        Simuliert Spiel für 'depth' Züge.
        """
        sim_board = board.copy()
        our_color = board.turn
        sim_board.push(first_move)

        for d in range(depth - 1):
            if sim_board.is_game_over():
                return self._terminal_value(sim_board, our_color)

            # Schnelle Zug-Auswahl (Policy-Sampling ohne Rollout)
            move = self._fast_select(sim_board)
            sim_board.push(move)

        # Energie am Ende
        energy = self._get_energy(sim_board)

        # Aus unserer Perspektive
        # Wenn sim_board.turn == our_color, dann ist energy aus unserer sicht
        # Aber die Energie wird vom Netz meist aus Sicht des aktuellen Spielers (oder absolut)
        # In unserem Netz wird BoardEncoder benutzt, der Kanal 18 für "turn" hat.
        # Wir müssen sicherstellen, dass wir die Energie konsistent interpretieren.
        # Im Spec steht: "Aus unserer Perspektive: Wenn sim_board.turn != our_color: energy = -energy"
        if sim_board.turn != our_color:
            energy = -energy

        return energy

    def _fast_select(self, board: chess.Board) -> chess.Move:
        """Schnelle Zug-Auswahl ohne Rollout"""
        # Erst Taktik prüfen
        threats = self.tactics.detect(board)
        if threats['mate_in_1']:
            return threats['mate_in_1']

        # Sonst Policy-Sampling
        board_tensor = self.encoder.encode(board).unsqueeze(0).to(self.device)
        field_tensor = self.field_calc.compute(board).unsqueeze(0).to(self.device)
        legal_mask = self.move_encoder.get_legal_mask(board).to(self.device)

        with torch.no_grad():
            policy_logits, _ = self.network(board_tensor, field_tensor)

        policy_logits[~legal_mask.unsqueeze(0)] = float('-inf')
        probs = F.softmax(policy_logits, dim=1).squeeze(0)

        idx = torch.multinomial(probs, 1).item()
        return self.move_encoder.index_to_move(idx, board)

    def _get_energy(self, board: chess.Board) -> float:
        """Energie einer Position"""
        board_tensor = self.encoder.encode(board).unsqueeze(0).to(self.device)
        field_tensor = self.field_calc.compute(board).unsqueeze(0).to(self.device)

        with torch.no_grad():
            _, energy = self.network(board_tensor, field_tensor)

        return energy.item()

    def _terminal_value(self, board: chess.Board, our_color: chess.Color) -> float:
        """Wert einer Endstellung"""
        result = board.result()
        if result == "1-0":
            return 10.0 if our_color == chess.WHITE else -10.0
        elif result == "0-1":
            return -10.0 if our_color == chess.WHITE else 10.0
        return 0.0
