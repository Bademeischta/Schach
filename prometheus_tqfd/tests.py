import torch
import chess
import numpy as np
from prometheus_tqfd.config import PrometheusConfig
from prometheus_tqfd.encoding import BoardEncoder, MoveEncoder
from prometheus_tqfd.atlas.network import AtlasNetwork
from prometheus_tqfd.atlas.mcts import MCTS
from prometheus_tqfd.entropy.network import EntropyNetworkV2
from prometheus_tqfd.entropy.rollout import MiniRolloutSelector
from prometheus_tqfd.tactics import TacticsDetector
from prometheus_tqfd.physics import PhysicsFieldCalculator

def test_encoding_roundtrip():
    encoder = BoardEncoder()
    board = chess.Board()
    tensor = encoder.encode(board)
    assert tensor.shape == (19, 8, 8)
    return True

def test_mcts_basic():
    config = PrometheusConfig()
    config.atlas_mcts_simulations = 10
    network = AtlasNetwork(config)
    mcts = MCTS(config, network, 'cpu')
    board = chess.Board()
    root = mcts.search(board)
    assert root.visit_count > 0
    return True

def test_tactics_detector():
    config = PrometheusConfig()
    detector = TacticsDetector(config)

    # Test Mate-in-1 detection
    # Position where White can play Qxh7# (if pawn at h7 is gone and king at g8)
    # Scholar's Mate
    board = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 4 4")
    threats = detector.detect(board)
    assert threats['mate_in_1'] is not None
    assert threats['mate_in_1'].uci() == "f3f7"

    # Test hanging piece detection
    board = chess.Board("rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2")
    threats = detector.detect(board)
    # e5 is attacked by f3 and not defended
    hanging_squares = [s for s, p in threats['hanging_pieces']]
    assert chess.E5 in hanging_squares

    return True

def test_physics_symmetry():
    config = PrometheusConfig()
    calc = PhysicsFieldCalculator(config)
    board = chess.Board()
    fields = calc.compute(board)
    assert fields.shape == (3, 8, 8)
    return True

def run_smoke_tests():
    print("🧪 Running Smoke Tests...")
    tests = [
        ("Encoding", test_encoding_roundtrip),
        ("MCTS", test_mcts_basic),
        ("Tactics", test_tactics_detector),
        ("Physics", test_physics_symmetry),
    ]

    all_passed = True
    for name, fn in tests:
        try:
            if fn():
                print(f"  ✅ {name} passed")
            else:
                print(f"  ❌ {name} failed")
                all_passed = False
        except Exception as e:
            print(f"  ❌ {name} error: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False

    return all_passed

if __name__ == "__main__":
    run_smoke_tests()
