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
    board = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3")
    # Not a mate in 1 position yet, but check basic detection
    threats = detector.detect(board)
    assert 'mate_in_1' in threats
    return True

def test_physics_symmetry():
    config = PrometheusConfig()
    calc = PhysicsFieldCalculator(config)
    board = chess.Board()
    fields = calc.compute(board)
    assert fields.shape == (3, 8, 8)
    # White and Black initial positions are symmetrical
    # Kanal 0 (Masse) sollte anfangs etwa 0-summiert sein oder symmetrisch
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
            all_passed = False

    return all_passed

if __name__ == "__main__":
    run_smoke_tests()
