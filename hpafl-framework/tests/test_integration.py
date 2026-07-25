"""Integration tests for the complete HPAFL federated learning pipeline.

Tests the full workflow:
  1. Model creation with FedBN
  2. Client initialization
  3. One complete federated round
  4. Privacy accounting
  5. Secure aggregation
  6. Adaptive weighting
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import logging
import tempfile
from typing import Dict, List

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from clients.client import HAM10000FlowerClient
from config import CLASS_NAMES, HAPFLConfig
from models.efficientnet_model import get_model, get_bn_layer_names
from privacy.dp_trainer import DPTrainer, PrivacyBudgetLogger
from security.secure_agg import SecureAggregator
from server.adaptive_agg import AdaptiveAggregationStrategy

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@pytest.fixture
def config():
    """Create a minimal test configuration."""
    cfg = HAPFLConfig()
    cfg.pretrained = False
    cfg.num_rounds = 1
    cfg.local_epochs = 1
    cfg.batch_size = 8
    cfg.image_size = 64
    cfg.noise_multiplier = 1.1
    cfg.target_epsilon = 8.0
    cfg.target_delta = 1e-5

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg.output_root = tmpdir
        cfg.__post_init__()
        yield cfg


@pytest.fixture
def device():
    """Get device (CPU for testing, GPU if available)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestModelCreation:
    """Test model initialization and architecture."""

    def test_model_loads_without_pretrained(self, config):
        """Model should load without ImageNet pretrained weights for speed."""
        model = get_model(config)
        assert model is not None
        assert isinstance(model, torch.nn.Module)
        logger.info("✓ Model created successfully without pretrained weights")

    def test_bn_layers_detected(self, config):
        """Model must have detectable BN layers for FedBN."""
        model = get_model(config)
        bn_names = get_bn_layer_names(model)
        assert len(bn_names) > 0, "Model should have batch norm layers"
        logger.info(f"✓ Detected {len(bn_names)} batch norm parameters")

    def test_model_forward_pass(self, config, device):
        """Model forward pass should work with expected I/O shapes."""
        model = get_model(config).to(device)
        batch_size = 4
        images = torch.randn(batch_size, 3, config.image_size, config.image_size).to(device)

        with torch.no_grad():
            logits = model(images)

        assert logits.shape == (batch_size, config.num_classes)
        logger.info(f"✓ Forward pass successful: {images.shape} → {logits.shape}")


class TestDataAndDataLoader:
    """Test data handling and loader creation."""

    def test_synthetic_dataloader(self, config, device):
        """Create synthetic dataloaders for testing."""
        num_samples = 100
        images = torch.randn(num_samples, 3, config.image_size, config.image_size)
        labels = torch.randint(0, config.num_classes, (num_samples,))

        dataset = TensorDataset(images, labels)
        dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

        assert len(dataloader) > 0
        batch_images, batch_labels = next(iter(dataloader))
        assert batch_images.shape[0] <= config.batch_size
        assert batch_labels.shape[0] <= config.batch_size
        logger.info(f"✓ Dataloader created with {len(dataset)} synthetic samples")

    def test_dataloader_batching(self, config):
        """Verify dataloader produces correct batch shapes."""
        num_samples = 25
        images = torch.randn(num_samples, 3, config.image_size, config.image_size)
        labels = torch.randint(0, config.num_classes, (num_samples,))
        dataset = TensorDataset(images, labels)
        dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)

        batch_count = 0
        for batch_images, batch_labels in dataloader:
            batch_count += 1
            assert batch_images.shape[1:] == (3, config.image_size, config.image_size)
            assert batch_labels.shape[0] == batch_images.shape[0]

        assert batch_count > 0
        logger.info(f"✓ All {batch_count} batches validated")


class TestPrivacy:
    """Test differential privacy implementation."""

    def test_dp_trainer_initialization(self, config):
        """DPTrainer should initialize without errors."""
        dp_trainer = DPTrainer(config)
        assert dp_trainer is not None
        assert dp_trainer.config == config
        logger.info("✓ DPTrainer initialized")

    def test_privacy_budget_logger(self, config):
        """PrivacyBudgetLogger should create CSV file."""
        logger_obj = PrivacyBudgetLogger(config.results_dir)
        assert logger_obj.path.exists()

        logger_obj.log(
            round_num=1,
            hospital_id="A",
            epsilon=2.5,
            delta=1e-5,
            target_epsilon=8.0,
        )

        # Verify CSV was written
        with open(logger_obj.path) as f:
            lines = f.readlines()
        assert len(lines) >= 2  # Header + at least 1 row
        logger.info(f"✓ Privacy budget logged to {logger_obj.path}")

    def test_dp_trainer_attach_requires_dp_compatible_model(self, config, device):
        """DP attachment should handle model conversion."""
        model = get_model(config).to(device)

        # Fix model for DP compatibility BEFORE creating optimizer
        try:
            from opacus.validators import ModuleValidator
            if not ModuleValidator.is_valid(model):
                model = ModuleValidator.fix(model)
                model = model.to(device)

            # Create optimizer AFTER model is fixed
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

            # Create synthetic dataloader
            images = torch.randn(16, 3, config.image_size, config.image_size)
            labels = torch.randint(0, config.num_classes, (16,))
            dataset = TensorDataset(images, labels)
            dataloader = DataLoader(dataset, batch_size=config.batch_size)

            dp_trainer = DPTrainer(config)
            dp_model, dp_optimizer, dp_dataloader = dp_trainer.attach(
                model, optimizer, dataloader
            )
            assert dp_model is not None
            logger.info("✓ DP attachment successful")
        except ImportError:
            logger.warning("Opacus not installed; skipping DP attachment test")


class TestSecureAggregation:
    """Test secure aggregation protocol."""

    def test_secure_aggregator_initialization(self, config):
        """SecureAggregator should initialize correctly."""
        agg = SecureAggregator(num_clients=3, threshold=2, seed=42)
        assert agg.num_clients == 3
        assert agg.threshold == 2
        logger.info("✓ SecureAggregator initialized")

    def test_secure_aggregator_masking(self):
        """Test pairwise masking and aggregation."""
        num_clients = 3
        num_layers = 2
        layer_sizes = [10, 20]

        agg = SecureAggregator(num_clients=num_clients, threshold=2, seed=42)

        # Create synthetic updates
        updates: Dict[int, List[np.ndarray]] = {}
        for i in range(num_clients):
            updates[i] = [np.random.randn(size) for size in layer_sizes]

        # Mask updates
        masked_updates: Dict[int, List[np.ndarray]] = {}
        for client_id, update in updates.items():
            masked_updates[client_id] = agg.mask_update(
                client_id=client_id,
                update=update,
                round_num=1,
            )

        # Aggregate
        true_aggregate = agg.aggregate_with_round(masked_updates, round_num=1)
        assert len(true_aggregate) == num_layers
        assert true_aggregate[0].shape == (layer_sizes[0],)
        assert true_aggregate[1].shape == (layer_sizes[1],)

        logger.info("✓ Secure aggregation (masking + unmasking) successful")

    def test_secure_aggregator_security_verification(self):
        """Test that individual updates are not recoverable."""
        num_clients = 3
        layer_sizes = [5, 10]

        agg = SecureAggregator(num_clients=num_clients, threshold=2)
        updates: Dict[int, List[np.ndarray]] = {
            i: [np.random.randn(size) for size in layer_sizes]
            for i in range(num_clients)
        }

        masked_updates: Dict[int, List[np.ndarray]] = {}
        for client_id, update in updates.items():
            masked_updates[client_id] = agg.mask_update(
                client_id=client_id, update=update, round_num=1
            )

        true_aggregate = agg.aggregate_with_round(masked_updates, round_num=1)

        # Verify security property
        is_secure = agg.verify_security(masked_updates, true_aggregate)
        assert is_secure, "Security verification failed"
        logger.info("✓ Security verification passed (individual updates not recoverable)")


class TestAdaptiveAggregation:
    """Test adaptive weighted aggregation."""

    def test_adaptive_strategy_initialization(self, config):
        """AdaptiveAggregationStrategy should initialize."""
        agg_obj = SecureAggregator(num_clients=3, threshold=2)
        strategy = AdaptiveAggregationStrategy(
            config=config,
            secure_aggregator=agg_obj,
        )
        assert strategy is not None
        assert strategy.config == config
        logger.info("✓ AdaptiveAggregationStrategy initialized")

    def test_adaptive_weight_computation(self, config):
        """Weights should be computed based on accuracy, reliability, etc."""
        agg_obj = SecureAggregator(num_clients=3, threshold=2)
        strategy = AdaptiveAggregationStrategy(config, agg_obj)

        # Simulate client metrics
        weight, score = strategy._compute_adaptive_weight(
            client_id="A",
            accuracy=0.85,
            num_samples=100,
            max_samples=100,
            round_num=1,
        )

        assert 0 < weight <= 1, f"Weight should be in (0, 1], got {weight}"
        assert score > 0, f"Score should be positive, got {score}"
        logger.info(f"✓ Adaptive weight computed: {weight:.4f} for Hospital A")


class TestEndToEnd:
    """Full end-to-end integration tests."""

    def test_single_federated_round(self, config, device):
        """Simulate one complete federated learning round."""
        logger.info("=" * 70)
        logger.info("RUNNING FULL FEDERATED LEARNING ROUND TEST")
        logger.info("=" * 70)

        # 1. Create model
        logger.info("\n1. Creating model...")
        global_model = get_model(config).to(device)
        logger.info(f"   ✓ Model created on {device}")

        # 2. Fix model for DP compatibility
        logger.info("\n2. Preparing model for DP-SGD...")
        try:
            from opacus.validators import ModuleValidator
            if not ModuleValidator.is_valid(global_model):
                global_model = ModuleValidator.fix(global_model)
                global_model = global_model.to(device)
            logger.info("   ✓ Model is DP-compatible")
        except ImportError:
            logger.warning("   ⚠ Opacus not available; skipping DP validation")

        # 3. Create synthetic hospital data
        logger.info("\n3. Creating synthetic hospital data...")
        hospital_dataloaders = {}
        for hospital_id in ["A", "B", "C"]:
            num_samples = 32
            images = torch.randn(num_samples, 3, config.image_size, config.image_size)
            labels = torch.randint(0, config.num_classes, (num_samples,))
            dataset = TensorDataset(images, labels)
            dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
            hospital_dataloaders[hospital_id] = dataloader
            logger.info(f"   ✓ Hospital {hospital_id}: {num_samples} synthetic samples")

        # 4. Simulate local training and updates
        logger.info("\n4. Simulating local training...")
        client_updates = {}
        client_accuracies = {}
        for hospital_id in ["A", "B", "C"]:
            model_copy = get_model(config).to(device)
            # Fix model copy for DP compatibility before loading state dict
            try:
                from opacus.validators import ModuleValidator
                if not ModuleValidator.is_valid(model_copy):
                    model_copy = ModuleValidator.fix(model_copy)
                    model_copy = model_copy.to(device)
            except ImportError:
                pass
            model_copy.load_state_dict(global_model.state_dict())

            # Dummy local training (just add small noise to simulate update)
            for param in model_copy.parameters():
                param.data += torch.randn_like(param.data) * 0.01

            # Get non-BN parameters
            bn_names = set(get_bn_layer_names(model_copy))
            params = [
                param.detach().cpu().numpy()
                for name, param in model_copy.named_parameters()
                if name not in bn_names
            ]
            client_updates[hospital_id] = params
            client_accuracies[hospital_id] = 0.7 + np.random.random() * 0.2
            logger.info(f"   ✓ Hospital {hospital_id}: trained model (accuracy={client_accuracies[hospital_id]:.3f})")

        # 5. Apply secure aggregation masking
        logger.info("\n5. Applying secure aggregation...")
        agg = SecureAggregator(num_clients=3, threshold=2, seed=42)
        masked_updates = {}
        for i, hospital_id in enumerate(["A", "B", "C"]):
            masked = agg.mask_update(
                client_id=i,
                update=client_updates[hospital_id],
                round_num=1,
            )
            masked_updates[i] = masked
            logger.info(f"   ✓ Hospital {hospital_id}: masked update applied")

        # 6. Unmasking and aggregation
        logger.info("\n6. Aggregating updates...")
        aggregated_params = agg.aggregate_with_round(masked_updates, round_num=1)
        logger.info("   ✓ Secure aggregate computed")

        # 7. Compute adaptive weights
        logger.info("\n7. Computing adaptive weights...")
        adaptive_strategy = AdaptiveAggregationStrategy(config, agg)
        weights = {}
        for hospital_id in ["A", "B", "C"]:
            weight, score = adaptive_strategy._compute_adaptive_weight(
                hospital_id,
                client_accuracies[hospital_id],
                num_samples=32,
                max_samples=32,
                round_num=1,
            )
            weights[hospital_id] = weight
            logger.info(f"   ✓ Hospital {hospital_id}: weight={weight:.4f}, score={score:.4f}")

        # Normalize weights
        total_weight = sum(weights.values())
        norm_weights = {h: w / total_weight for h, w in weights.items()}
        logger.info(f"   ✓ Normalized weights: {norm_weights}")

        # 8. Weighted aggregation
        logger.info("\n8. Performing weighted aggregation...")
        num_params = len(client_updates["A"])
        global_updated = [np.zeros_like(client_updates["A"][i]) for i in range(num_params)]

        for i, hospital_id in enumerate(["A", "B", "C"]):
            w = norm_weights[hospital_id]
            for j in range(num_params):
                global_updated[j] += w * client_updates[hospital_id][j]

        logger.info("   ✓ Global model updated")

        # 9. Verify outputs
        logger.info("\n9. Verifying outputs...")
        assert len(global_updated) == len(client_updates["A"])
        assert all(isinstance(p, np.ndarray) for p in global_updated)
        logger.info("   ✓ All outputs verified")

        logger.info("\n" + "=" * 70)
        logger.info("✓ FULL FEDERATED ROUND COMPLETED SUCCESSFULLY")
        logger.info("=" * 70)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
