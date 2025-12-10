"""
SlowFast model loader and warmup.

Handles:
- Model loading from PyTorch Hub or custom weights
- Comprehensive warmup sequence
- Model validation
"""
import time
import torch
import torch.nn as nn
from typing import Optional, List
import numpy as np


class SlowFastModelLoader:
    """
    Loads and warms up SlowFast model.

    Provides comprehensive warmup to ensure stable inference.
    """

    def __init__(self,
                 model_name: str = "slowfast_r50",
                 weights_path: Optional[str] = None,
                 class_names: Optional[List[str]] = None,
                 device: str = "cuda",
                 use_fp16: bool = True):
        """
        Initialize model loader.

        Args:
            model_name: Model architecture name
            weights_path: Path to custom weights (optional)
            class_names: List of class names
            device: Device to load model on
            use_fp16: Use half precision
        """
        self.model_name = model_name
        self.weights_path = weights_path
        self.class_names = class_names or []
        self.device = torch.device(device)
        self.use_fp16 = use_fp16

    def load_model(self) -> nn.Module:
        """
        Load SlowFast model.

        Returns:
            Loaded model in eval mode
        """
        print(f"[MODEL_LOADER] Loading {self.model_name}...")

        try:
            # Load from PyTorch Hub
            model = torch.hub.load("facebookresearch/pytorchvideo",
                                  self.model_name,
                                  pretrained=True)

            # Load custom weights if provided
            if self.weights_path and len(self.class_names) > 0:
                print(f"[MODEL_LOADER] Loading custom weights for {len(self.class_names)} classes")

                # Replace final projection layer to match custom num_classes
                if hasattr(model, 'blocks') and hasattr(model.blocks[-1], 'proj'):
                    original_features = model.blocks[-1].proj.in_features
                    model.blocks[-1].proj = nn.Linear(original_features, len(self.class_names))

                    # Now load the custom weights
                    state_dict = torch.load(self.weights_path, map_location="cpu")
                    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

                    print(f"[MODEL_LOADER] Custom weights loaded ({len(missing_keys)} missing keys)")
                else:
                    raise RuntimeError("Could not find expected classifier structure")

            # Move to device
            model = model.to(self.device)

            # Enable FP16 if requested
            if self.use_fp16 and self.device.type == "cuda":
                model = model.half()

            model.eval()

            print(f"[MODEL_LOADER] Model loaded successfully")
            return model

        except Exception as e:
            print(f"[MODEL_LOADER] Error loading model: {e}")
            raise

    def warmup_model(self, model: nn.Module, iterations: int = 5) -> bool:
        """
        Comprehensive warmup for reliable inference.

        Args:
            model: Model to warmup
            iterations: Number of warmup iterations

        Returns:
            True if warmup successful
        """
        print(f"[WARMUP] Starting comprehensive warmup ({iterations} iterations)...")

        try:
            with torch.no_grad():
                for i in range(iterations):
                    # Create dummy input
                    slow = torch.randn(1, 3, 8, 224, 224).to(self.device)
                    fast = torch.randn(1, 3, 32, 224, 224).to(self.device)

                    if self.use_fp16 and self.device.type == "cuda":
                        slow = slow.half()
                        fast = fast.half()

                    # Forward pass
                    start = time.time()
                    logits = model([slow, fast])
                    elapsed = time.time() - start

                    print(f"[WARMUP] Iteration {i+1}/{iterations}: "
                          f"{elapsed*1000:.1f}ms, output shape: {logits.shape}")

            print("[WARMUP] Warmup completed successfully")
            return True

        except Exception as e:
            print(f"[WARMUP] Error during warmup: {e}")
            return False

    def test_production_pipeline(self, model: nn.Module,
                                batch_processor) -> bool:
        """
        Test full production pipeline.

        Args:
            model: Loaded model
            batch_processor: BatchProcessor instance

        Returns:
            True if test successful
        """
        print("[WARMUP] Testing production pipeline...")

        try:
            # Create dummy BGR frames
            dummy_frames = [
                np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                for _ in range(64)
            ]

            # Process through batch processor
            slow, fast = batch_processor.prepare_batch([dummy_frames])

            # Inference
            with torch.no_grad():
                start = time.time()
                logits = model([slow, fast])
                elapsed = time.time() - start

            # Get prediction
            probs = torch.softmax(logits, dim=1)
            top_prob, top_idx = torch.max(probs, dim=1)

            print(f"[WARMUP] Production pipeline test successful")
            print(f"[WARMUP] Inference time: {elapsed*1000:.1f}ms")
            print(f"[WARMUP] Output: class {top_idx.item()}, "
                  f"prob {top_prob.item():.3f}")

            return True

        except Exception as e:
            print(f"[WARMUP] Production pipeline test failed: {e}")
            return False

    def run_comprehensive_warmup(self, model: nn.Module,
                                batch_processor) -> bool:
        """
        Run complete warmup sequence.

        Args:
            model: Loaded model
            batch_processor: BatchProcessor instance

        Returns:
            True if all warmup successful
        """
        print("="*60)
        print("STARTING COMPREHENSIVE MODEL WARMUP")
        print("="*60)

        # Stage 1: Basic warmup
        if not self.warmup_model(model, iterations=5):
            return False

        # Stage 2: Production pipeline test
        if not self.test_production_pipeline(model, batch_processor):
            return False

        print("="*60)
        print("WARMUP COMPLETED SUCCESSFULLY")
        print("="*60)

        return True
