import torch
import numpy as np
import threading
from typing import Tuple, Optional

from torchreid.scripts.main import build_config
from torchreid.tools.feature_extractor import FeatureExtractor
from torchreid.utils.constants import bn_correspondants


class SimpleArgs:
    """Minimal args object for build_config compatibility."""
    def __init__(self):
        self.root = ''
        self.save_dir = 'log'
        self.job_id = 'embedding_extractor'
        self.inference_enabled = False
        self.sources = None
        self.targets = None
        self.transforms = None
        self.opts = []


class PersonEmbedder:
    """
    Thread-safe person embedding extractor using BPBreID.
    Extracts embeddings from person crops for re-identification.
    """

    def __init__(self,
                 config_path: str = '/home/shaahidh/bpbreid/configs/test_reid.yaml',
                 weights_path: Optional[str] = None,
                 device: str = 'cuda'):
        """
        Initialize PersonEmbedder with BPBreID model.

        Args:
            config_path: Path to BPBreID config file
            weights_path: Optional path to model weights (overrides config)
            device: Device to use ('cuda' or 'cpu')
        """
        print("[PersonEmbedder] Initializing BPBreID model...")

        # Thread safety lock for inference
        self._inference_lock = threading.Lock()

        # Build config
        dummy_args = SimpleArgs()
        self.cfg = build_config(args=dummy_args, config_file=config_path)
        self.cfg.use_gpu = (device.startswith('cuda') and torch.cuda.is_available())

        # Determine device
        if torch.cuda.is_available() and device.startswith('cuda'):
            self.device = device
        else:
            self.device = 'cpu'
            print("[PersonEmbedder] CUDA not available, using CPU")

        # Load model weights path
        model_path = weights_path if weights_path else self.cfg.model.load_weights

        # Initialize feature extractor
        print(f"[PersonEmbedder] Loading model from: {model_path}")
        self.extractor = FeatureExtractor(
            self.cfg,
            model_path=model_path,
            device=self.device,
            num_classes=1,
            verbose=True
        )

        # Configure CUDA settings for thread safety
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.backends.cudnn.benchmark = False  # Disable for deterministic behavior
            torch.backends.cudnn.deterministic = True

        print(f"[PersonEmbedder] Initialized successfully on {self.device}")
        print(f"[PersonEmbedder] Thread-safe inference enabled")

    def extract_test_embeddings(self, model_output: tuple) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract and concatenate test embeddings from model output.

        Args:
            model_output: Tuple containing embeddings_dict, visibility_dict, and other outputs

        Returns:
            Tuple of (embeddings, visibility_scores)
        """
        embeddings_dict, visibility_dict, _, _, _, _ = model_output

        embeddings_list = []
        visibility_scores_list = []

        for test_emb in self.cfg.model.bpbreid.test_embeddings:
            embds = embeddings_dict[test_emb]
            embeddings_list.append(embds if len(embds.shape) == 3 else embds.unsqueeze(1))

            vis_key = test_emb
            if test_emb in bn_correspondants:
                vis_key = bn_correspondants[test_emb]

            vis_scores = visibility_dict[vis_key]
            visibility_scores_list.append(vis_scores if len(vis_scores.shape) == 2 else vis_scores.unsqueeze(1))

        embeddings = torch.cat(embeddings_list, dim=1)
        visibility_scores = torch.cat(visibility_scores_list, dim=1)

        return embeddings, visibility_scores

    def extract_embedding(self, person_crop: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract embedding from a single person crop with thread-safe inference.

        Args:
            person_crop: RGB numpy array of person crop (H, W, 3)

        Returns:
            Embedding as numpy array (1D vector) or None if extraction fails
        """
        try:
            if person_crop is None or person_crop.size == 0:
                print("[PersonEmbedder] Invalid person crop")
                return None

            # Thread-safe inference with explicit CUDA synchronization
            with self._inference_lock:
                # Ensure CUDA synchronization before inference
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                # Extract features using the feature extractor
                # Pass as a list to avoid mask requirement
                model_output = self.extractor([person_crop])

                # Extract embeddings and visibility scores
                embeddings, _ = self.extract_test_embeddings(model_output)

                # Ensure CUDA synchronization after inference
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

            # Convert to numpy and flatten (get first element since we passed a list)
            embedding_np = embeddings[0].cpu().numpy().flatten()

            return embedding_np

        except Exception as e:
            print(f"[PersonEmbedder] Error extracting embedding: {e}")
            return None

    def extract_embeddings_batch(self, person_crops: list) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Extract embeddings from multiple person crops in batch with thread-safe inference.

        Args:
            person_crops: List of RGB numpy arrays (each H, W, 3)

        Returns:
            Tuple of (embeddings array, visibility_scores array) or (None, None) if extraction fails
        """
        try:
            if not person_crops or len(person_crops) == 0:
                print("[PersonEmbedder] Empty person crops list")
                return None, None

            # Thread-safe inference with explicit CUDA synchronization
            with self._inference_lock:
                # Ensure CUDA synchronization before inference
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                # Extract features using the feature extractor
                model_output = self.extractor(person_crops)

                # Extract embeddings and visibility scores
                embeddings, visibility_scores = self.extract_test_embeddings(model_output)

                # Ensure CUDA synchronization after inference
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

            # Convert to numpy
            embeddings_np = embeddings.cpu().numpy()
            visibility_np = visibility_scores.cpu().numpy()

            return embeddings_np, visibility_np

        except Exception as e:
            print(f"[PersonEmbedder] Error extracting batch embeddings: {e}")
            return None, None

    def warmup(self):
        """
        Warmup the model with a dummy input to initialize CUDA contexts.
        """
        try:
            print("[PersonEmbedder] Warming up model...")
            dummy_crop = np.random.randint(0, 255, (256, 128, 3), dtype=np.uint8)
            _ = self.extract_embedding(dummy_crop)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            print("[PersonEmbedder] Warmup complete")
        except Exception as e:
            print(f"[PersonEmbedder] Warmup failed: {e}")
