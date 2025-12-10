"""
Embedding-based weight recognition.

Uses RepVGG embeddings and gallery search for weight classification.
"""
import json
import cv2
import numpy as np
import torch
import timm
from collections import defaultdict
from .config import EMB_MODEL_NAME, EMBEDDINGS_JSON, DIST_ACCEPT, DIST_MARGIN


class WeightRecognizer:
    """
    Embedding-based weight recognizer.

    Uses RepVGG model to extract embeddings and performs
    gallery search for weight classification.
    """

    def __init__(self, embeddings_path=EMBEDDINGS_JSON, model_name=EMB_MODEL_NAME,
                 device="cuda:0", use_fp16=True, use_prototypes=True):
        """
        Initialize weight recognizer.

        Args:
            embeddings_path: Path to embeddings JSON file
            model_name: RepVGG model name
            device: Device to run on
            use_fp16: Use half precision
            use_prototypes: Use prototype embeddings (averaged per class)
        """
        self.device = device
        self.use_fp16 = use_fp16

        print(f"[WEIGHT_RECOGNIZER] Initializing on {device}")

        # Build embedding model
        self.embedder = self.build_embedder(model_name)

        # Load gallery
        E_np, M = self.load_embeddings(embeddings_path)
        if use_prototypes:
            E_np, M = self.build_prototypes(E_np, M)

        self.gallery = torch.from_numpy(E_np).to(device, dtype=torch.float32)
        self.metadata = M

        print(f"[WEIGHT_RECOGNIZER] Loaded {len(self.metadata)} reference weights")

        # Warmup
        dummy = np.zeros((224, 224, 3), np.uint8)
        self.embed_crops([dummy])

    def build_embedder(self, model_name):
        """Build RepVGG embedding model."""
        m = timm.create_model(model_name, pretrained=True, num_classes=0,
                            global_pool="avg").to(self.device).eval()
        if self.use_fp16 and "cuda" in self.device:
            m.half()
        return m

    def load_embeddings(self, path):
        """
        Load reference embeddings from JSON.

        Returns:
            (embeddings_array, metadata_list)
        """
        with open(path, "r") as f:
            data = json.load(f)

        embs, meta = [], []
        label_keys = ["dumbbell", "dummbbell", "video", "label", "id", "name"]

        for rec in data:
            e = np.asarray(rec.get("embedding", None), np.float32)
            if e.ndim != 1 or not np.all(np.isfinite(e)):
                continue
            e = e / (np.linalg.norm(e) + 1e-8)
            embs.append(e)

            # Find label
            label = "unknown"
            for k in label_keys:
                if k in rec:
                    label = str(rec[k]).replace(".mp4", "")
                    break
            meta.append({"video": label})

        return np.stack(embs, 0), meta

    def build_prototypes(self, E, M):
        """
        Build prototype embeddings by averaging per class.

        Returns:
            (prototype_embeddings, prototype_metadata)
        """
        buckets = defaultdict(list)

        for e, m in zip(E, M):
            buckets[m["video"]].append(e)

        keys, embs = [], []
        for k, vs in buckets.items():
            keys.append(k)
            m = np.mean(np.stack(vs, 0), 0)
            m = m / (np.linalg.norm(m) + 1e-8)
            embs.append(m)

        return np.stack(embs, 0), [{"video": k} for k in keys]

    @torch.no_grad()
    def embed_crops(self, crops_bgr):
        """
        Generate embeddings for crops.

        Args:
            crops_bgr: List of BGR crop images

        Returns:
            Normalized embeddings matrix (N, feature_dim)
        """
        if not crops_bgr:
            return np.zeros((0, 0), np.float32)

        batch = []
        for c in crops_bgr:
            img = cv2.resize(c, (224, 224), interpolation=cv2.INTER_AREA)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            img[..., 0] = (img[..., 0] - 0.485) / 0.229
            img[..., 1] = (img[..., 1] - 0.456) / 0.224
            img[..., 2] = (img[..., 2] - 0.406) / 0.225
            batch.append(np.transpose(img, (2, 0, 1)))

        x = np.stack(batch, 0)
        dtype = torch.float16 if (self.use_fp16 and "cuda" in self.device) else torch.float32
        x_t = torch.from_numpy(x).to(self.device, dtype=dtype, non_blocking=True)
        z = self.embedder(x_t)
        z = torch.nn.functional.normalize(z, dim=1)
        return z.float().cpu().numpy()

    @torch.no_grad()
    def search_gallery(self, Q):
        """
        Search gallery for best match.

        Args:
            Q: Query embeddings matrix

        Returns:
            (label, distance, margin)
        """
        if Q.shape[0] == 0:
            return "unknown", 999.0, 0.0

        Q_t = torch.from_numpy(Q).to(self.device, dtype=torch.float32, non_blocking=True)
        sims = Q_t @ self.gallery.T
        dist2 = 2.0 - 2.0 * sims

        # Get top-2 for margin calculation
        d2vals, idxs = torch.topk(dist2, k=min(2, dist2.shape[1]), dim=1, largest=False)

        # Find best across all crops
        best_idx = 0
        best_dist = 999.0
        best_margin = 0.0

        for i in range(Q.shape[0]):
            d1 = torch.sqrt(d2vals[i, 0]).item()
            d2 = torch.sqrt(d2vals[i, 1]).item() if d2vals.shape[1] > 1 else d1 + 1.0
            margin = d2 - d1

            if d1 < best_dist or (abs(d1 - best_dist) < 1e-6 and margin > best_margin):
                best_dist = d1
                best_margin = margin
                best_idx = idxs[i, 0].item()

        label = self.metadata[best_idx]["video"]
        return label, best_dist, best_margin

    def recognize(self, crops):
        """
        Recognize weight from crops.

        Args:
            crops: List of BGR crop images

        Returns:
            (label, confidence) or ("unknown", 0.0)
        """
        if not crops:
            return "unknown", 0.0

        # Generate embeddings
        Q = self.embed_crops(crops)
        if Q.size == 0:
            return "unknown", 0.0

        # Search gallery
        label, dist, margin = self.search_gallery(Q)

        # Check thresholds
        if dist <= DIST_ACCEPT and margin >= DIST_MARGIN:
            # Convert distance to confidence (cosine similarity)
            confidence = 1.0 - (dist ** 2) / 2.0
            return label, confidence

        return "unknown", 0.0
