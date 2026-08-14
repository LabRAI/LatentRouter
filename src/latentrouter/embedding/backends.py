from __future__ import annotations

import base64
import io
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize

from latentrouter.embedding.coconut_text_encoder import CoconutTextEncoder

TSV_IMAGE_PREFIX = "tsvref::"


class BaseEncoder(ABC):
    @abstractmethod
    def encode_texts(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def encode_image_sets(self, image_sets: list[list[str]]) -> np.ndarray:
        raise NotImplementedError


class HashingEncoder(BaseEncoder):
    def __init__(self, dim: int = 256):
        self.dim = dim
        self.text_vectorizer = HashingVectorizer(
            n_features=dim,
            alternate_sign=False,
            norm=None,
            analyzer="word",
        )
        self.image_vectorizer = HashingVectorizer(
            n_features=dim,
            alternate_sign=False,
            norm=None,
            analyzer="char_wb",
            ngram_range=(3, 5),
        )

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        matrix = self.text_vectorizer.transform(texts).astype(np.float32).toarray()
        return normalize(matrix, norm="l2", copy=False)

    def encode_image_sets(self, image_sets: list[list[str]]) -> np.ndarray:
        serialized = [" ".join(Path(path).name for path in image_paths) for image_paths in image_sets]
        matrix = self.image_vectorizer.transform(serialized).astype(np.float32).toarray()
        return normalize(matrix, norm="l2", copy=False)


class OpenClipEncoder(BaseEncoder):
    def __init__(
        self,
        model_name: str,
        pretrained: str,
        batch_size: int = 32,
        device: str = "cpu",
    ):
        try:
            import open_clip
            import torch
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError(
                "OpenCLIP backend requested but dependencies are missing. "
                "Install latentrouter with the '[vision]' extra."
            ) from exc

        self.open_clip = open_clip
        self.torch = torch
        self.Image = Image
        self.device = torch.device(device)
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
            device=self.device,
        )
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.batch_size = batch_size
        projection = getattr(self.model, "text_projection", None)
        self.dim = int(projection.shape[-1]) if projection is not None else 512
        self.tsv_cache: dict[Path, pd.DataFrame] = {}

    def _load_tsv_image(self, ref: str):
        if not ref.startswith(TSV_IMAGE_PREFIX):
            candidate = Path(ref)
            if not candidate.exists():
                return None
            return self.Image.open(candidate).convert("RGB")

        payload = ref.removeprefix(TSV_IMAGE_PREFIX)
        try:
            tsv_path_text, index_value = payload.split("::index::", 1)
        except ValueError:
            return None
        tsv_path = Path(tsv_path_text)
        if tsv_path not in self.tsv_cache:
            if len(self.tsv_cache) >= 2:
                self.tsv_cache.clear()
            frame = pd.read_csv(tsv_path, sep="\t", usecols=["index", "image"], dtype={"index": str, "image": str})
            frame["index"] = frame["index"].astype(str)
            self.tsv_cache[tsv_path] = frame.set_index("index")
        frame = self.tsv_cache[tsv_path]
        if index_value not in frame.index:
            return None
        encoded = str(frame.at[index_value, "image"])
        try:
            image_bytes = base64.b64decode(encoded)
        except Exception:
            return None
        try:
            return self.Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception:
            # VL-RouterBench contains a small number of malformed or empty
            # TSV payloads. Treat them like missing images so one bad record
            # does not abort the whole feature extraction pass.
            return None

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        vectors: list[np.ndarray] = []
        with self.torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                batch = texts[start : start + self.batch_size]
                tokens = self.tokenizer(batch).to(self.device)
                encoded = self.model.encode_text(tokens)
                encoded = encoded / encoded.norm(dim=-1, keepdim=True)
                vectors.append(encoded.cpu().numpy().astype(np.float32))
        return np.concatenate(vectors, axis=0) if vectors else np.zeros((0, self.dim), dtype=np.float32)

    def encode_image_sets(self, image_sets: list[list[str]]) -> np.ndarray:
        zero = np.zeros((self.dim,), dtype=np.float32)
        outputs: list[np.ndarray] = []
        with self.torch.no_grad():
            for paths in image_sets:
                embeddings: list[np.ndarray] = []
                for path in paths:
                    image = self._load_tsv_image(path)
                    if image is None:
                        continue
                    tensor = self.preprocess(image).unsqueeze(0).to(self.device)
                    encoded = self.model.encode_image(tensor)
                    encoded = encoded / encoded.norm(dim=-1, keepdim=True)
                    embeddings.append(encoded.squeeze(0).cpu().numpy().astype(np.float32))
                if embeddings:
                    outputs.append(np.mean(np.stack(embeddings, axis=0), axis=0))
                else:
                    outputs.append(zero.copy())
        matrix = np.stack(outputs, axis=0) if outputs else np.zeros((0, self.dim), dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms


class OpenClipCoconutEncoder(BaseEncoder):
    def __init__(
        self,
        *,
        model_name: str,
        pretrained: str,
        batch_size: int = 32,
        device: str = "cpu",
        coconut_repo_path: str | None = None,
        coconut_model_id: str = "openai-community/gpt2",
        coconut_checkpoint: str | None = None,
        coconut_checkpoint_filename: str = "pytorch_model.bin",
        coconut_latent_tokens: int = 2,
        coconut_max_length: int = 256,
    ):
        if not coconut_repo_path:
            raise RuntimeError(
                "openclip_coconut requires features.coconut_repo_path to point to a clone of "
                "facebookresearch/coconut."
            )
        self.image_encoder = OpenClipEncoder(
            model_name=model_name,
            pretrained=pretrained,
            batch_size=batch_size,
            device=device,
        )
        self.text_encoder = CoconutTextEncoder(
            repo_path=coconut_repo_path,
            model_id=coconut_model_id,
            checkpoint_source=coconut_checkpoint,
            checkpoint_filename=coconut_checkpoint_filename,
            latent_tokens=coconut_latent_tokens,
            max_length=coconut_max_length,
            batch_size=max(1, min(batch_size, 8)),
            device=device,
        )
        self.dim = self.text_encoder.dim

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        return self.text_encoder.encode_texts(texts)

    def encode_image_sets(self, image_sets: list[list[str]]) -> np.ndarray:
        return self.image_encoder.encode_image_sets(image_sets)


def create_encoder(
    backend: str,
    hashing_dim: int = 256,
    openclip_model: str = "ViT-B-32",
    openclip_pretrained: str = "laion2b_s34b_b79k",
    coconut_repo_path: str | None = None,
    coconut_model_id: str = "openai-community/gpt2",
    coconut_checkpoint: str | None = None,
    coconut_checkpoint_filename: str = "pytorch_model.bin",
    coconut_latent_tokens: int = 2,
    coconut_max_length: int = 256,
    batch_size: int = 32,
    device: str = "cpu",
) -> BaseEncoder:
    backend_normalized = backend.lower()
    if backend_normalized == "hashing":
        return HashingEncoder(dim=hashing_dim)
    if backend_normalized == "openclip":
        return OpenClipEncoder(
            model_name=openclip_model,
            pretrained=openclip_pretrained,
            batch_size=batch_size,
            device=device,
        )
    if backend_normalized == "openclip_coconut":
        return OpenClipCoconutEncoder(
            model_name=openclip_model,
            pretrained=openclip_pretrained,
            batch_size=batch_size,
            device=device,
            coconut_repo_path=coconut_repo_path,
            coconut_model_id=coconut_model_id,
            coconut_checkpoint=coconut_checkpoint,
            coconut_checkpoint_filename=coconut_checkpoint_filename,
            coconut_latent_tokens=coconut_latent_tokens,
            coconut_max_length=coconut_max_length,
        )
    raise ValueError(f"Unknown embedding backend: {backend}")
