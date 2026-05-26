from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


def _load_coconut_module(repo_path: str | Path):
    candidate = Path(repo_path).expanduser().resolve()
    module_path = candidate / "coconut.py"
    if not module_path.exists():
        raise RuntimeError(
            f"COCONUT repo path does not contain coconut.py: {candidate}. "
            "Clone facebookresearch/coconut and set features.coconut_repo_path."
        )
    module_name = "latentrouter_external_coconut"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load COCONUT module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _resolve_checkpoint_file(checkpoint_source: str | None, filename: str) -> Path | None:
    if checkpoint_source is None:
        return None
    candidate = Path(checkpoint_source).expanduser()
    if candidate.exists():
        if candidate.is_dir():
            checkpoint_path = candidate / filename
            if not checkpoint_path.exists():
                raise RuntimeError(f"COCONUT checkpoint file not found: {checkpoint_path}")
            return checkpoint_path
        return candidate

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "Downloading a COCONUT checkpoint requires huggingface-hub. "
            "Install latentrouter with the '[hf]' extra."
        ) from exc

    return Path(hf_hub_download(repo_id=checkpoint_source, filename=filename, repo_type="model"))


class CoconutTextEncoder:
    def __init__(
        self,
        *,
        repo_path: str | Path,
        model_id: str,
        checkpoint_source: str | None,
        checkpoint_filename: str,
        latent_tokens: int,
        max_length: int,
        batch_size: int,
        device: str,
    ):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Real COCONUT text encoding requires transformers and torch. "
                "Install latentrouter with the '[mllm]' extra."
            ) from exc

        coconut_module = _load_coconut_module(repo_path)
        Coconut = getattr(coconut_module, "Coconut", None)
        if Coconut is None:
            raise RuntimeError(f"COCONUT module at {repo_path} does not expose Coconut")

        class LegacyCacheCompatibleCausalLM(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self._model = model

            def forward(self, *args, **kwargs):
                past_key_values = kwargs.get("past_key_values")
                if isinstance(past_key_values, list):
                    kwargs["past_key_values"] = tuple(past_key_values)
                return self._model(*args, **kwargs)

            def __getattr__(self, name: str):
                if name == "_model":
                    return super().__getattr__(name)
                try:
                    return super().__getattr__(name)
                except AttributeError:
                    return getattr(self._model, name)

        self.torch = torch
        self.device = torch.device(device)
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)
        self.latent_tokens = int(latent_tokens)
        self.model_id = str(model_id)
        self.checkpoint_source = checkpoint_source
        self.repo_path = str(Path(repo_path).expanduser().resolve())

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.add_tokens("<|start-latent|>")
        tokenizer.add_tokens("<|end-latent|>")
        tokenizer.add_tokens("<|latent|>")

        base_model = AutoModelForCausalLM.from_pretrained(model_id)
        base_model.resize_token_embeddings(len(tokenizer))

        latent_token = "<|latent|>"
        start_latent = "<|start-latent|>"
        end_latent = "<|end-latent|>"
        latent_id = tokenizer.convert_tokens_to_ids(latent_token)
        start_id = tokenizer.convert_tokens_to_ids(start_latent)
        end_id = tokenizer.convert_tokens_to_ids(end_latent)
        target_token_id = tokenizer.convert_tokens_to_ids("<<")
        if target_token_id is None or target_token_id < 0:
            target_token_id = tokenizer.eos_token_id
        embeddings = base_model.get_input_embeddings()
        for token_id in [latent_id, start_id, end_id]:
            embeddings.weight.data[token_id] = embeddings.weight.data[target_token_id]
            if hasattr(base_model, "lm_head"):
                base_model.lm_head.weight.data[token_id] = base_model.lm_head.weight.data[target_token_id]

        model = Coconut(base_model, latent_id, start_id, end_id, tokenizer.eos_token_id)
        checkpoint_path = _resolve_checkpoint_file(checkpoint_source, checkpoint_filename)
        if checkpoint_path is not None:
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            model.load_state_dict(state_dict, strict=False)
        model.base_causallm = LegacyCacheCompatibleCausalLM(model.base_causallm)

        model = model.to(self.device)
        model.eval()

        hidden_size = getattr(base_model.config, "n_embd", None) or getattr(base_model.config, "hidden_size", None)
        if hidden_size is None:
            raise RuntimeError("Unable to infer hidden size for the loaded COCONUT base model.")

        self.tokenizer = tokenizer
        self.model = model
        self.dim = int(hidden_size)
        self.latent_id = int(latent_id)
        self.start_id = int(start_id)
        self.end_id = int(end_id)
        self.pad_token_id = int(tokenizer.pad_token_id)

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        reserved_tokens = self.latent_tokens + 2
        encoded_sequences: list[list[int]] = []
        for text in texts:
            question_tokens = self.tokenizer.encode(f"{text.rstrip()}\n", add_special_tokens=False)
            max_question_tokens = max(self.max_length - reserved_tokens, 0)
            if len(question_tokens) > max_question_tokens:
                question_tokens = question_tokens[:max_question_tokens]
            encoded_sequences.append(
                question_tokens + [self.start_id] + [self.latent_id] * self.latent_tokens + [self.end_id]
            )
        outputs: list[np.ndarray] = []
        with self.torch.no_grad():
            for start in range(0, len(encoded_sequences), self.batch_size):
                batch = encoded_sequences[start : start + self.batch_size]
                earliest_latent_positions = [seq.index(self.latent_id) for seq in batch]
                latest_earliest_latent = max(earliest_latent_positions) if earliest_latent_positions else 0
                left_padded_batch = []
                position_id_rows = []
                for seq, earliest_latent in zip(batch, earliest_latent_positions, strict=False):
                    left_pad = latest_earliest_latent - earliest_latent
                    left_padded = [self.pad_token_id] * left_pad + seq
                    left_padded_batch.append(left_padded)
                    position_id_rows.append([0] * left_pad + list(range(len(seq))))

                max_seq_len = max(len(seq) for seq in left_padded_batch) if left_padded_batch else 0
                input_id_rows = []
                attention_rows = []
                full_position_rows = []
                for seq, position_ids in zip(left_padded_batch, position_id_rows, strict=False):
                    pad_count = max_seq_len - len(seq)
                    input_id_rows.append(seq + [self.pad_token_id] * pad_count)
                    attention_rows.append([0 if token_id == self.pad_token_id else 1 for token_id in seq] + [0] * pad_count)
                    full_position_rows.append(position_ids + [0] * pad_count)
                input_ids = self.torch.tensor(input_id_rows, dtype=self.torch.long, device=self.device)
                attention_mask = self.torch.tensor(attention_rows, dtype=self.torch.long, device=self.device)
                labels = input_ids.clone()
                position_ids = self.torch.tensor(full_position_rows, dtype=self.torch.long, device=self.device)
                coconut_outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    position_ids=position_ids,
                )
                embeds = coconut_outputs.inputs_embeds
                latent_mask = input_ids == self.latent_id
                for row_idx in range(input_ids.shape[0]):
                    if latent_mask[row_idx].any():
                        vector = embeds[row_idx][latent_mask[row_idx]].mean(dim=0)
                    else:
                        last_index = int(attention_mask[row_idx].sum().item()) - 1
                        last_index = max(last_index, 0)
                        vector = embeds[row_idx, last_index]
                    vector = vector / vector.norm(p=2).clamp_min(1e-6)
                    outputs.append(vector.detach().cpu().numpy().astype(np.float32, copy=False))
        return np.stack(outputs, axis=0) if outputs else np.zeros((0, self.dim), dtype=np.float32)
