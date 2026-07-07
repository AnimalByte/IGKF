from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import llama_cpp
from llama_cpp import Llama
from transformers import AutoTokenizer


class QwenNonThinkingError(RuntimeError):
    pass


class Qwen3LocalGenerator:
    """Qwen3-8B llama.cpp generator with verified non-thinking chat templating."""

    def __init__(self, config: Dict[str, Any], load_model: bool = True):
        self.config = config
        self.model_path = Path(config["local_model_path"])
        self.tokenizer_id = config["tokenizer_identifier"]
        self.generation = config["generation"]
        self.thinking_mode = config["thinking_mode"]
        self.thinking_disable_mechanism = config["thinking_disable_mechanism"]
        if self.thinking_mode is not False:
            raise QwenNonThinkingError("Qwen3 thinking_mode must be false.")

        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_id, trust_remote_code=True)
        self._validate_non_thinking_template()
        self.llm: Optional[Llama] = None
        if load_model:
            self.load_model()

    def load_model(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Qwen3 local model file not found: {self.model_path}. "
                "Download the exact configured GGUF before generation."
            )
        self.llm = Llama(
            model_path=str(self.model_path),
            n_gpu_layers=int(self.config["gpu_layers"]),
            n_ctx=int(self.config["context_length"]),
            n_batch=int(self.config["batch_size"]),
            n_ubatch=int(self.config["micro_batch_size"]),
            flash_attn=bool(self.config["flash_attention"]),
            seed=int(self.config["seed"]),
            verbose=False,
        )

    def _apply_template(self, messages: List[Dict[str, str]]) -> str:
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError as exc:
            raise QwenNonThinkingError(
                "Installed tokenizer/chat template does not support enable_thinking=False."
            ) from exc

    def _validate_non_thinking_template(self) -> None:
        messages = [
            {"role": "system", "content": "You are a concise assistant."},
            {"role": "user", "content": "Say hello."},
        ]
        prompt = self._apply_template(messages)
        if "enable_thinking" not in str(getattr(self.tokenizer, "chat_template", "")):
            raise QwenNonThinkingError("Qwen tokenizer chat template does not expose enable_thinking.")
        if "<think>" in prompt or "</think>" in prompt:
            if not re.search(r"<think>\s*</think>\s*$", prompt):
                raise QwenNonThinkingError(
                    "Qwen non-thinking prompt contains a non-empty or misplaced think block."
                )

    def build_prompt(self, system_prompt: str, user_prompt: str) -> str:
        return self._apply_template([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ])

    def count_prompt_tokens(self, prompt: str) -> int:
        if self.llm is not None:
            return len(self.llm.tokenize(prompt.encode("utf-8"), add_bos=False))
        return len(self.tokenizer.encode(prompt, add_special_tokens=False))

    def generate(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        if self.llm is None:
            self.load_model()
        assert self.llm is not None
        prompt = self.build_prompt(system_prompt, user_prompt)
        prompt_tokens = self.count_prompt_tokens(prompt)
        context_limit = int(self.config["context_length"])
        max_tokens = int(self.generation["max_output_tokens"])
        if prompt_tokens + max_tokens > context_limit:
            raise ValueError(
                f"Qwen prompt would exceed context window: prompt_tokens={prompt_tokens}, "
                f"max_output_tokens={max_tokens}, context_length={context_limit}"
            )
        raw = self.llm(
            prompt,
            max_tokens=max_tokens,
            temperature=float(self.generation["temperature"]),
            top_p=float(self.generation["top_p"]),
            top_k=int(self.generation["top_k"]),
            min_p=float(self.generation["min_p"]),
            repeat_penalty=float(self.generation["repeat_penalty"]),
            seed=int(self.config["seed"]),
            stream=False,
        )
        text = raw["choices"][0]["text"]
        output_tokens = len(self.llm.tokenize(text.encode("utf-8"), add_bos=False))
        if "<think>" in text or "</think>" in text:
            raise QwenNonThinkingError("Qwen output contained think tags despite non-thinking template.")
        return {
            "prompt": prompt,
            "raw_model_output": raw,
            "parsed_answer": text.strip(),
            "prompt_token_count": prompt_tokens,
            "output_token_count": output_tokens,
            "context_length": context_limit,
            "requested_output_tokens": max_tokens,
        }

    def metadata(self) -> Dict[str, Any]:
        model_sha256 = None
        if self.model_path.exists():
            h = hashlib.sha256()
            with self.model_path.open("rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(chunk)
            model_sha256 = h.hexdigest()
        return {
            "model_name": self.config["model_name"],
            "model_identifier": self.config["model_identifier"],
            "local_model_path": self.config["local_model_path"],
            "gguf_filename": self.config["gguf_filename"],
            "model_file_sha256": model_sha256,
            "quantization": self.config["quantization"],
            "context_length": self.config["context_length"],
            "gpu_layers": self.config["gpu_layers"],
            "batch_size": self.config["batch_size"],
            "micro_batch_size": self.config["micro_batch_size"],
            "flash_attention": self.config["flash_attention"],
            "generation": self.generation,
            "thinking_mode": False,
            "thinking_disable_mechanism": self.thinking_disable_mechanism,
            "inference_backend": "llama-cpp-python",
            "inference_backend_version": getattr(llama_cpp, "__version__", None),
        }
