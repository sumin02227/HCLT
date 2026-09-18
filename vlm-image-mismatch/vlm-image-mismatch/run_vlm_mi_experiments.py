#!/usr/bin/env python3
"""Unified causal-intervention runner for LLaVA-OneVision and Qwen VLMs.

The runner is intentionally configuration driven so the model is loaded once and
multiple expensive experiments can be executed sequentially.  It supports:

* fixed head-output scaling with prefill/decode/first-token scopes;
* MC option-label permutation controls;
* paired residual/head activation patching sweeps;
* paired/same-answer/different-answer/random source controls;
* sender-head -> receiver-residual bridge tracing;
* sample-adaptive VHD head selection;
* paper-faithful L22 VHR selection and direct head-output reinforcement;
* VCD-style MC logit contrast with plausibility constraints;
* full-option semantic likelihood ranking with calibrated selective contrast;
* detector-gated label-logit, permutation, and evidence-first correction;
* option-permutation ensembles and disjoint label-prior calibration;
* image-sensitive, option-permutation-invariant head reinforcement;
* late-to-early residual back-patching;
* activation export for later SAE/probe analysis.

Model loading, decoder lookup, and attention-head counting support
LLaVA-OneVision, Qwen2.5-VL, and Qwen3-VL (dense and MoE) checkpoints.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import os
import random
import re
from collections import OrderedDict
from contextlib import ExitStack, nullcontext
from copy import deepcopy
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageFilter, ImageOps
from tqdm.auto import tqdm
import transformers
from transformers import AutoProcessor


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
DEFAULT_MAX_NEW_TOKENS = {"MC": 8, "SA": 64, "LA": 384}
OPTION_LABELS = list("123456789") + list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

DEFAULT_PROMPTS = {
    "system": "질문에 정확하고 간결하게 답하세요.",
    # 출력 형식 규칙을 명시한 대회용 프롬프트. 복수 정답은 "/" 로 구분한다.
    "MC_1": (
        "다음 객관식 문제에 답하세요.\n\n"
        "문제: {question}\n\n"
        "선택지:\n{options}\n\n"
        "출력 규칙:\n"
        "- 정답 선택지 번호만 출력하세요.\n"
        "- 단일 정답은 숫자 하나만 출력하세요. 예: 2\n"
        "- 복수 정답은 번호를 오름차순으로 /로 구분하세요. 예: 1/3\n"
        "- 선택지 문구, 설명, 근거, 괄호, 접두어, 마침표를 출력하지 마세요.\n"
    ),
    # Candidate-content ranking deliberately hides the option list and its
    # numeric labels.  Each semantic option is appended as a continuation and
    # scored token by token, following the SEED-Bench answer-ranking protocol.
    "MC_CONTENT_1": (
        "이미지를 참고하여 다음 질문의 정답을 완성하세요.\n"
        "질문: {question}\n답변:"
    ),
    "MC_EVIDENCE_1": (
        "다음 질문에 답하는 데 필요한 이미지 속 사실만 1~3문장으로 기술하세요.\n"
        "선택지 번호나 정답을 추측하지 말고, 이미지에서 직접 확인되는 근거만 쓰세요.\n"
        "질문: {question}\n이미지 근거:"
    ),
    "SA_1": (
        "다음 단답형 문제에 답하세요.\n\n"
        "문제: {question}\n\n"
        "출력 규칙:\n"
        "- 질문에서 요구하는 최종 답만 짧게 출력하세요.\n"
        "- 설명, 풀이 과정, 서론 또는 결론을 쓰지 마세요.\n"
        "- 요구된 단위, 수치, 날짜, 인명, 지명 및 표기 형식을 정확히 지키세요.\n"
        "- 여러 항목을 요구할 때만 / 로 구분하세요 예: 꿈/희망.\n"
        "- '음절'은 한글 한 글자 단위입니다. 예: '대한민국'은 4음절입니다.\n"
        "- '어절'은 띄어쓰기로 구분되는 단위입니다. 예: '대한 민국'은 2어절이고, "
        "'대한민국'은 1어절입니다.\n"
        "- '몇 음절'이면 띄어쓰기를 제외한 한글 글자 수를 맞추세요.\n"
        "- '몇 어절'이면 띄어쓰기 기준의 덩어리 수를 맞추세요.\n"
        "- 음절과 어절을 서로 혼동하지 마세요.\n"
    ),
    "LA_1": (
        "이미지가 제공된 경우 이미지의 구체적인 내용을 반드시 참고하여 "
        "다음 서술형 문제에 답하세요.\n\n"
        "문제: {question}\n\n"
        "출력 규칙:\n"
        "- 질문의 모든 요구사항에 빠짐없이 답하세요.\n"
        "- 이미지에서 확인되는 대상, 글자, 배치, 행동 및 관계를 근거로 활용하세요.\n"
        "- 필요한 경우 관련 문화·역사·사회적 맥락을 결합하세요.\n"
        "- 이미지에서 확인할 수 없는 내용을 본 것처럼 단정하지 마세요.\n"
        "- 별도의 '분석', '근거', '정답' 제목 없이 완결된 한국어 답변만 출력하세요.\n"
        "- 불필요한 메타 발언이나 풀이 과정은 쓰지 마세요.\n"
    ),
}


PROMPT = dict(DEFAULT_PROMPTS)


def stable_hash(value: Any, length: int = 12) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:length]


def safe_name(value: str) -> str:
    value = re.sub(r"[^0-9A-Za-z_.-]+", "_", value.strip())
    return value.strip("._") or "experiment"


def norm_condition(value: Any) -> str:
    text = str(value or "full").strip().lower()
    text = re.sub(r"[\s/]+", "_", text)
    text = text.replace("-", "_")
    aliases = {
        "original": "full",
        "full_image": "full",
        "text": "text_only",
        "textonly": "text_only",
        "no_image": "text_only",
        "none": "text_only",
        "shuffled_image": "shuffled",
        "shuffle": "shuffled",
        "blurred_image": "blurred",
        "blur": "blurred",
    }
    return aliases.get(text, text)


def norm_form(value: Any) -> str:
    text = str(value or "").strip().upper().replace("-", "_")
    aliases = {
        "MULTIPLE_CHOICE": "MC",
        "SHORT_ANSWER": "SA",
        "LONG_ANSWER": "LA",
    }
    return aliases.get(text, text)


def expand_path(value: str | os.PathLike[str], base_dir: Path | None = None) -> Path:
    expanded = os.path.expandvars(os.path.expanduser(str(value)))
    path = Path(expanded)
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    return path.resolve()


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    return value


def parse_maybe_list(value: Any) -> list[str]:
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, tuple):
        return [str(item) for item in value]
    text = str(value).strip()
    for loader in (json.loads, ast.literal_eval):
        try:
            parsed = loader(text)
        except (ValueError, SyntaxError, json.JSONDecodeError):
            continue
        if isinstance(parsed, (list, tuple)):
            return [str(item) for item in parsed]
    if "\n" in text:
        return [part.strip() for part in text.splitlines() if part.strip()]
    if "|||" in text:
        return [part.strip() for part in text.split("|||") if part.strip()]
    return [text]


def strip_option_label(option: Any) -> str:
    """Remove a leading MC symbol while preserving the semantic answer text."""
    return re.sub(
        r"^\s*(?:\(?[0-9]+\)?|\(?[A-Za-z]\)?)[.):_-]\s*",
        "",
        str(option),
    ).strip()


def js_divergence(values_a: Sequence[float], values_b: Sequence[float]) -> float:
    """Jensen-Shannon divergence after treating arbitrary scores as logits."""
    a = torch.softmax(torch.tensor(list(values_a), dtype=torch.float32), dim=0)
    b = torch.softmax(torch.tensor(list(values_b), dtype=torch.float32), dim=0)
    midpoint = 0.5 * (a + b)
    kl_a = (a * (torch.log(a.clamp_min(1e-12)) - torch.log(midpoint))).sum()
    kl_b = (b * (torch.log(b.clamp_min(1e-12)) - torch.log(midpoint))).sum()
    return float((0.5 * (kl_a + kl_b)).item())


def canonical_row(raw: Mapping[str, Any], index: int) -> dict[str, Any]:
    metadata = raw.get("metadata") if isinstance(raw.get("metadata"), Mapping) else {}
    model_input = (
        raw.get("model_input") if isinstance(raw.get("model_input"), Mapping) else {}
    )
    model_output = (
        raw.get("model_output") if isinstance(raw.get("model_output"), Mapping) else {}
    )

    question_id = (
        raw.get("question_id")
        or metadata.get("question_id")
        or raw.get("id")
        or f"row_{index:06d}"
    )
    form = norm_form(raw.get("question_form") or metadata.get("question_form"))
    question = raw.get("question") or model_input.get("question") or ""
    options = parse_maybe_list(
        raw.get("options_json")
        or raw.get("options")
        or model_input.get("options")
    )
    reference = (
        raw.get("reference")
        or raw.get("answer")
        or model_output.get("answer")
        or ""
    )
    condition = norm_condition(raw.get("condition") or "full")
    image_variant = norm_condition(
        raw.get("input_image_variant")
        or raw.get("image_variant")
        or condition
    )
    if condition == "text_only":
        image_variant = "none"
    elif image_variant == "full":
        image_variant = "original"

    image_name = (
        raw.get("input_image_name")
        or raw.get("image_name")
        or model_input.get("image_name")
        or ""
    )
    row = dict(raw)
    row.update(
        {
            "question_id": str(question_id),
            "question_form": form,
            "question": str(question),
            "options_json": json.dumps(options, ensure_ascii=False),
            "reference": str(reference),
            "answer": str(reference),
            "condition": condition,
            "input_image_variant": image_variant,
            "input_image_name": str(image_name),
            "split": str(raw.get("split") or metadata.get("split") or ""),
            "task_type": str(
                raw.get("task_type") or metadata.get("task_type") or ""
            ),
            "corpus_name": str(
                raw.get("corpus_name") or metadata.get("corpus_name") or ""
            ),
            "_row_index": index,
        }
    )
    row["sample_uid"] = "::".join(
        [
            row["question_id"],
            row["condition"],
            row["input_image_variant"],
            row["input_image_name"],
            str(index),
        ]
    )
    return row


def read_input(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            raw_rows = list(csv.DictReader(stream))
    elif suffix == ".jsonl":
        raw_rows = []
        with path.open("r", encoding="utf-8-sig") as stream:
            for line in stream:
                if line.strip():
                    raw_rows.append(json.loads(line))
    elif suffix == ".json":
        with path.open("r", encoding="utf-8-sig") as stream:
            raw_rows = json.load(stream)
        if not isinstance(raw_rows, list):
            raise ValueError("JSON input must contain a top-level list")
    else:
        raise ValueError(f"Unsupported input extension: {suffix}")
    return [canonical_row(row, index) for index, row in enumerate(raw_rows)]


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            cooked = {}
            for key in fieldnames:
                value = jsonable(row.get(key, ""))
                if isinstance(value, (dict, list)):
                    value = json.dumps(value, ensure_ascii=False)
                cooked[key] = value
            writer.writerow(cooked)


def parse_heads(value: Any) -> list[tuple[int, int]]:
    if value is None or value == "":
        return []
    if isinstance(value, str):
        items: Iterable[Any] = value.split(",")
    else:
        items = value
    heads: list[tuple[int, int]] = []
    for item in items:
        if isinstance(item, str):
            layer, head = item.strip().split(":")
        else:
            layer, head = item
        heads.append((int(layer), int(head)))
    return list(dict.fromkeys(heads))


MODEL_FAMILIES = {
    "auto",
    "llava_onevision",
    "qwen2_5_vl",
    "qwen3_vl",
    "qwen3_vl_moe",
}


def normalize_model_family(value: Any) -> str:
    family = str(value or "auto").strip().lower().replace("-", "_").replace(".", "_")
    aliases = {
        "llava": "llava_onevision",
        "llava_one_vision": "llava_onevision",
        "qwen25_vl": "qwen2_5_vl",
        "qwen2_5": "qwen2_5_vl",
        "qwen3": "qwen3_vl",
        "qwen3vl": "qwen3_vl",
    }
    family = aliases.get(family, family)
    if family not in MODEL_FAMILIES:
        raise ValueError(
            f"Unsupported model family {value!r}; choose one of {sorted(MODEL_FAMILIES)}"
        )
    return family


def infer_model_family(model_id: str, requested: Any = "auto") -> str:
    requested_family = normalize_model_family(requested)
    if requested_family != "auto":
        return requested_family
    name = model_id.lower().replace("-", "_").replace(".", "_")
    if "qwen3" in name and ("moe" in name or re.search(r"\d+b_a\d+b", name)):
        return "qwen3_vl_moe"
    if "qwen3" in name:
        return "qwen3_vl"
    if "qwen2_5" in name or "qwen25" in name:
        return "qwen2_5_vl"
    return "llava_onevision"


def load_vlm_model(
    model_id: str,
    family: str,
    *,
    dtype: torch.dtype,
    attn_implementation: str,
    device_map: Any,
    trust_remote_code: bool,
) -> torch.nn.Module:
    class_names = {
        "llava_onevision": ("LlavaOnevisionForConditionalGeneration",),
        "qwen2_5_vl": ("Qwen2_5_VLForConditionalGeneration",),
        "qwen3_vl": ("Qwen3VLForConditionalGeneration",),
        "qwen3_vl_moe": ("Qwen3VLMoeForConditionalGeneration",),
    }
    family = normalize_model_family(family)
    if family == "auto":
        family = infer_model_family(model_id)
    class_name = class_names[family][0]
    model_class = getattr(transformers, class_name, None)
    if model_class is None:
        raise RuntimeError(
            f"Installed transformers {transformers.__version__} does not provide "
            f"{class_name}, required for {family}. Upgrade transformers and retry."
        )
    return model_class.from_pretrained(
        model_id,
        torch_dtype=dtype,
        attn_implementation=attn_implementation,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
    ).eval()


def decoder_layers(model: torch.nn.Module) -> tuple[Any, str]:
    candidates = (
        "model.language_model.layers",
        "model.language_model.model.layers",
        "language_model.model.layers",
        "language_model.layers",
        "model.model.layers",
        "model.layers",
    )
    for path in candidates:
        obj: Any = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
        except AttributeError:
            continue
        if hasattr(obj, "__len__") and len(obj):
            return obj, path
    raise RuntimeError("Could not locate decoder layers")


def infer_num_attention_heads(model: torch.nn.Module, layers: Any) -> int:
    """Return query-head count (not KV-head count) for pre-o_proj slicing."""
    candidates: list[Any] = []
    if len(layers):
        attention = getattr(layers[0], "self_attn", None)
        if attention is not None:
            candidates.extend(
                [
                    getattr(attention, "num_heads", None),
                    getattr(attention, "num_attention_heads", None),
                    getattr(getattr(attention, "config", None), "num_attention_heads", None),
                ]
            )
    config = getattr(model, "config", None)
    for nested_name in (None, "text_config", "language_config"):
        nested = config if nested_name is None else getattr(config, nested_name, None)
        candidates.append(getattr(nested, "num_attention_heads", None))
    language_model = getattr(getattr(model, "model", None), "language_model", None)
    candidates.append(
        getattr(getattr(language_model, "config", None), "num_attention_heads", None)
    )
    for value in candidates:
        if isinstance(value, int) and value > 0:
            return value
    raise RuntimeError(
        "Could not infer the language model's query-head count. Set model.num_heads "
        "in the JSON config or pass --num-heads for an embedded preset."
    )


def attention_output_projection(layer: torch.nn.Module) -> torch.nn.Module:
    for path in ("self_attn.o_proj", "self_attn.out_proj", "attention.o_proj"):
        obj: Any = layer
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
        except AttributeError:
            continue
        return obj
    raise RuntimeError("Could not locate attention output projection")


def hidden_from_output(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    raise RuntimeError(f"Unsupported layer output type: {type(output)!r}")


def replace_hidden_output(output: Any, hidden: torch.Tensor) -> Any:
    if isinstance(output, torch.Tensor):
        return hidden
    if isinstance(output, tuple):
        return (hidden,) + tuple(output[1:])
    if isinstance(output, list):
        return [hidden] + list(output[1:])
    raise RuntimeError(f"Unsupported layer output type: {type(output)!r}")


class ImageStore:
    def __init__(self, roots: Mapping[str, Path]):
        self.roots = {norm_condition(key): Path(value) for key, value in roots.items()}
        if "full" in self.roots and "original" not in self.roots:
            self.roots["original"] = self.roots["full"]
        self.indices: dict[str, dict[str, Path]] = {}

    def _index(self, root_key: str) -> dict[str, Path]:
        if root_key in self.indices:
            return self.indices[root_key]
        root = self.roots.get(root_key)
        if root is None or not root.exists():
            self.indices[root_key] = {}
            return self.indices[root_key]
        index: dict[str, Path] = {}
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                index.setdefault(path.name, path)
                try:
                    relative = path.relative_to(root).as_posix()
                    index.setdefault(relative, path)
                except ValueError:
                    pass
        self.indices[root_key] = index
        return index

    def resolve(self, row: Mapping[str, Any]) -> Image.Image | None:
        variant = norm_condition(row.get("input_image_variant") or row.get("condition"))
        image_name = str(row.get("input_image_name") or "")
        if variant in {"none", "text_only"} or not image_name:
            return None
        root_key = variant
        if root_key not in self.roots:
            root_key = "original"
        image_path = self._index(root_key).get(image_name)
        if image_path is None and root_key != "original":
            image_path = self._index("original").get(image_name)
        if image_path is None:
            raise FileNotFoundError(
                f"Missing image: variant={variant}, name={image_name}, roots={self.roots}"
            )
        with Image.open(image_path) as image:
            resolved = ImageOps.exif_transpose(image).convert("RGB")
        corruption = row.get("_image_corruption")
        if isinstance(corruption, Mapping) and corruption.get("type") == "gaussian_noise":
            sigma = float(corruption.get("sigma", 0.2))
            seed = int(corruption.get("seed", 0))
            rng = np.random.default_rng(seed)
            pixels = np.asarray(resolved, dtype=np.float32)
            noisy = pixels + rng.normal(0.0, sigma * 255.0, pixels.shape)
            resolved = Image.fromarray(np.clip(noisy, 0, 255).astype(np.uint8), "RGB")
        if isinstance(corruption, Mapping) and corruption.get("type") == "gaussian_blur":
            radius = float(corruption.get("radius", 12.0))
            resolved = resolved.filter(ImageFilter.GaussianBlur(radius=radius))
        target_size = row.get("_target_image_size")
        if isinstance(target_size, (list, tuple)) and len(target_size) == 2:
            width, height = (int(target_size[0]), int(target_size[1]))
            if width <= 0 or height <= 0:
                raise ValueError(f"Invalid _target_image_size: {target_size}")
            resampling = getattr(Image, "Resampling", Image).BICUBIC
            fit_mode = str(row.get("_target_image_fit", "stretch")).lower()
            if fit_mode == "letterbox":
                resolved = ImageOps.pad(
                    resolved,
                    (width, height),
                    method=resampling,
                    color=(0, 0, 0),
                    centering=(0.5, 0.5),
                )
            elif fit_mode == "stretch" and resolved.size != (width, height):
                resolved = resolved.resize((width, height), resampling)
            elif fit_mode not in {"letterbox", "stretch"}:
                raise ValueError(f"Unsupported _target_image_fit: {fit_mode}")
        return resolved

    def counts(self) -> dict[str, int]:
        return {key: len(self._index(key)) for key in sorted(self.roots)}


def validate_prompt_key(question_form: str, prompt_key: str) -> None:
    if prompt_key not in PROMPT:
        available = sorted(key for key in PROMPT if key.startswith(f"{question_form}_"))
        raise ValueError(
            f"Unsupported {question_form} prompt key {prompt_key!r}; available={available}"
        )
    if not prompt_key.startswith(f"{question_form}_"):
        raise ValueError(f"Prompt {prompt_key!r} is not a {question_form} prompt")


def build_prompt(
    prompt_key: str,
    question_form: str,
    question: str,
    options: Sequence[str] | None = None,
) -> str:
    validate_prompt_key(question_form, prompt_key)
    rendered_options = list(options or [])
    if question_form == "MC":
        rendered_options = []
        for index, option in enumerate(options or []):
            # Re-label after option permutation; avoid stale prefixes such as
            # "3. ..." moving into slot 1.
            option_text = re.sub(
                r"^\s*(?:\(?[0-9]+\)?|\(?[A-Za-z]\)?)[.):_-]\s*",
                "",
                str(option),
            )
            rendered_options.append(f"{OPTION_LABELS[index]}. {option_text}")
    return PROMPT[prompt_key].format(
        question=str(question or "").strip(),
        options="\n".join(str(option) for option in rendered_options),
    )


def find_subsequence(haystack: Sequence[int], needle: Sequence[int]) -> tuple[int, int] | None:
    if not needle or len(needle) > len(haystack):
        return None
    found: tuple[int, int] | None = None
    for start in range(len(haystack) - len(needle) + 1):
        if list(haystack[start : start + len(needle)]) == list(needle):
            found = (start, start + len(needle))
    return found


@dataclass
class PreparedInput:
    row: dict[str, Any]
    inputs: dict[str, torch.Tensor]
    masks: dict[str, torch.Tensor]
    prompt_text: str


def collect_image_token_ids(model: torch.nn.Module, processor: Any) -> set[int]:
    token_ids: set[int] = set()
    configs = [getattr(model, "config", None)]
    for config_name in ("text_config", "language_config"):
        model_config = getattr(getattr(model, "config", None), config_name, None)
        if model_config is not None:
            configs.append(model_config)
    for config in configs:
        if config is None:
            continue
        for attr in ("image_token_index", "image_token_id", "video_token_index"):
            value = getattr(config, attr, None)
            if isinstance(value, int) and value >= 0:
                token_ids.add(value)
    tokenizer = processor.tokenizer
    unk_id = getattr(tokenizer, "unk_token_id", None)
    for token in ("<image>", "<|image_pad|>", "<|vision_start|>"):
        token_id = tokenizer.convert_tokens_to_ids(token)
        if isinstance(token_id, int) and token_id >= 0 and token_id != unk_id:
            token_ids.add(token_id)
    return token_ids


def move_inputs(
    inputs: Mapping[str, Any], device: torch.device, dtype: torch.dtype
) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            if torch.is_floating_point(value):
                moved[key] = value.to(device=device, dtype=dtype)
            else:
                moved[key] = value.to(device=device)
        else:
            moved[key] = value
    return moved


def make_conversation(
    row: Mapping[str, Any],
    image_store: ImageStore,
    prompt_keys: Mapping[str, str],
    prompt_key_override: str | None = None,
) -> tuple[list[dict[str, Any]], str]:
    form = norm_form(row.get("question_form"))
    options = parse_maybe_list(row.get("options_json"))
    prompt_key = prompt_key_override or prompt_keys[form]
    prompt_text = build_prompt(
        prompt_key, form, str(row.get("question") or ""), options
    )
    user_content: list[dict[str, Any]] = []
    image = image_store.resolve(row)
    if image is not None:
        user_content.append({"type": "image", "image": image})
    user_content.append({"type": "text", "text": prompt_text})
    conversation = [
        {
            "role": "system",
            "content": [{"type": "text", "text": PROMPT["system"]}],
        },
        {"role": "user", "content": user_content},
    ]
    return conversation, prompt_text


def apply_chat_template_compat(processor: Any, conversation: list[dict[str, Any]]) -> Any:
    """Tokenize one multimodal chat across old and new Transformers APIs."""
    common = {
        "add_generation_prompt": True,
        "tokenize": True,
        "return_dict": True,
        "return_tensors": "pt",
    }
    try:
        return processor.apply_chat_template(
            [conversation], processor_kwargs={"padding": False}, **common
        )
    except TypeError as exc:
        if "processor_kwargs" not in str(exc):
            raise
        # Transformers versions predating processor_kwargs accepted padding directly.
        return processor.apply_chat_template([conversation], padding=False, **common)


def make_position_masks(
    row: Mapping[str, Any],
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    tokenizer: Any,
    image_token_ids: set[int],
) -> dict[str, torch.Tensor]:
    if input_ids.shape[0] != 1:
        raise ValueError("Causal experiments require batch size 1")
    valid = attention_mask.bool()
    image = torch.zeros_like(valid)
    for token_id in image_token_ids:
        image |= input_ids.eq(token_id)
    image &= valid
    text = valid & ~image
    last = torch.zeros_like(valid)
    valid_positions = torch.nonzero(valid[0], as_tuple=False).flatten()
    if valid_positions.numel():
        last[0, int(valid_positions[-1])] = True

    query = torch.zeros_like(valid)
    question_ids = tokenizer.encode(
        str(row.get("question") or ""), add_special_tokens=False
    )
    sequence = input_ids[0].detach().cpu().tolist()
    span = find_subsequence(sequence, question_ids)
    if span is not None:
        query[0, span[0] : span[1]] = True
        query &= valid
    else:
        image_positions = torch.nonzero(image[0], as_tuple=False).flatten()
        start = int(image_positions[-1]) + 1 if image_positions.numel() else 0
        query[0, start:] = text[0, start:]
        # Exclude the final chat-template marker when a precise question span was not found.
        if valid_positions.numel() > 4:
            query[0, int(valid_positions[-4]) + 1 :] = False
        if not query.any():
            fallback = torch.nonzero(text[0], as_tuple=False).flatten()[-32:]
            query[0, fallback] = True

    return {
        "all": valid,
        "image": image,
        "text": text,
        "query": query,
        "last": last,
    }


def parse_gold_indices(row: Mapping[str, Any], options: Sequence[str]) -> list[int]:
    if not options:
        return []
    answer = str(row.get("reference") or row.get("answer") or "").strip()
    label_parts = re.split(r"\s*[/,|;]\s*", answer)
    if label_parts and all(
        re.fullmatch(r"\(?(?:[0-9]+|[A-Za-z])\)?", part.strip())
        for part in label_parts
    ):
        indices: list[int] = []
        for part in label_parts:
            token = part.strip().strip("()").upper()
            index = int(token) - 1 if token.isdigit() else ord(token) - ord("A")
            if 0 <= index < len(options):
                indices.append(index)
        if indices:
            return sorted(set(indices))
    normalized = re.sub(r"\s+", " ", answer).strip().lower()
    for index, option in enumerate(options):
        option_text = re.sub(
            r"^\s*(?:\(?[0-9]+\)?|\(?[A-Za-z]\)?)[.):_-]\s*", "", option
        )
        if re.sub(r"\s+", " ", option_text).strip().lower() == normalized:
            return [index]
    return []


def parse_gold_index(row: Mapping[str, Any], options: Sequence[str]) -> int | None:
    """Backward-compatible helper for code that explicitly requires one label."""
    indices = parse_gold_indices(row, options)
    return indices[0] if len(indices) == 1 else None


def option_label_token_ids(tokenizer: Any, option_count: int) -> list[list[int]]:
    if option_count > len(OPTION_LABELS):
        raise ValueError(f"Too many options: {option_count}")
    ids: list[list[int]] = []
    for label in OPTION_LABELS[:option_count]:
        variants = [label, f" {label}"]
        encoded_variants = [
            tokenizer.encode(candidate, add_special_tokens=False)
            for candidate in variants
        ]
        singles = list(
            dict.fromkeys(tokens[0] for tokens in encoded_variants if len(tokens) == 1)
        )
        if not singles:
            raise ValueError(
                f"Option label {label!r} is not a single token. "
                "Use a prompt/label scheme with single-token labels."
            )
        ids.append([int(token_id) for token_id in singles])
    return ids


def score_candidate_values(
    row: Mapping[str, Any], options: Sequence[str], values: Sequence[float]
) -> dict[str, Any]:
    """Turn arbitrary MC candidate scores into the standard result schema."""
    candidate_scores = [float(value) for value in values]
    if len(candidate_scores) != len(options):
        raise ValueError("Candidate score count does not match option count")
    predicted_index = max(range(len(options)), key=candidate_scores.__getitem__)
    gold_indices = parse_gold_indices(row, options)
    gold_index = gold_indices[0] if len(gold_indices) == 1 else None
    ordered = sorted(candidate_scores, reverse=True)
    top1_margin = ordered[0] - ordered[1] if len(ordered) > 1 else float("nan")
    candidate_tensor = torch.tensor(candidate_scores, dtype=torch.float32)
    probabilities = torch.softmax(candidate_tensor, dim=0)
    entropy = float(
        (-(probabilities * torch.log(probabilities.clamp_min(1e-12))).sum()).item()
    )
    gold_margin: float | None = None
    is_correct: bool | None = None
    if gold_indices:
        competitors = [
            score
            for index, score in enumerate(candidate_scores)
            if index not in gold_indices
        ]
        if competitors:
            gold_margin = max(candidate_scores[index] for index in gold_indices) - max(
                competitors
            )
        is_correct = predicted_index in gold_indices
    return {
        "prediction": OPTION_LABELS[predicted_index],
        "predicted_index": predicted_index,
        "predicted_option": options[predicted_index],
        "gold_index": gold_index,
        "gold_indices": gold_indices,
        "gold_label": (
            "/".join(OPTION_LABELS[index] for index in gold_indices)
            if gold_indices
            else None
        ),
        "is_correct": is_correct,
        "gold_margin": gold_margin,
        "top1_margin": top1_margin,
        "candidate_entropy": entropy,
        "candidate_log_probs": candidate_scores,
    }


class HookGroup:
    def __init__(self) -> None:
        self.handles: list[Any] = []

    def __enter__(self) -> "HookGroup":
        return self

    def __exit__(self, *_exc: Any) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def position_mask_for_hidden(
    hidden: torch.Tensor,
    prepared: PreparedInput,
    position: str,
    phase: str = "all",
) -> torch.Tensor | None:
    seq_len = hidden.shape[-2]
    prompt_len = prepared.inputs["input_ids"].shape[-1]
    is_prefill = seq_len == prompt_len and seq_len > 1
    is_decode = seq_len == 1 and not is_prefill
    if seq_len > 1 and seq_len != prompt_len:
        raise RuntimeError(
            "The hooked language sequence length does not match processor input_ids "
            f"({seq_len} != {prompt_len}). Token-position interventions cannot be "
            "aligned safely with this model/Transformers version."
        )

    if phase == "prefill" and not is_prefill:
        return None
    if phase == "decode" and not is_decode:
        return None
    if phase == "first_token":
        if not is_prefill:
            return None
        position = "last"
    if phase not in {"all", "prefill", "decode", "first_token"}:
        raise ValueError(f"Unsupported phase: {phase}")

    if is_prefill:
        mask = prepared.masks.get(position)
        if mask is None:
            raise ValueError(f"Unsupported token position group: {position}")
        return mask.to(device=hidden.device)
    if is_decode:
        if position in {"all", "last", "text"}:
            return torch.ones(
                hidden.shape[0], 1, dtype=torch.bool, device=hidden.device
            )
        return None
    return None


class HeadOutputScaler(HookGroup):
    """Scale selected per-head slices before the attention output projection."""

    def __init__(
        self,
        model: torch.nn.Module,
        prepared: PreparedInput,
        selected_heads: Sequence[tuple[int, int]],
        alpha: float,
        num_heads: int,
        phase: str = "all",
        position: str = "all",
    ) -> None:
        super().__init__()
        self.model = model
        self.prepared = prepared
        self.selected_heads = list(selected_heads)
        self.alpha = float(alpha)
        self.num_heads = int(num_heads)
        self.phase = phase
        self.position = position

    def __enter__(self) -> "HeadOutputScaler":
        layers, _ = decoder_layers(self.model)
        grouped: dict[int, list[int]] = {}
        for layer, head in self.selected_heads:
            grouped.setdefault(layer, []).append(head)
        for layer_idx, heads in grouped.items():
            if not 0 <= layer_idx < len(layers):
                raise ValueError(f"Layer out of range: {layer_idx}")
            projection = attention_output_projection(layers[layer_idx])

            def hook(
                _module: torch.nn.Module,
                args: tuple[Any, ...],
                heads: tuple[int, ...] = tuple(heads),
                layer_idx: int = layer_idx,
            ) -> tuple[Any, ...] | None:
                hidden = args[0]
                mask = position_mask_for_hidden(
                    hidden, self.prepared, self.position, self.phase
                )
                if mask is None or not mask.any() or self.alpha == 0:
                    return None
                if hidden.shape[-1] % self.num_heads:
                    raise RuntimeError(
                        f"Layer {layer_idx} width {hidden.shape[-1]} is not "
                        f"divisible by {self.num_heads} heads"
                    )
                head_dim = hidden.shape[-1] // self.num_heads
                changed = hidden.clone()
                for head in heads:
                    if not 0 <= head < self.num_heads:
                        raise ValueError(f"Head out of range: {layer_idx}:{head}")
                    start = head * head_dim
                    selected = changed[..., start : start + head_dim]
                    selected[mask] = selected[mask] * (1.0 + self.alpha)
                return (changed,) + tuple(args[1:])

            self.handles.append(projection.register_forward_pre_hook(hook))
        return self


class ComponentCapture(HookGroup):
    def __init__(
        self,
        model: torch.nn.Module,
        prepared: PreparedInput,
        component: str,
        layer_idx: int,
        position: str,
    ) -> None:
        super().__init__()
        self.model = model
        self.prepared = prepared
        self.component = component
        self.layer_idx = int(layer_idx)
        self.position = position
        self.values: torch.Tensor | None = None

    def _capture(self, hidden: torch.Tensor) -> None:
        mask = position_mask_for_hidden(hidden, self.prepared, self.position, "prefill")
        if mask is None or not mask.any():
            self.values = torch.empty(0, hidden.shape[-1], dtype=hidden.dtype)
            return
        self.values = hidden[mask].detach().cpu()

    def __enter__(self) -> "ComponentCapture":
        layers, _ = decoder_layers(self.model)
        if not 0 <= self.layer_idx < len(layers):
            raise ValueError(f"Layer out of range: {self.layer_idx}")
        if self.component == "head":
            projection = attention_output_projection(layers[self.layer_idx])

            def pre_hook(_module: torch.nn.Module, args: tuple[Any, ...]) -> None:
                self._capture(args[0])

            self.handles.append(projection.register_forward_pre_hook(pre_hook))
        elif self.component == "residual":

            def forward_hook(
                _module: torch.nn.Module, _args: tuple[Any, ...], output: Any
            ) -> None:
                self._capture(hidden_from_output(output))

            self.handles.append(layers[self.layer_idx].register_forward_hook(forward_hook))
        else:
            raise ValueError(f"Unsupported component: {self.component}")
        return self


class MultiLayerHeadCapture(HookGroup):
    """Capture selected token positions from many attention layers in one pass."""

    def __init__(
        self,
        model: torch.nn.Module,
        prepared: PreparedInput,
        layer_indices: Sequence[int],
        position: str = "last",
    ) -> None:
        super().__init__()
        self.model = model
        self.prepared = prepared
        self.layer_indices = [int(value) for value in layer_indices]
        self.position = position
        self.values: dict[int, torch.Tensor] = {}

    def __enter__(self) -> "MultiLayerHeadCapture":
        layers, _ = decoder_layers(self.model)
        for layer_idx in self.layer_indices:
            if not 0 <= layer_idx < len(layers):
                raise ValueError(f"Layer out of range: {layer_idx}")
            projection = attention_output_projection(layers[layer_idx])

            def pre_hook(
                _module: torch.nn.Module,
                args: tuple[Any, ...],
                layer_idx: int = layer_idx,
            ) -> None:
                hidden = args[0]
                mask = position_mask_for_hidden(
                    hidden, self.prepared, self.position, "prefill"
                )
                if mask is None or not mask.any():
                    self.values[layer_idx] = torch.empty(
                        0, hidden.shape[-1], dtype=hidden.dtype
                    )
                else:
                    self.values[layer_idx] = hidden[mask].detach().cpu()

            self.handles.append(projection.register_forward_pre_hook(pre_hook))
        return self


def align_source_values(source: torch.Tensor, target_count: int) -> torch.Tensor:
    if source.ndim != 2:
        raise ValueError(f"Expected [positions, hidden], got {tuple(source.shape)}")
    if target_count <= 0:
        return source[:0]
    if source.shape[0] == target_count:
        return source
    if source.shape[0] == 0:
        raise ValueError("Source token group is empty")
    if source.shape[0] == 1:
        return source.expand(target_count, -1)
    data = source.transpose(0, 1).unsqueeze(0).float()
    aligned = F.interpolate(data, size=target_count, mode="linear", align_corners=False)
    return aligned.squeeze(0).transpose(0, 1).to(dtype=source.dtype)


class ComponentPatcher(HookGroup):
    def __init__(
        self,
        model: torch.nn.Module,
        prepared: PreparedInput,
        component: str,
        layer_idx: int,
        position: str,
        source_values: torch.Tensor,
        strength: float,
        num_heads: int,
        selected_heads: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.prepared = prepared
        self.component = component
        self.layer_idx = int(layer_idx)
        self.position = position
        self.source_values = source_values.detach().cpu()
        self.strength = float(strength)
        self.num_heads = int(num_heads)
        self.selected_heads = list(selected_heads or [])

    def _patch(self, hidden: torch.Tensor) -> torch.Tensor:
        mask = position_mask_for_hidden(hidden, self.prepared, self.position, "prefill")
        if mask is None or not mask.any():
            return hidden
        target_count = int(mask.sum().item())
        source = align_source_values(self.source_values, target_count).to(
            device=hidden.device, dtype=hidden.dtype
        )
        changed = hidden.clone()
        target = changed[mask]
        if self.component == "head" and self.selected_heads:
            if hidden.shape[-1] % self.num_heads:
                raise RuntimeError(
                    f"Width {hidden.shape[-1]} is not divisible by {self.num_heads}"
                )
            head_dim = hidden.shape[-1] // self.num_heads
            for head in self.selected_heads:
                start = head * head_dim
                stop = start + head_dim
                target[:, start:stop] += self.strength * (
                    source[:, start:stop] - target[:, start:stop]
                )
        else:
            target += self.strength * (source - target)
        changed[mask] = target
        return changed

    def __enter__(self) -> "ComponentPatcher":
        layers, _ = decoder_layers(self.model)
        if not 0 <= self.layer_idx < len(layers):
            raise ValueError(f"Layer out of range: {self.layer_idx}")
        if self.component == "head":
            projection = attention_output_projection(layers[self.layer_idx])

            def pre_hook(
                _module: torch.nn.Module, args: tuple[Any, ...]
            ) -> tuple[Any, ...]:
                return (self._patch(args[0]),) + tuple(args[1:])

            self.handles.append(projection.register_forward_pre_hook(pre_hook))
        elif self.component == "residual":

            def forward_hook(
                _module: torch.nn.Module, _args: tuple[Any, ...], output: Any
            ) -> Any:
                return replace_hidden_output(output, self._patch(hidden_from_output(output)))

            self.handles.append(layers[self.layer_idx].register_forward_hook(forward_hook))
        else:
            raise ValueError(f"Unsupported component: {self.component}")
        return self


class ResultWriter:
    def __init__(self, root: Path, name: str) -> None:
        self.root = root / safe_name(name)
        self.root.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.root / "results.jsonl"
        self.csv_path = self.root / "results.csv"
        self.summary_path = self.root / "summary.csv"
        self.status_path = self.root / "status.json"
        self.records: dict[str, dict[str, Any]] = {}
        if self.jsonl_path.exists():
            with self.jsonl_path.open("r", encoding="utf-8") as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    self.records[str(record["result_id"])] = record

    def has(self, result_id: str) -> bool:
        return result_id in self.records

    def append(self, record: Mapping[str, Any]) -> None:
        cooked = jsonable(dict(record))
        result_id = str(cooked["result_id"])
        if result_id in self.records:
            return
        with self.jsonl_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(cooked, ensure_ascii=False) + "\n")
        self.records[result_id] = cooked

    def finalize(self) -> None:
        records = list(self.records.values())
        write_csv(self.csv_path, records)
        write_csv(self.summary_path, summarize_records(records))
        with self.status_path.open("w", encoding="utf-8") as stream:
            json.dump(
                {
                    "n_records": len(records),
                    "results_jsonl": str(self.jsonl_path),
                    "note": (
                        "No eligible rows/pairs matched this experiment."
                        if not records
                        else "completed"
                    ),
                },
                stream,
                ensure_ascii=False,
                indent=2,
            )


def summarize_records(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str], list[Mapping[str, Any]]] = {}
    for record in records:
        key = (
            str(record.get("config_id", "")),
            str(record.get("question_form", "")),
            str(record.get("condition", record.get("target_condition", ""))),
            str(record.get("control", "")),
        )
        groups.setdefault(key, []).append(record)
    summaries: list[dict[str, Any]] = []
    for key, rows in sorted(groups.items()):
        representative = rows[0]
        scored = [row for row in rows if isinstance(row.get("is_correct"), bool)]
        margins = [
            float(row["gold_margin"])
            for row in rows
            if row.get("gold_margin") not in (None, "")
        ]
        recoveries = [
            float(row["normalized_recovery"])
            for row in rows
            if row.get("normalized_recovery") not in (None, "")
            and math.isfinite(float(row["normalized_recovery"]))
        ]
        state_recoveries = [
            float(row["state_recovery"])
            for row in rows
            if row.get("state_recovery") not in (None, "")
            and math.isfinite(float(row["state_recovery"]))
        ]
        margin_deltas = [
            float(row["margin_delta"])
            for row in rows
            if row.get("margin_delta") not in (None, "")
            and math.isfinite(float(row["margin_delta"]))
        ]
        copy_flags = [
            bool(row["source_answer_copy"])
            for row in rows
            if isinstance(row.get("source_answer_copy"), bool)
        ]
        label_copy_flags = [
            bool(row["source_answer_label_copy"])
            for row in rows
            if isinstance(row.get("source_answer_label_copy"), bool)
        ]
        semantic_transfer_flags = [
            bool(row["semantic_option_transfer"])
            for row in rows
            if isinstance(row.get("semantic_option_transfer"), bool)
        ]
        repairs = sum(
            bool(row.get("is_correct")) and row.get("baseline_correct") is False
            for row in rows
        )
        damage = sum(
            (not bool(row.get("is_correct"))) and row.get("baseline_correct") is True
            for row in rows
            if isinstance(row.get("is_correct"), bool)
        )
        gate_flags = [
            bool(row["gate_applied"])
            for row in rows
            if isinstance(row.get("gate_applied"), bool)
        ]
        summaries.append(
            {
                "config_id": key[0],
                "question_form": key[1],
                "condition": key[2],
                "control": key[3],
                "head_set": representative.get("head_set", ""),
                "heads": representative.get("heads", ""),
                "layer": representative.get(
                    "layer", representative.get("sender_layer", "")
                ),
                "position": representative.get(
                    "position", representative.get("sender_position", "")
                ),
                "alpha": representative.get("alpha", ""),
                "scoring_method": representative.get("scoring_method", ""),
                "prior": representative.get("prior", ""),
                "gate_type": representative.get("gate_type", ""),
                "gate_threshold": representative.get("gate_threshold", ""),
                "receiver_layer": representative.get("receiver_layer", ""),
                "n": len(rows),
                "n_scored": len(scored),
                "accuracy": (
                    sum(bool(row["is_correct"]) for row in scored) / len(scored)
                    if scored
                    else ""
                ),
                "mean_gold_margin": sum(margins) / len(margins) if margins else "",
                "mean_normalized_recovery": (
                    sum(recoveries) / len(recoveries) if recoveries else ""
                ),
                "mean_state_recovery": (
                    sum(state_recoveries) / len(state_recoveries)
                    if state_recoveries
                    else ""
                ),
                "mean_margin_delta": (
                    sum(margin_deltas) / len(margin_deltas)
                    if margin_deltas
                    else ""
                ),
                "source_answer_copy_rate": (
                    sum(copy_flags) / len(copy_flags) if copy_flags else ""
                ),
                "source_answer_label_copy_rate": (
                    sum(label_copy_flags) / len(label_copy_flags)
                    if label_copy_flags
                    else ""
                ),
                "semantic_option_transfer_rate": (
                    sum(semantic_transfer_flags) / len(semantic_transfer_flags)
                    if semantic_transfer_flags
                    else ""
                ),
                "repairs": repairs,
                "damage": damage,
                "net_repairs": repairs - damage,
                "intervention_rate": (
                    sum(gate_flags) / len(gate_flags) if gate_flags else ""
                ),
            }
        )
    return summaries


class ExperimentRunner:
    def __init__(
        self,
        model: torch.nn.Module,
        processor: Any,
        rows: list[dict[str, Any]],
        image_store: ImageStore,
        output_dir: Path,
        prompt_keys: Mapping[str, str],
        num_heads: int,
        dtype: torch.dtype,
        max_new_tokens: Mapping[str, int] | None = None,
        seed: int = 17,
        activation_cache_gb: float = 1.0,
    ) -> None:
        self.model = model
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.rows = rows
        self.image_store = image_store
        self.output_dir = output_dir
        self.prompt_keys = dict(prompt_keys)
        self.num_heads = int(num_heads)
        self.dtype = dtype
        self.max_new_tokens = dict(DEFAULT_MAX_NEW_TOKENS)
        if max_new_tokens:
            self.max_new_tokens.update(
                {norm_form(key): int(value) for key, value in max_new_tokens.items()}
            )
        self.seed = int(seed)
        self.device = next(model.parameters()).device
        self.image_token_ids = collect_image_token_ids(model, processor)
        self.mc_cache: dict[str, dict[str, Any]] = {}
        self.semantic_mc_cache: dict[tuple[str, str, str], dict[str, Any]] = {}
        self.content_batch_disabled = False
        self.rows_by_question_condition: dict[tuple[str, str], dict[str, Any]] = {}
        for row in self.rows:
            key = (str(row.get("question_id")), norm_condition(row.get("condition")))
            self.rows_by_question_condition.setdefault(key, row)
        self.activation_cache: OrderedDict[tuple[Any, ...], torch.Tensor] = OrderedDict()
        self.activation_cache_bytes = 0
        self.activation_cache_limit_bytes = max(
            0, int(float(activation_cache_gb) * (1024**3))
        )

        for form, key in self.prompt_keys.items():
            validate_prompt_key(form, key)

    def prepare(
        self, row: Mapping[str, Any], prompt_key_override: str | None = None
    ) -> PreparedInput:
        row_copy = dict(row)
        conversation, prompt_text = make_conversation(
            row_copy,
            self.image_store,
            self.prompt_keys,
            prompt_key_override=prompt_key_override,
        )
        batch = apply_chat_template_compat(self.processor, conversation)
        # Some Qwen3-VL processor versions emit this legacy key although the
        # conditional-generation forward signature does not consume it.
        batch.pop("token_type_ids", None)
        inputs = move_inputs(batch, self.device, self.dtype)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(inputs["input_ids"])
            inputs["attention_mask"] = attention_mask
        masks = make_position_masks(
            row_copy,
            inputs["input_ids"],
            attention_mask,
            self.tokenizer,
            self.image_token_ids,
        )
        return PreparedInput(row_copy, inputs, masks, prompt_text)

    def _score_prepared(
        self,
        prepared: PreparedInput,
        context: Any = None,
    ) -> dict[str, Any]:
        options = parse_maybe_list(prepared.row.get("options_json"))
        if not options:
            raise ValueError(
                f"MC row has no options: {prepared.row.get('question_id')}"
            )
        token_id_groups = option_label_token_ids(self.tokenizer, len(options))
        manager = context if context is not None else nullcontext()
        with torch.inference_mode(), manager:
            outputs = self.model(**prepared.inputs, use_cache=False, return_dict=True)
        last_position = int(
            torch.nonzero(prepared.masks["last"][0], as_tuple=False).flatten()[-1]
        )
        logits = outputs.logits[0, last_position].float()
        log_probs = torch.log_softmax(logits, dim=-1)
        candidate_log_probs = [
            float(torch.logsumexp(log_probs[token_ids], dim=0).item())
            for token_ids in token_id_groups
        ]
        return score_candidate_values(prepared.row, options, candidate_log_probs)

    def score_mc(
        self,
        row: Mapping[str, Any],
        context_factory: Callable[[PreparedInput], Any] | None = None,
        cache_baseline: bool = True,
    ) -> dict[str, Any]:
        cache_key = str(row.get("sample_uid"))
        if context_factory is None and cache_baseline and cache_key in self.mc_cache:
            return dict(self.mc_cache[cache_key])
        prepared = self.prepare(row)
        context = context_factory(prepared) if context_factory else None
        result = self._score_prepared(prepared, context)
        if context_factory is None and cache_baseline:
            self.mc_cache[cache_key] = dict(result)
        return result

    @staticmethod
    def _repeat_tensor_for_candidates(
        key: str, value: torch.Tensor, batch_size: int
    ) -> torch.Tensor:
        """Repeat one prepared multimodal sample for a small candidate batch."""
        if batch_size == 1 or value.ndim == 0:
            return value
        if key in {"image_grid_thw", "video_grid_thw"}:
            return torch.cat([value] * batch_size, dim=0)
        if key in {"pixel_values", "pixel_values_videos"}:
            # Qwen stores flattened patches as [all_patches, dim], whereas
            # LLaVA commonly keeps an explicit leading batch dimension.
            if value.ndim >= 4 and value.shape[0] == 1:
                return value.repeat((batch_size,) + (1,) * (value.ndim - 1))
            return torch.cat([value] * batch_size, dim=0)
        if value.shape[0] == 1:
            return value.repeat((batch_size,) + (1,) * (value.ndim - 1))
        return value

    def _content_scores_prepared(
        self,
        prepared: PreparedInput,
        options: Sequence[str],
        *,
        normalization: str,
        candidate_batch_size: int,
        continuation_prefix: str,
    ) -> tuple[list[float], list[int]]:
        """Score complete semantic option strings as causal continuations."""
        normalization = str(normalization).lower()
        if normalization not in {"mean", "sum", "sqrt"}:
            raise ValueError("content normalization must be mean, sum, or sqrt")
        tokenized: list[list[int]] = []
        for option in options:
            semantic_text = strip_option_label(option)
            token_ids = self.tokenizer.encode(
                f"{continuation_prefix}{semantic_text}", add_special_tokens=False
            )
            if not token_ids:
                raise ValueError(f"Option produced no tokens: {option!r}")
            tokenized.append([int(token_id) for token_id in token_ids])

        prompt_ids = prepared.inputs["input_ids"]
        prompt_attention = prepared.inputs["attention_mask"]
        if prompt_ids.shape[0] != 1:
            raise ValueError("Content ranking expects one prepared sample")
        prompt_length = int(prompt_ids.shape[1])
        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_id is None:
            pad_id = getattr(self.tokenizer, "eos_token_id", 0)
        requested_batch_size = max(1, int(candidate_batch_size))
        if self.content_batch_disabled:
            requested_batch_size = 1

        scores: list[float] = []
        try:
            for start in range(0, len(tokenized), requested_batch_size):
                chunk = tokenized[start : start + requested_batch_size]
                batch_size = len(chunk)
                max_candidate_length = max(len(tokens) for tokens in chunk)
                sequences = torch.full(
                    (batch_size, prompt_length + max_candidate_length),
                    int(pad_id),
                    dtype=prompt_ids.dtype,
                    device=prompt_ids.device,
                )
                attention = torch.zeros(
                    (batch_size, prompt_length + max_candidate_length),
                    dtype=prompt_attention.dtype,
                    device=prompt_attention.device,
                )
                sequences[:, :prompt_length] = prompt_ids.expand(batch_size, -1)
                attention[:, :prompt_length] = prompt_attention.expand(batch_size, -1)
                for index, tokens in enumerate(chunk):
                    length = len(tokens)
                    sequences[index, prompt_length : prompt_length + length] = torch.tensor(
                        tokens, dtype=prompt_ids.dtype, device=prompt_ids.device
                    )
                    attention[index, prompt_length : prompt_length + length] = 1

                model_inputs: dict[str, Any] = {
                    "input_ids": sequences,
                    "attention_mask": attention,
                }
                for key, value in prepared.inputs.items():
                    if key in {"input_ids", "attention_mask", "position_ids", "cache_position"}:
                        continue
                    if isinstance(value, torch.Tensor):
                        if (
                            value.ndim == 2
                            and value.shape[0] == 1
                            and value.shape[1] == prompt_length
                        ):
                            # Qwen3-VL returns mm_token_type_ids with one entry
                            # per prompt token (0=text, 1=image, 2=video).  The
                            # appended answer candidates are ordinary text, so
                            # extend every sequence-aligned auxiliary tensor
                            # with zeros to exactly match input_ids.
                            expanded = torch.zeros(
                                (batch_size, prompt_length + max_candidate_length),
                                dtype=value.dtype,
                                device=value.device,
                            )
                            expanded[:, :prompt_length] = value.expand(batch_size, -1)
                            model_inputs[key] = expanded
                            continue
                        model_inputs[key] = self._repeat_tensor_for_candidates(
                            key, value, batch_size
                        )
                    else:
                        model_inputs[key] = value

                with torch.inference_mode():
                    outputs = self.model(
                        **model_inputs, use_cache=False, return_dict=True
                    )
                logits = outputs.logits.float()
                for index, tokens in enumerate(chunk):
                    length = len(tokens)
                    positions = torch.arange(
                        prompt_length - 1,
                        prompt_length + length - 1,
                        device=logits.device,
                    )
                    target = torch.tensor(tokens, device=logits.device, dtype=torch.long)
                    token_log_probs = torch.log_softmax(
                        logits[index, positions], dim=-1
                    ).gather(-1, target.unsqueeze(-1)).squeeze(-1)
                    total = token_log_probs.sum()
                    if normalization == "mean":
                        total = total / length
                    elif normalization == "sqrt":
                        total = total / math.sqrt(length)
                    scores.append(float(total.item()))
        except (RuntimeError, ValueError, IndexError) as exc:
            if requested_batch_size <= 1:
                raise
            print(
                "[content-ranking] candidate batching failed; retrying with "
                f"batch_size=1 ({type(exc).__name__}: {exc})"
            )
            self.content_batch_disabled = True
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return self._content_scores_prepared(
                prepared,
                options,
                normalization=normalization,
                candidate_batch_size=1,
                continuation_prefix=continuation_prefix,
            )
        return scores, [len(tokens) for tokens in tokenized]

    def score_mc_content(
        self,
        row: Mapping[str, Any],
        *,
        prompt_key: str = "MC_CONTENT_1",
        normalization: str = "mean",
        candidate_batch_size: int = 1,
        continuation_prefix: str = " ",
        cache_baseline: bool = True,
    ) -> dict[str, Any]:
        """SEED-style answer ranking using the full option content."""
        options = parse_maybe_list(row.get("options_json"))
        if not options:
            raise ValueError(f"MC row has no options: {row.get('question_id')}")
        cache_key = (
            str(row.get("sample_uid")),
            str(prompt_key),
            f"{normalization}:{continuation_prefix}",
        )
        if cache_baseline and cache_key in self.semantic_mc_cache:
            return dict(self.semantic_mc_cache[cache_key])
        prepared = self.prepare(row, prompt_key_override=prompt_key)
        values, token_counts = self._content_scores_prepared(
            prepared,
            options,
            normalization=normalization,
            candidate_batch_size=candidate_batch_size,
            continuation_prefix=continuation_prefix,
        )
        result = score_candidate_values(row, options, values)
        result.update(
            {
                "scoring_method": "content_likelihood",
                "content_prompt_key": prompt_key,
                "content_normalization": normalization,
                "candidate_token_counts": token_counts,
            }
        )
        if cache_baseline:
            self.semantic_mc_cache[cache_key] = dict(result)
        return result

    def generate(
        self,
        row: Mapping[str, Any],
        context_factory: Callable[[PreparedInput], Any] | None = None,
        *,
        prompt_key_override: str | None = None,
        max_new_tokens_override: int | None = None,
    ) -> str:
        prepared = self.prepare(row, prompt_key_override=prompt_key_override)
        context = context_factory(prepared) if context_factory else None
        form = norm_form(row.get("question_form"))
        manager = context if context is not None else nullcontext()
        max_new_tokens = (
            int(max_new_tokens_override)
            if max_new_tokens_override is not None
            else self.max_new_tokens[form]
        )
        with torch.inference_mode(), manager:
            generated = self.model.generate(
                **prepared.inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
        input_length = prepared.inputs["input_ids"].shape[-1]
        trimmed = generated[:, input_length:]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()

    def capture_values(
        self,
        row: Mapping[str, Any],
        component: str,
        layer_idx: int,
        position: str,
    ) -> torch.Tensor:
        cache_key = (
            row.get("sample_uid"),
            component,
            int(layer_idx),
            position,
        )
        if cache_key in self.activation_cache:
            cached = self.activation_cache.pop(cache_key)
            self.activation_cache[cache_key] = cached
            return cached.clone()
        prepared = self.prepare(row)
        capture = ComponentCapture(
            self.model, prepared, component, int(layer_idx), position
        )
        with torch.inference_mode(), capture:
            self.model(**prepared.inputs, use_cache=False, return_dict=True)
        if capture.values is None:
            raise RuntimeError("Capture hook did not run")
        cached = capture.values.clone()
        cached_bytes = cached.numel() * cached.element_size()
        if self.activation_cache_limit_bytes and cached_bytes <= self.activation_cache_limit_bytes:
            while (
                self.activation_cache
                and self.activation_cache_bytes + cached_bytes
                > self.activation_cache_limit_bytes
            ):
                _old_key, old_value = self.activation_cache.popitem(last=False)
                self.activation_cache_bytes -= old_value.numel() * old_value.element_size()
            self.activation_cache[cache_key] = cached
            self.activation_cache_bytes += cached_bytes
        return capture.values.clone()

    def capture_head_values_many(
        self,
        row: Mapping[str, Any],
        layer_indices: Sequence[int],
        position: str = "last",
    ) -> dict[int, torch.Tensor]:
        """Capture all requested head-output tensors with one model forward."""
        prepared = self.prepare(row)
        capture = MultiLayerHeadCapture(
            self.model, prepared, layer_indices, position
        )
        with torch.inference_mode(), capture:
            self.model(**prepared.inputs, use_cache=False, return_dict=True)
        missing = sorted(set(int(value) for value in layer_indices) - set(capture.values))
        if missing:
            raise RuntimeError(f"Head capture hooks did not run for layers: {missing}")
        return {layer: values.clone() for layer, values in capture.values.items()}

    def score_and_capture_head_values_many(
        self,
        row: Mapping[str, Any],
        layer_indices: Sequence[int],
        position: str = "last",
    ) -> tuple[dict[str, Any], dict[int, torch.Tensor]]:
        """Score an MC row and capture many layers during that same forward."""
        prepared = self.prepare(row)
        capture = MultiLayerHeadCapture(
            self.model, prepared, layer_indices, position
        )
        score = self._score_prepared(prepared, capture)
        missing = sorted(set(int(value) for value in layer_indices) - set(capture.values))
        if missing:
            raise RuntimeError(f"Head capture hooks did not run for layers: {missing}")
        self.mc_cache[str(row.get("sample_uid"))] = dict(score)
        return score, {
            layer: values.clone() for layer, values in capture.values.items()
        }

    def clear_activation_cache(self) -> None:
        self.activation_cache.clear()
        self.activation_cache_bytes = 0

    def rows_filtered(
        self,
        forms: Sequence[str] | None = None,
        conditions: Sequence[str] | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        form_set = {norm_form(form) for form in (forms or [])}
        condition_set = {norm_condition(condition) for condition in (conditions or [])}
        rows = [
            row
            for row in self.rows
            if (not form_set or row["question_form"] in form_set)
            and (not condition_set or row["condition"] in condition_set)
        ]
        if limit is not None:
            rows = rows[: int(limit)]
        return rows

    def synthesize_text_only(self, row: Mapping[str, Any]) -> dict[str, Any]:
        result = deepcopy(dict(row))
        result["condition"] = "text_only"
        result["input_image_variant"] = "none"
        result["input_image_name"] = ""
        result["sample_uid"] = f"{row.get('question_id')}::text_only::synthetic"
        result["synthetic_condition"] = True
        return result

    def row_for_condition(
        self, row: Mapping[str, Any], condition: str
    ) -> dict[str, Any] | None:
        matched = self.rows_by_question_condition.get(
            (str(row.get("question_id")), norm_condition(condition))
        )
        return deepcopy(matched) if matched is not None else None

    def synthesize_blurred(
        self, row: Mapping[str, Any], radius: float = 12.0
    ) -> dict[str, Any]:
        """Use a paired blurred row when available, otherwise blur deterministically."""
        paired = self.row_for_condition(row, "blurred")
        if paired is not None:
            return paired
        result = deepcopy(dict(row))
        result["condition"] = "blurred"
        result["_image_corruption"] = {
            "type": "gaussian_blur",
            "radius": float(radius),
        }
        result["sample_uid"] = f"{row.get('sample_uid')}::gaussian_blur_{radius:g}"
        result["synthetic_condition"] = True
        return result

    def synthesize_noisy(
        self, row: Mapping[str, Any], sigma: float = 0.2
    ) -> dict[str, Any]:
        result = deepcopy(dict(row))
        seed_hex = stable_hash([self.seed, row.get("question_id"), sigma], 8)
        result["_image_corruption"] = {
            "type": "gaussian_noise",
            "sigma": float(sigma),
            "seed": int(seed_hex, 16),
        }
        result["sample_uid"] = (
            f"{row.get('sample_uid')}::gaussian_noise_{float(sigma):g}"
        )
        result["synthetic_condition"] = True
        return result

    def synthesize_shuffled(
        self,
        row: Mapping[str, Any],
        seed: int | None = None,
        *,
        prefer_manifest: bool = True,
    ) -> dict[str, Any]:
        """Pair a question with a deterministic, different image.

        A manifest-provided shuffled row takes precedence unless
        ``prefer_manifest`` is false.  Otherwise a donor is chosen from the Full
        MC rows without using answer labels.  The caller can later set
        ``_target_image_size`` so the donor is processed on the exact same visual
        grid as the target Full image.
        """
        paired = self.row_for_condition(row, "shuffled") if prefer_manifest else None
        if paired is not None:
            return paired
        candidates = [
            candidate
            for candidate in self.rows_filtered(["MC"], ["full"], None)
            if str(candidate.get("question_id")) != str(row.get("question_id"))
            and str(candidate.get("input_image_name") or "")
            and str(candidate.get("input_image_name"))
            != str(row.get("input_image_name"))
        ]
        if not candidates:
            raise ValueError(
                "Cannot synthesize a shuffled condition: fewer than two distinct "
                "Full images are available"
            )
        effective_seed = self.seed if seed is None else int(seed)
        offset = int(
            stable_hash(
                [effective_seed, str(row.get("question_id")), "shuffled_donor"],
                12,
            ),
            16,
        ) % len(candidates)
        donor = candidates[offset]
        result = deepcopy(dict(row))
        result["condition"] = "shuffled"
        result["input_image_variant"] = "original"
        result["input_image_name"] = str(donor.get("input_image_name") or "")
        result["shuffled_from_image_name"] = result["input_image_name"]
        result["shuffled_from_question_id"] = str(donor.get("question_id"))
        result["sample_uid"] = (
            f"{row.get('question_id')}::shuffled::synthetic_seed_{effective_seed}::"
            f"{result['input_image_name']}"
        )
        result["synthetic_condition"] = True
        result["shuffle_seed"] = effective_seed
        return result

    def expanded_counterfactual_rows(
        self,
        conditions: Sequence[str],
        limit: int | None = None,
        *,
        blur_radius: float = 12.0,
        shuffle_seed: int | None = None,
        match_image_size: bool = True,
        canonical_image_size: Sequence[int] | None = None,
        image_fit_mode: str = "stretch",
        prefer_manifest_shuffled: bool = True,
    ) -> list[dict[str, Any]]:
        """Return one matched condition row per Full question.

        This is used by mismatch and always-on controls so a plain Full-only JSON
        can still produce deterministic Text-only, Blurred, and Shuffled rows.
        Existing manifest rows are reused when present.
        """
        normalized = [norm_condition(value) for value in conditions]
        full_rows = self.rows_filtered(["MC"], ["full"], limit)
        expanded: list[dict[str, Any]] = []
        for full_row in full_rows:
            target_size: tuple[int, int] | None = None
            if canonical_image_size is not None:
                if len(canonical_image_size) != 2:
                    raise ValueError("canonical_image_size must contain width and height")
                target_size = (
                    int(canonical_image_size[0]),
                    int(canonical_image_size[1]),
                )
            elif match_image_size:
                full_image = self.image_store.resolve(full_row)
                if full_image is None:
                    raise ValueError(
                        f"Full row has no image: {full_row.get('question_id')}"
                    )
                target_size = tuple(int(value) for value in full_image.size)
            for condition in normalized:
                if condition == "full":
                    selected = deepcopy(full_row)
                elif condition == "text_only":
                    selected = (
                        self.row_for_condition(full_row, "text_only")
                        or self.synthesize_text_only(full_row)
                    )
                elif condition == "blurred":
                    selected = self.synthesize_blurred(full_row, blur_radius)
                elif condition == "shuffled":
                    selected = self.synthesize_shuffled(
                        full_row,
                        shuffle_seed,
                        prefer_manifest=prefer_manifest_shuffled,
                    )
                else:
                    selected = self.row_for_condition(full_row, condition)
                    if selected is None:
                        raise ValueError(
                            f"Missing condition {condition!r} for question "
                            f"{full_row.get('question_id')}"
                        )
                if target_size is not None and condition != "text_only":
                    selected["_target_image_size"] = list(target_size)
                    selected["_target_image_fit"] = str(image_fit_mode)
                    selected["sample_uid"] = (
                        f"{selected.get('sample_uid')}::grid_{target_size[0]}x"
                        f"{target_size[1]}::fit_{image_fit_mode}"
                    )
                expanded.append(selected)
        return expanded

    def paired_rows(
        self,
        source_condition: str,
        target_condition: str,
        limit: int | None = None,
    ) -> list[tuple[dict[str, Any], dict[str, Any]]]:
        source_condition = norm_condition(source_condition)
        target_condition = norm_condition(target_condition)
        source_map = {
            row["question_id"]: row
            for row in self.rows
            if row["question_form"] == "MC" and row["condition"] == source_condition
        }
        target_map = {
            row["question_id"]: row
            for row in self.rows
            if row["question_form"] == "MC" and row["condition"] == target_condition
        }
        if target_condition == "text_only":
            for question_id, source in source_map.items():
                target_map.setdefault(question_id, self.synthesize_text_only(source))
        if source_condition == "text_only":
            for question_id, target in target_map.items():
                source_map.setdefault(question_id, self.synthesize_text_only(target))
        pairs = [
            (source_map[qid], target_map[qid])
            for qid in sorted(source_map.keys() & target_map.keys())
        ]
        if limit is not None:
            pairs = pairs[: int(limit)]
        return pairs

    @staticmethod
    def pair_matches_filter(
        source_score: Mapping[str, Any],
        target_score: Mapping[str, Any],
        pair_filter: str,
    ) -> bool:
        source_correct = source_score.get("is_correct")
        target_correct = target_score.get("is_correct")
        if pair_filter == "all":
            return True
        if pair_filter == "discordant":
            return isinstance(source_correct, bool) and isinstance(target_correct, bool) and source_correct != target_correct
        if pair_filter == "source_only":
            return source_correct is True and target_correct is False
        if pair_filter == "target_only":
            return source_correct is False and target_correct is True
        if pair_filter == "both_correct":
            return source_correct is True and target_correct is True
        if pair_filter == "both_wrong":
            return source_correct is False and target_correct is False
        raise ValueError(f"Unsupported pair_filter: {pair_filter}")

    def run_generation_scaling(self, exp: Mapping[str, Any], profile: str) -> None:
        writer = ResultWriter(self.output_dir, str(exp["name"]))
        heads = parse_heads(exp.get("heads"))
        settings = list(exp.get(f"{profile}_settings") or exp.get("settings") or [])
        if not settings:
            settings = [
                {
                    "alpha": exp.get("alpha", 0.2),
                    "phase": exp.get("phase", "all"),
                    "position": exp.get("position", "all"),
                }
            ]
        limit = exp.get(f"{profile}_limit", exp.get("limit"))
        rows = self.rows_filtered(exp.get("forms"), exp.get("conditions"), limit)
        for setting in settings:
            alpha = float(setting.get("alpha", 0.0))
            phase = str(setting.get("phase", "all"))
            position = str(setting.get("position", "all"))
            config = {
                "heads": heads,
                "alpha": alpha,
                "phase": phase,
                "position": position,
            }
            config_id = stable_hash(config)
            for row in tqdm(rows, desc=f"{exp['name']}:{config_id}"):
                result_id = f"{row['sample_uid']}::{config_id}"
                if writer.has(result_id):
                    continue

                def factory(prepared: PreparedInput) -> HeadOutputScaler:
                    return HeadOutputScaler(
                        self.model,
                        prepared,
                        heads,
                        alpha,
                        self.num_heads,
                        phase,
                        position,
                    )

                if row["question_form"] == "MC":
                    score = self.score_mc(row, factory, cache_baseline=False)
                    record = dict(score)
                    record["raw_output"] = score["prediction"]
                else:
                    prediction = self.generate(row, factory)
                    record = {"prediction": prediction, "raw_output": prediction}
                record.update(
                    {
                        "result_id": result_id,
                        "experiment": exp["name"],
                        "experiment_type": "generation_scaling",
                        "config_id": config_id,
                        "question_id": row["question_id"],
                        "question_form": row["question_form"],
                        "condition": row["condition"],
                        "reference": row["reference"],
                        "scaled_heads": heads,
                        "head_scale": alpha,
                        "multiplier": 1.0 + alpha,
                        "phase": phase,
                        "position": position,
                    }
                )
                writer.append(record)
        writer.finalize()

    def permute_options(
        self, row: Mapping[str, Any], permutation_index: int, seed: int
    ) -> tuple[dict[str, Any], list[int], list[int]]:
        options = parse_maybe_list(row.get("options_json"))
        gold_original = parse_gold_indices(row, options)
        permutation = list(range(len(options)))
        if permutation_index > 0:
            rng = random.Random(f"{seed}:{row.get('question_id')}:{permutation_index}")
            rng.shuffle(permutation)
        permuted = [options[index] for index in permutation]
        result = deepcopy(dict(row))
        result["options_json"] = json.dumps(permuted, ensure_ascii=False)
        if gold_original:
            new_gold = sorted(permutation.index(index) for index in gold_original)
            new_reference = "/".join(OPTION_LABELS[index] for index in new_gold)
            result["reference"] = new_reference
            result["answer"] = new_reference
        result["sample_uid"] = f"{row.get('sample_uid')}::perm_{permutation_index}"
        return result, permutation, gold_original

    def run_option_permutation(self, exp: Mapping[str, Any], profile: str) -> None:
        writer = ResultWriter(self.output_dir, str(exp["name"]))
        limit = exp.get(f"{profile}_limit", exp.get("limit"))
        rows = self.rows_filtered(["MC"], exp.get("conditions", ["full"]), limit)
        n_permutations = int(exp.get(f"{profile}_permutations", exp.get("permutations", 5)))
        heads = parse_heads(exp.get("heads"))
        alpha = float(exp.get("alpha", 0.2))
        phase = str(exp.get("phase", "first_token"))
        position = str(exp.get("position", "last"))
        config = {
            "heads": heads,
            "alpha": alpha,
            "phase": phase,
            "position": position,
            "permutations": n_permutations,
        }
        config_id = stable_hash(config)
        for row in tqdm(rows, desc=str(exp["name"])):
            for permutation_index in range(n_permutations + 1):
                permuted, permutation, gold_original = self.permute_options(
                    row, permutation_index, self.seed
                )
                result_id = f"{permuted['sample_uid']}::{config_id}"
                if writer.has(result_id):
                    continue

                def factory(prepared: PreparedInput) -> HeadOutputScaler:
                    return HeadOutputScaler(
                        self.model,
                        prepared,
                        heads,
                        alpha,
                        self.num_heads,
                        phase,
                        position,
                    )

                score = self.score_mc(permuted, factory, cache_baseline=False)
                semantic_prediction = None
                if score["predicted_index"] < len(permutation):
                    semantic_prediction = permutation[score["predicted_index"]]
                record = {
                    **score,
                    "result_id": result_id,
                    "experiment": exp["name"],
                    "experiment_type": "option_permutation",
                    "config_id": config_id,
                    "question_id": row["question_id"],
                    "question_form": "MC",
                    "condition": row["condition"],
                    "reference": permuted["reference"],
                    "permutation_index": permutation_index,
                    "permutation_new_to_old": permutation,
                    "gold_original_indices": gold_original,
                    "semantic_prediction_original_index": semantic_prediction,
                    "scaled_heads": heads,
                    "head_scale": alpha,
                    "phase": phase,
                    "position": position,
                }
                writer.append(record)
        writer.finalize()
        by_question: dict[str, list[Mapping[str, Any]]] = {}
        for record in writer.records.values():
            by_question.setdefault(str(record.get("question_id", "")), []).append(record)
        consistency_rows: list[dict[str, Any]] = []
        for question_id, records in sorted(by_question.items()):
            records = sorted(records, key=lambda row: int(row["permutation_index"]))
            baseline = next(
                (row for row in records if int(row["permutation_index"]) == 0), None
            )
            if baseline is None:
                continue
            baseline_semantic = baseline.get("semantic_prediction_original_index")
            comparisons = [
                row
                for row in records
                if int(row["permutation_index"]) != 0
            ]
            consistent = [
                row.get("semantic_prediction_original_index") == baseline_semantic
                for row in comparisons
            ]
            consistency_rows.append(
                {
                    "question_id": question_id,
                    "n_permutations": len(comparisons),
                    "baseline_semantic_prediction": baseline_semantic,
                    "semantic_consistency": (
                        sum(consistent) / len(consistent) if consistent else ""
                    ),
                    "permuted_accuracy": (
                        sum(bool(row["is_correct"]) for row in comparisons)
                        / len(comparisons)
                        if comparisons
                        and all(isinstance(row.get("is_correct"), bool) for row in comparisons)
                        else ""
                    ),
                }
            )
        write_csv(writer.root / "permutation_consistency.csv", consistency_rows)

    def run_patching_sweep(self, exp: Mapping[str, Any], profile: str) -> None:
        writer = ResultWriter(self.output_dir, str(exp["name"]))
        limit = exp.get(f"{profile}_limit", exp.get("limit"))
        pairs = self.paired_rows(
            str(exp["source_condition"]), str(exp["target_condition"]), limit
        )
        component = str(exp.get("component", "residual"))
        layers = [int(layer) for layer in exp.get("layers", [])]
        positions = list(exp.get("positions", ["last"]))
        strengths = [float(value) for value in exp.get("strengths", [1.0])]
        all_heads = parse_heads(exp.get("heads"))
        raw_head_sets = exp.get("head_sets") or []
        head_sets: list[tuple[str, list[tuple[int, int]]]] = []
        if component == "head" and raw_head_sets:
            for index, head_set in enumerate(raw_head_sets):
                if not isinstance(head_set, Mapping):
                    raise ValueError("Each head_sets entry must contain name and heads")
                head_sets.append(
                    (
                        str(head_set.get("name", f"head_set_{index}")),
                        parse_heads(head_set.get("heads")),
                    )
                )
        else:
            head_sets = [(str(exp.get("head_set", "selected")), all_heads)]
        pair_filter = str(exp.get("pair_filter", "all"))

        for source, target in tqdm(pairs, desc=str(exp["name"])):
            source_score = self.score_mc(source)
            target_score = self.score_mc(target)
            if not self.pair_matches_filter(source_score, target_score, pair_filter):
                continue
            for layer_idx in layers:
                for position in positions:
                    source_values = self.capture_values(
                        source, component, layer_idx, position
                    )
                    for head_set_name, head_pairs in head_sets:
                        layer_heads = [
                            head for layer, head in head_pairs if layer == layer_idx
                        ]
                        if component == "head" and head_pairs and not layer_heads:
                            continue
                        for strength in strengths:
                            config = {
                                "component": component,
                                "layer": layer_idx,
                                "head_set": head_set_name,
                                "heads": layer_heads,
                                "position": position,
                                "strength": strength,
                                "source_condition": source["condition"],
                                "target_condition": target["condition"],
                            }
                            config_id = stable_hash(config)
                            result_id = f"{target['sample_uid']}::{config_id}"
                            if writer.has(result_id):
                                continue

                            def factory(
                                prepared: PreparedInput,
                                source_values: torch.Tensor = source_values,
                                layer_heads: list[int] = layer_heads,
                            ) -> ComponentPatcher:
                                return ComponentPatcher(
                                    self.model,
                                    prepared,
                                    component,
                                    layer_idx,
                                    position,
                                    source_values,
                                    strength,
                                    self.num_heads,
                                    layer_heads,
                                )

                            score = self.score_mc(target, factory, cache_baseline=False)
                            denominator = (
                                float(source_score["gold_margin"])
                                - float(target_score["gold_margin"])
                                if source_score.get("gold_margin") is not None
                                and target_score.get("gold_margin") is not None
                                else float("nan")
                            )
                            normalized = float("nan")
                            if score.get("gold_margin") is not None and abs(denominator) > 1e-8:
                                normalized = (
                                    float(score["gold_margin"])
                                    - float(target_score["gold_margin"])
                                ) / denominator
                            record = {
                                **score,
                                "result_id": result_id,
                                "experiment": exp["name"],
                                "experiment_type": "patching_sweep",
                                "config_id": config_id,
                                "question_id": target["question_id"],
                                "question_form": "MC",
                                "source_condition": source["condition"],
                                "target_condition": target["condition"],
                                "condition": target["condition"],
                                "component": component,
                                "layer": layer_idx,
                                "head_set": head_set_name,
                                "heads": layer_heads,
                                "position": position,
                                "strength": strength,
                                "source_prediction": source_score["prediction"],
                                "target_baseline_prediction": target_score["prediction"],
                                "source_correct": source_score["is_correct"],
                                "baseline_correct": target_score["is_correct"],
                                "source_gold_margin": source_score["gold_margin"],
                                "baseline_gold_margin": target_score["gold_margin"],
                                "normalized_recovery": normalized,
                                "source_answer_copy": score["predicted_index"]
                                == source_score["predicted_index"],
                            }
                            writer.append(record)
        writer.finalize()

    def run_visual_contrast_reinforcement(
        self, exp: Mapping[str, Any], profile: str
    ) -> None:
        """Amplify the within-sample visual component at selected L22 heads.

        A synthetic text-only view supplies z_null.  Patching the full view with
        strength=-alpha implements z_full + alpha * (z_full - z_null), without
        borrowing an activation or answer state from another example.
        """
        writer = ResultWriter(self.output_dir, str(exp["name"]))
        limit = exp.get(f"{profile}_limit", exp.get("limit"))
        rows = self.rows_filtered(
            ["MC"], exp.get("conditions", ["full"]), limit
        )
        component = str(exp.get("component", "head"))
        layer_idx = int(exp.get("layer", 22))
        position = str(exp.get("position", "query"))
        alphas = [
            float(value)
            for value in exp.get(
                f"{profile}_alphas", exp.get("alphas", [0.1, 0.2, 0.5, 1.0])
            )
        ]
        raw_head_sets = exp.get("head_sets") or [
            {"name": "selected", "heads": exp.get("heads", [])}
        ]
        head_sets: list[tuple[str, list[int]]] = []
        for index, head_set in enumerate(raw_head_sets):
            if not isinstance(head_set, Mapping):
                raise ValueError("Each head_sets entry must contain name and heads")
            pairs = parse_heads(head_set.get("heads"))
            heads = [head for layer, head in pairs if layer == layer_idx]
            if component == "head" and pairs and not heads:
                raise ValueError(
                    f"Head set {head_set.get('name', index)!r} has no layer "
                    f"{layer_idx} heads"
                )
            head_sets.append(
                (str(head_set.get("name", f"head_set_{index}")), heads)
            )

        baseline_config = {
            "control": "baseline",
            "component": component,
            "layer": layer_idx,
            "position": position,
        }
        baseline_config_id = stable_hash(baseline_config)
        setting_specs: list[tuple[str, list[int], float, str]] = []
        for head_set_name, heads in head_sets:
            for alpha in alphas:
                config = {
                    "control": "visual_contrast",
                    "component": component,
                    "layer": layer_idx,
                    "head_set": head_set_name,
                    "heads": heads,
                    "position": position,
                    "alpha": alpha,
                    "null_condition": "text_only",
                }
                setting_specs.append(
                    (head_set_name, heads, alpha, stable_hash(config))
                )

        for row in tqdm(rows, desc=str(exp["name"])):
            baseline_result_id = f"{row['sample_uid']}::{baseline_config_id}"
            pending_specs = [
                spec
                for spec in setting_specs
                if not writer.has(f"{row['sample_uid']}::{spec[3]}")
            ]
            if writer.has(baseline_result_id) and not pending_specs:
                continue
            baseline = self.score_mc(row)
            if not writer.has(baseline_result_id):
                writer.append(
                    {
                        **baseline,
                        "result_id": baseline_result_id,
                        "experiment": exp["name"],
                        "experiment_type": "visual_contrast_reinforcement",
                        "config_id": baseline_config_id,
                        "question_id": row["question_id"],
                        "question_form": "MC",
                        "condition": row["condition"],
                        "control": "baseline",
                        "component": component,
                        "layer": layer_idx,
                        "head_set": "baseline",
                        "heads": [],
                        "position": position,
                        "alpha": 0.0,
                        "patch_strength": 0.0,
                        "baseline_correct": baseline["is_correct"],
                        "baseline_gold_margin": baseline["gold_margin"],
                        "margin_delta": 0.0,
                    }
                )

            if not pending_specs:
                continue
            null_row = self.synthesize_text_only(row)
            null_values = self.capture_values(
                null_row, component, layer_idx, position
            )
            for head_set_name, heads, alpha, config_id in pending_specs:
                result_id = f"{row['sample_uid']}::{config_id}"

                def factory(
                    prepared: PreparedInput,
                    null_values: torch.Tensor = null_values,
                    heads: list[int] = heads,
                    alpha: float = alpha,
                ) -> ComponentPatcher:
                    return ComponentPatcher(
                        self.model,
                        prepared,
                        component,
                        layer_idx,
                        position,
                        null_values,
                        -alpha,
                        self.num_heads,
                        heads,
                    )

                score = self.score_mc(row, factory, cache_baseline=False)
                margin_delta = (
                    float(score["gold_margin"])
                    - float(baseline["gold_margin"])
                    if score.get("gold_margin") is not None
                    and baseline.get("gold_margin") is not None
                    else None
                )
                writer.append(
                    {
                        **score,
                        "result_id": result_id,
                        "experiment": exp["name"],
                        "experiment_type": "visual_contrast_reinforcement",
                        "config_id": config_id,
                        "question_id": row["question_id"],
                        "question_form": "MC",
                        "condition": row["condition"],
                        "control": "visual_contrast",
                        "component": component,
                        "layer": layer_idx,
                        "head_set": head_set_name,
                        "heads": heads,
                        "position": position,
                        "alpha": alpha,
                        "patch_strength": -alpha,
                        "null_condition": "text_only",
                        "baseline_prediction": baseline["prediction"],
                        "baseline_correct": baseline["is_correct"],
                        "baseline_gold_margin": baseline["gold_margin"],
                        "margin_delta": margin_delta,
                    }
                )
        writer.finalize()

    def run_source_control(self, exp: Mapping[str, Any], profile: str) -> None:
        writer = ResultWriter(self.output_dir, str(exp["name"]))
        component = str(exp.get("component", "residual"))
        layers = [int(layer) for layer in exp.get("layers", [31, 35])]
        position = str(exp.get("position", "last"))
        strengths = [float(value) for value in exp.get("strengths", [1.0])]
        controls = list(
            exp.get(
                "controls", ["paired", "same_answer", "different_answer", "random"]
            )
        )
        all_heads = parse_heads(exp.get("heads"))
        pair_filter = str(exp.get("pair_filter", "all"))
        limit = exp.get(f"{profile}_limit", exp.get("limit"))
        directions = exp.get(
            "directions",
            [
                {"source": "full", "target": "text_only"},
                {"source": "text_only", "target": "full"},
            ],
        )

        for direction in directions:
            source_condition = norm_condition(direction["source"])
            target_condition = norm_condition(direction["target"])
            pairs = self.paired_rows(source_condition, target_condition, limit)
            if not pairs:
                continue
            source_pool = [source for source, _target in pairs]
            source_scores = {
                source["sample_uid"]: self.score_mc(source) for source in source_pool
            }
            for source_paired, target in tqdm(
                pairs, desc=f"{exp['name']}:{source_condition}->{target_condition}"
            ):
                target_score = self.score_mc(target)
                paired_source_score = source_scores[source_paired["sample_uid"]]
                if not self.pair_matches_filter(
                    paired_source_score, target_score, pair_filter
                ):
                    continue
                other_sources = [
                    source
                    for source in source_pool
                    if source["question_id"] != target["question_id"]
                ]
                rng = random.Random(
                    f"{self.seed}:{source_condition}:{target_condition}:{target['question_id']}"
                )
                same_answer_candidates = [
                    source
                    for source in other_sources
                    if source_scores[source["sample_uid"]]["predicted_index"]
                    == target_score["predicted_index"]
                ]
                different_answer_candidates = [
                    source
                    for source in other_sources
                    if source_scores[source["sample_uid"]]["predicted_index"]
                    != target_score["predicted_index"]
                ]
                source_label_matched_candidates = [
                    source
                    for source in other_sources
                    if source_scores[source["sample_uid"]]["predicted_index"]
                    == paired_source_score["predicted_index"]
                ]
                source_label_mismatched_candidates = [
                    source
                    for source in other_sources
                    if source_scores[source["sample_uid"]]["predicted_index"]
                    != paired_source_score["predicted_index"]
                ]
                target_gold_indices = set(
                    parse_gold_indices(target, parse_maybe_list(target.get("options_json")))
                )
                target_gold_label_candidates = [
                    source
                    for source in other_sources
                    if source_scores[source["sample_uid"]]["predicted_index"]
                    in target_gold_indices
                ]
                non_target_gold_label_candidates = [
                    source
                    for source in other_sources
                    if source_scores[source["sample_uid"]]["predicted_index"]
                    not in target_gold_indices
                ]

                def choose(candidates: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
                    return rng.choice(list(candidates)) if candidates else None

                selections: dict[str, dict[str, Any] | None] = {
                    "paired": source_paired,
                    "same_answer": choose(same_answer_candidates),
                    "different_answer": choose(different_answer_candidates),
                    "source_label_matched": choose(source_label_matched_candidates),
                    "source_label_mismatched": choose(source_label_mismatched_candidates),
                    "target_gold_label": choose(target_gold_label_candidates),
                    "non_target_gold_label": choose(non_target_gold_label_candidates),
                    "random": rng.choice(other_sources) if other_sources else None,
                }
                for control in controls:
                    source = selections.get(control)
                    if source is None:
                        continue
                    source_score = source_scores[source["sample_uid"]]
                    for layer_idx in layers:
                        layer_heads = [head for layer, head in all_heads if layer == layer_idx]
                        if component == "head" and all_heads and not layer_heads:
                            continue
                        source_values = self.capture_values(
                            source, component, layer_idx, position
                        )
                        for strength in strengths:
                            config = {
                                "direction": f"{source_condition}->{target_condition}",
                                "control": control,
                                "component": component,
                                "layer": layer_idx,
                                "heads": layer_heads,
                                "position": position,
                                "strength": strength,
                            }
                            config_id = stable_hash(config)
                            result_id = f"{target['sample_uid']}::{config_id}"
                            if writer.has(result_id):
                                continue

                            def factory(
                                prepared: PreparedInput,
                                source_values: torch.Tensor = source_values,
                                layer_heads: list[int] = layer_heads,
                            ) -> ComponentPatcher:
                                return ComponentPatcher(
                                    self.model,
                                    prepared,
                                    component,
                                    layer_idx,
                                    position,
                                    source_values,
                                    strength,
                                    self.num_heads,
                                    layer_heads,
                                )

                            score = self.score_mc(target, factory, cache_baseline=False)
                            record = {
                                **score,
                                "result_id": result_id,
                                "experiment": exp["name"],
                                "experiment_type": "source_control",
                                "config_id": config_id,
                                "question_id": target["question_id"],
                                "question_form": "MC",
                                "condition": target_condition,
                                "source_condition": source_condition,
                                "target_condition": target_condition,
                                "source_question_id": source["question_id"],
                                "control": control,
                                "component": component,
                                "layer": layer_idx,
                                "heads": layer_heads,
                                "position": position,
                                "strength": strength,
                                "source_prediction": source_score["prediction"],
                                "paired_source_prediction": paired_source_score["prediction"],
                                "target_gold_indices": sorted(target_gold_indices),
                                "target_baseline_prediction": target_score["prediction"],
                                "source_correct": source_score["is_correct"],
                                "baseline_correct": target_score["is_correct"],
                                "baseline_gold_margin": target_score["gold_margin"],
                                "source_answer_copy": score["predicted_index"]
                                == source_score["predicted_index"],
                            }
                            writer.append(record)
        writer.finalize()

    def cyclic_label_swap(
        self, row: Mapping[str, Any]
    ) -> tuple[dict[str, Any], list[int], list[int]] | None:
        """Rotate option labels while preserving semantic option content.

        The mapping is new-position -> old-position. For a single-label item this
        always moves the correct semantic option to a different answer label.
        """
        options = parse_maybe_list(row.get("options_json"))
        gold_original = parse_gold_indices(row, options)
        if len(options) < 2 or len(gold_original) != 1:
            return None
        rng = random.Random(
            f"{self.seed}:label-swap:{row.get('question_id')}"
        )
        shift = rng.randrange(1, len(options))
        permutation = list(range(shift, len(options))) + list(range(shift))
        new_gold = sorted(permutation.index(index) for index in gold_original)
        result = deepcopy(dict(row))
        result["options_json"] = json.dumps(
            [options[index] for index in permutation], ensure_ascii=False
        )
        new_reference = "/".join(OPTION_LABELS[index] for index in new_gold)
        result["reference"] = new_reference
        result["answer"] = new_reference
        result["sample_uid"] = f"{row.get('sample_uid')}::cyclic_label_swap"
        return result, permutation, gold_original

    def run_label_swap_control(self, exp: Mapping[str, Any], profile: str) -> None:
        """Distinguish semantic transfer from copying the source answer label."""
        writer = ResultWriter(self.output_dir, str(exp["name"]))
        limit = exp.get(f"{profile}_limit", exp.get("limit"))
        pairs = self.paired_rows(
            str(exp.get("source_condition", "full")),
            str(exp.get("target_condition", "shuffled")),
            limit,
        )
        component = str(exp.get("component", "head"))
        layer_idx = int(exp.get("layer", 22))
        position = str(exp.get("position", "query"))
        strength = float(exp.get("strength", 1.0))
        all_heads = parse_heads(exp.get("heads"))
        layer_heads = [head for layer, head in all_heads if layer == layer_idx]
        pair_filter = str(exp.get("pair_filter", "source_only"))
        require_swapped_correct = bool(exp.get("require_swapped_source_correct", True))

        for source, target in tqdm(pairs, desc=str(exp["name"])):
            source_score = self.score_mc(source)
            target_score = self.score_mc(target)
            if not self.pair_matches_filter(source_score, target_score, pair_filter):
                continue
            swapped = self.cyclic_label_swap(source)
            if swapped is None:
                continue
            swapped_source, permutation, source_gold_original = swapped
            swapped_score = self.score_mc(swapped_source)
            if require_swapped_correct and swapped_score.get("is_correct") is not True:
                continue

            controls = [
                ("paired_original", source, source_score, list(range(len(permutation)))),
                ("paired_label_swapped", swapped_source, swapped_score, permutation),
            ]
            target_gold = parse_gold_indices(
                target, parse_maybe_list(target.get("options_json"))
            )
            for control, source_row, control_source_score, semantic_mapping in controls:
                source_values = self.capture_values(
                    source_row, component, layer_idx, position
                )
                config = {
                    "control": control,
                    "component": component,
                    "layer": layer_idx,
                    "heads": layer_heads,
                    "position": position,
                    "strength": strength,
                }
                config_id = stable_hash(config)
                result_id = f"{target['sample_uid']}::{config_id}"
                if writer.has(result_id):
                    continue

                def factory(
                    prepared: PreparedInput,
                    source_values: torch.Tensor = source_values,
                ) -> ComponentPatcher:
                    return ComponentPatcher(
                        self.model,
                        prepared,
                        component,
                        layer_idx,
                        position,
                        source_values,
                        strength,
                        self.num_heads,
                        layer_heads,
                    )

                score = self.score_mc(target, factory, cache_baseline=False)
                source_predicted_index = int(control_source_score["predicted_index"])
                semantic_source_index = int(semantic_mapping[source_predicted_index])
                record = {
                    **score,
                    "result_id": result_id,
                    "experiment": exp["name"],
                    "experiment_type": "label_swap_control",
                    "config_id": config_id,
                    "question_id": target["question_id"],
                    "question_form": "MC",
                    "condition": target["condition"],
                    "source_condition": source["condition"],
                    "target_condition": target["condition"],
                    "control": control,
                    "component": component,
                    "layer": layer_idx,
                    "heads": layer_heads,
                    "position": position,
                    "strength": strength,
                    "permutation_new_to_old": semantic_mapping,
                    "source_gold_original_indices": source_gold_original,
                    "target_gold_indices": target_gold,
                    "source_prediction": control_source_score["prediction"],
                    "source_predicted_index": source_predicted_index,
                    "source_semantic_prediction_original_index": semantic_source_index,
                    "target_baseline_prediction": target_score["prediction"],
                    "baseline_correct": target_score["is_correct"],
                    "baseline_gold_margin": target_score["gold_margin"],
                    "source_answer_label_copy": score["predicted_index"]
                    == source_predicted_index,
                    "semantic_option_transfer": score["predicted_index"]
                    == semantic_source_index,
                    "label_and_semantic_indices_differ": source_predicted_index
                    != semantic_source_index,
                }
                writer.append(record)
        writer.finalize()

    @staticmethod
    def pooled_vector(values: torch.Tensor) -> torch.Tensor:
        if values.numel() == 0:
            return values.flatten().float()
        return values.float().mean(dim=0)

    def score_with_patch_and_capture(
        self,
        target: Mapping[str, Any],
        sender_component: str,
        sender_layer: int,
        sender_position: str,
        sender_values: torch.Tensor,
        sender_heads: Sequence[int],
        receiver_layer: int,
        receiver_position: str,
        strength: float,
    ) -> tuple[dict[str, Any], torch.Tensor]:
        prepared = self.prepare(target)
        patcher = ComponentPatcher(
            self.model,
            prepared,
            sender_component,
            sender_layer,
            sender_position,
            sender_values,
            strength,
            self.num_heads,
            sender_heads,
        )
        receiver = ComponentCapture(
            self.model, prepared, "residual", receiver_layer, receiver_position
        )
        with ExitStack() as stack:
            stack.enter_context(patcher)
            stack.enter_context(receiver)
            score = self._score_prepared(prepared)
        if receiver.values is None:
            raise RuntimeError("Receiver capture hook did not run")
        return score, receiver.values

    def run_bridge(self, exp: Mapping[str, Any], profile: str) -> None:
        writer = ResultWriter(self.output_dir, str(exp["name"]))
        limit = exp.get(f"{profile}_limit", exp.get("limit"))
        pairs = self.paired_rows(
            str(exp["source_condition"]), str(exp["target_condition"]), limit
        )
        sender_layer = int(exp.get("sender_layer", 22))
        raw_sender_heads = list(exp.get("sender_heads") or [])
        if raw_sender_heads and all(
            isinstance(head, (int, float)) for head in raw_sender_heads
        ):
            sender_heads = [int(head) for head in raw_sender_heads]
        else:
            sender_heads = [
                head
                for layer, head in parse_heads(raw_sender_heads)
                if layer == sender_layer
            ]
        sender_component = str(exp.get("sender_component", "head"))
        sender_positions = list(exp.get("sender_positions", ["last", "query"]))
        receiver_layers = [int(layer) for layer in exp.get("receiver_layers", [31, 35])]
        receiver_position = str(exp.get("receiver_position", "last"))
        strength = float(exp.get("strength", 1.0))
        pair_filter = str(exp.get("pair_filter", "all"))

        for source, target in tqdm(pairs, desc=str(exp["name"])):
            source_score = self.score_mc(source)
            target_score = self.score_mc(target)
            if not self.pair_matches_filter(source_score, target_score, pair_filter):
                continue
            for sender_position in sender_positions:
                sender_values = self.capture_values(
                    source, sender_component, sender_layer, sender_position
                )
                for receiver_layer in receiver_layers:
                    source_receiver = self.capture_values(
                        source, "residual", receiver_layer, receiver_position
                    )
                    target_receiver = self.capture_values(
                        target, "residual", receiver_layer, receiver_position
                    )
                    config = {
                        "sender_layer": sender_layer,
                        "sender_heads": sender_heads,
                        "sender_position": sender_position,
                        "receiver_layer": receiver_layer,
                        "receiver_position": receiver_position,
                        "strength": strength,
                    }
                    config_id = stable_hash(config)
                    result_id = f"{target['sample_uid']}::{config_id}"
                    if writer.has(result_id):
                        continue
                    score, patched_receiver = self.score_with_patch_and_capture(
                        target,
                        sender_component,
                        sender_layer,
                        sender_position,
                        sender_values,
                        sender_heads,
                        receiver_layer,
                        receiver_position,
                        strength,
                    )
                    source_vector = self.pooled_vector(source_receiver)
                    target_vector = self.pooled_vector(target_receiver)
                    patched_vector = self.pooled_vector(patched_receiver)
                    baseline_distance = float(torch.linalg.vector_norm(source_vector - target_vector))
                    patched_distance = float(torch.linalg.vector_norm(source_vector - patched_vector))
                    state_recovery = (
                        (baseline_distance - patched_distance) / baseline_distance
                        if baseline_distance > 1e-8
                        else float("nan")
                    )
                    source_target_cosine = float(
                        F.cosine_similarity(source_vector, target_vector, dim=0).item()
                    )
                    source_patched_cosine = float(
                        F.cosine_similarity(source_vector, patched_vector, dim=0).item()
                    )
                    margin_denominator = (
                        float(source_score["gold_margin"])
                        - float(target_score["gold_margin"])
                        if source_score.get("gold_margin") is not None
                        and target_score.get("gold_margin") is not None
                        else float("nan")
                    )
                    logit_recovery = float("nan")
                    if score.get("gold_margin") is not None and abs(margin_denominator) > 1e-8:
                        logit_recovery = (
                            float(score["gold_margin"])
                            - float(target_score["gold_margin"])
                        ) / margin_denominator
                    record = {
                        **score,
                        "result_id": result_id,
                        "experiment": exp["name"],
                        "experiment_type": "bridge",
                        "config_id": config_id,
                        "question_id": target["question_id"],
                        "question_form": "MC",
                        "condition": target["condition"],
                        "source_condition": source["condition"],
                        "target_condition": target["condition"],
                        "sender_component": sender_component,
                        "sender_layer": sender_layer,
                        "sender_heads": sender_heads,
                        "sender_position": sender_position,
                        "receiver_layer": receiver_layer,
                        "receiver_position": receiver_position,
                        "strength": strength,
                        "baseline_correct": target_score["is_correct"],
                        "source_correct": source_score["is_correct"],
                        "baseline_gold_margin": target_score["gold_margin"],
                        "state_distance_baseline": baseline_distance,
                        "state_distance_patched": patched_distance,
                        "state_recovery": state_recovery,
                        "source_target_cosine": source_target_cosine,
                        "source_patched_cosine": source_patched_cosine,
                        "normalized_recovery": logit_recovery,
                        "source_answer_copy": score["predicted_index"]
                        == source_score["predicted_index"],
                    }
                    writer.append(record)
        writer.finalize()

    def run_dynamic_vhd(self, exp: Mapping[str, Any], profile: str) -> None:
        writer = ResultWriter(self.output_dir, str(exp["name"]))
        limit = exp.get(f"{profile}_limit", exp.get("limit"))
        pairs = self.paired_rows(
            str(exp.get("full_condition", "full")),
            str(exp.get("text_condition", "text_only")),
            limit,
        )
        candidate_layers = [int(layer) for layer in exp.get("candidate_layers", [17, 22, 27, 31, 35])]
        top_k = int(exp.get("top_k", 4))
        magnitude = abs(float(exp.get("alpha", 0.2)))
        routes = list(exp.get("routes", ["confidence"]))
        phase = str(exp.get("phase", "first_token"))
        position = str(exp.get("position", "last"))

        for full_row, text_row in tqdm(pairs, desc=str(exp["name"])):
            full_score = self.score_mc(full_row)
            text_score = self.score_mc(text_row)
            ranked: list[tuple[float, int, int]] = []
            for layer_idx in candidate_layers:
                full_values = self.capture_values(full_row, "head", layer_idx, "last")
                text_values = self.capture_values(text_row, "head", layer_idx, "last")
                full_vector = self.pooled_vector(full_values)
                text_vector = self.pooled_vector(text_values)
                if full_vector.numel() % self.num_heads:
                    raise RuntimeError(
                        f"Head output width {full_vector.numel()} is not divisible by {self.num_heads}"
                    )
                head_dim = full_vector.numel() // self.num_heads
                differences = (full_vector - text_vector).reshape(self.num_heads, head_dim)
                distances = torch.linalg.vector_norm(differences, dim=-1)
                ranked.extend(
                    (float(distances[head].item()), layer_idx, head)
                    for head in range(self.num_heads)
                )
            ranked.sort(reverse=True)
            selected = [(layer, head) for _score, layer, head in ranked[:top_k]]
            selected_scores = [score for score, _layer, _head in ranked[:top_k]]

            for route in routes:
                if route == "always_boost":
                    alpha = magnitude
                    route_reason = "fixed_boost"
                elif route == "always_suppress":
                    alpha = -magnitude
                    route_reason = "fixed_suppress"
                elif route == "confidence":
                    if full_score["predicted_index"] == text_score["predicted_index"]:
                        alpha = 0.0
                        route_reason = "agreement_no_intervention"
                    elif float(full_score["top1_margin"]) >= float(text_score["top1_margin"]):
                        alpha = magnitude
                        route_reason = "full_higher_margin"
                    else:
                        alpha = -magnitude
                        route_reason = "text_higher_margin"
                elif route == "oracle":
                    if full_score["is_correct"] is True and text_score["is_correct"] is False:
                        alpha = magnitude
                        route_reason = "oracle_image_help"
                    elif text_score["is_correct"] is True and full_score["is_correct"] is False:
                        alpha = -magnitude
                        route_reason = "oracle_image_harm"
                    else:
                        alpha = 0.0
                        route_reason = "oracle_no_intervention"
                else:
                    raise ValueError(f"Unsupported VHD route: {route}")
                config = {
                    "candidate_layers": candidate_layers,
                    "top_k": top_k,
                    "route": route,
                    "alpha": magnitude,
                    "phase": phase,
                    "position": position,
                }
                config_id = stable_hash(config)
                result_id = f"{full_row['sample_uid']}::{config_id}"
                if writer.has(result_id):
                    continue

                def factory(prepared: PreparedInput) -> HeadOutputScaler:
                    return HeadOutputScaler(
                        self.model,
                        prepared,
                        selected,
                        alpha,
                        self.num_heads,
                        phase,
                        position,
                    )

                score = self.score_mc(full_row, factory, cache_baseline=False)
                record = {
                    **score,
                    "result_id": result_id,
                    "experiment": exp["name"],
                    "experiment_type": "dynamic_vhd",
                    "config_id": config_id,
                    "question_id": full_row["question_id"],
                    "question_form": "MC",
                    "condition": full_row["condition"],
                    "route": route,
                    "route_reason": route_reason,
                    "applied_alpha": alpha,
                    "selected_heads": selected,
                    "selected_vhd": selected_scores,
                    "full_baseline_prediction": full_score["prediction"],
                    "text_baseline_prediction": text_score["prediction"],
                    "full_baseline_correct": full_score["is_correct"],
                    "text_baseline_correct": text_score["is_correct"],
                    "baseline_correct": full_score["is_correct"],
                    "baseline_gold_margin": full_score["gold_margin"],
                    "full_text_agree": full_score["predicted_index"]
                    == text_score["predicted_index"],
                    "phase": phase,
                    "position": position,
                }
                writer.append(record)
        writer.finalize()

    @staticmethod
    def _head_vectors(values: torch.Tensor, num_heads: int) -> torch.Tensor:
        vector = ExperimentRunner.pooled_vector(values).float()
        if vector.numel() % num_heads:
            raise RuntimeError(
                f"Head output width {vector.numel()} is not divisible by {num_heads}"
            )
        return vector.reshape(num_heads, vector.numel() // num_heads)

    def run_faithful_vhr(self, exp: Mapping[str, Any], profile: str) -> None:
        """VHR-style per-sample selection with the paper's negative-VHD filter.

        `vhr_reinforce` scales the actual full-image head output. `vhd_suppress`
        is an explicitly named SPIN-inspired proxy which suppresses the VHD-low
        complement; it does not claim to reproduce SPIN's attention-weight rule.
        """
        writer = ResultWriter(self.output_dir, str(exp["name"]))
        limit = exp.get(f"{profile}_limit", exp.get("limit"))
        rows = self.rows_filtered(["MC"], exp.get("conditions", ["full"]), limit)
        layers = [int(value) for value in exp.get("layers", [22])]
        position = str(exp.get("selection_position", "last"))
        phase = str(exp.get("phase", "first_token"))
        actions = list(exp.get("actions", ["vhr_reinforce"]))
        factors = [
            float(value)
            for value in exp.get(
                f"{profile}_scale_factors", exp.get("scale_factors", [1.1, 1.2])
            )
        ]
        suppression_factors = [
            float(value)
            for value in exp.get(
                f"{profile}_suppression_factors",
                exp.get("suppression_factors", [0.5]),
            )
        ]

        for row in tqdm(rows, desc=str(exp["name"])):
            baseline = self.score_mc(row)
            text_row = self.synthesize_text_only(row)
            selected: list[tuple[int, int]] = []
            rejected: list[tuple[int, int]] = []
            vhd_by_head: dict[str, float] = {}
            for layer_idx in layers:
                full_heads = self._head_vectors(
                    self.capture_values(row, "head", layer_idx, position), self.num_heads
                )
                text_heads = self._head_vectors(
                    self.capture_values(text_row, "head", layer_idx, position),
                    self.num_heads,
                )
                vhd = torch.linalg.vector_norm(full_heads - text_heads, dim=-1)
                text_norm = torch.linalg.vector_norm(text_heads, dim=-1)
                negative = (vhd > vhd.mean() + vhd.std(unbiased=False)) & (
                    text_norm > text_norm.mean() + text_norm.std(unbiased=False)
                )
                filtered = vhd.clone()
                filtered[negative] = 0.0
                threshold = torch.median(filtered)
                for head in range(self.num_heads):
                    vhd_by_head[f"{layer_idx}:{head}"] = float(vhd[head].item())
                    if bool(negative[head]):
                        rejected.append((layer_idx, head))
                    elif float(filtered[head].item()) > float(threshold.item()):
                        selected.append((layer_idx, head))
            all_heads = [(layer, head) for layer in layers for head in range(self.num_heads)]
            selected_set = set(selected)
            complement = [pair for pair in all_heads if pair not in selected_set]
            for action in actions:
                action_factors = factors if action == "vhr_reinforce" else suppression_factors
                for factor in action_factors:
                    if action == "vhr_reinforce":
                        intervention_heads = selected
                        scaler_alpha = factor - 1.0
                    elif action == "vhd_suppress":
                        intervention_heads = complement
                        scaler_alpha = factor - 1.0
                    else:
                        raise ValueError(f"Unsupported VHR action: {action}")
                    config = {
                        "layers": layers,
                        "action": action,
                        "factor": factor,
                        "selection_position": position,
                        "phase": phase,
                    }
                    config_id = stable_hash(config)
                    result_id = f"{row['sample_uid']}::{config_id}"
                    if writer.has(result_id):
                        continue

                    def factory(
                        prepared: PreparedInput,
                        intervention_heads: list[tuple[int, int]] = intervention_heads,
                        scaler_alpha: float = scaler_alpha,
                    ) -> HeadOutputScaler:
                        return HeadOutputScaler(
                            self.model,
                            prepared,
                            intervention_heads,
                            scaler_alpha,
                            self.num_heads,
                            phase,
                            "last",
                        )

                    score = self.score_mc(row, factory, cache_baseline=False)
                    writer.append(
                        {
                            **score,
                            "result_id": result_id,
                            "experiment": exp["name"],
                            "experiment_type": "faithful_vhr",
                            "config_id": config_id,
                            "question_id": row["question_id"],
                            "question_form": "MC",
                            "condition": row["condition"],
                            "control": action,
                            "layers": layers,
                            "selected_heads": selected,
                            "intervention_heads": intervention_heads,
                            "negative_sensitivity_heads": rejected,
                            "vhd_by_head": vhd_by_head,
                            "factor": factor,
                            "alpha": scaler_alpha,
                            "phase": phase,
                            "position": "last",
                            "baseline_prediction": baseline["prediction"],
                            "baseline_correct": baseline["is_correct"],
                            "baseline_gold_margin": baseline["gold_margin"],
                            "margin_delta": (
                                float(score["gold_margin"])
                                - float(baseline["gold_margin"])
                                if score.get("gold_margin") is not None
                                and baseline.get("gold_margin") is not None
                                else None
                            ),
                        }
                    )
        writer.finalize()

    def run_contrastive_decoding(self, exp: Mapping[str, Any], profile: str) -> None:
        """MC adaptation of VCD: combine saved candidate logits offline."""
        writer = ResultWriter(self.output_dir, str(exp["name"]))
        limit = exp.get(f"{profile}_limit", exp.get("limit"))
        rows = self.rows_filtered(["MC"], exp.get("conditions", ["full"]), limit)
        alphas = [
            float(value)
            for value in exp.get(f"{profile}_alphas", exp.get("alphas", [1.0]))
        ]
        betas = [
            float(value)
            for value in exp.get(
                f"{profile}_plausibility_betas",
                exp.get("plausibility_betas", [0.0, 0.1]),
            )
        ]
        priors = list(exp.get("priors", ["text_only", "gaussian_noise"]))
        noise_sigma = float(exp.get("noise_sigma", 0.2))
        for row in tqdm(rows, desc=str(exp["name"])):
            options = parse_maybe_list(row.get("options_json"))
            baseline = self.score_mc(row)
            full_values = [float(value) for value in baseline["candidate_log_probs"]]
            prior_scores: dict[str, dict[str, Any]] = {}
            if "text_only" in priors:
                prior_scores["text_only"] = self.score_mc(self.synthesize_text_only(row))
            if "gaussian_noise" in priors:
                prior_scores["gaussian_noise"] = self.score_mc(
                    self.synthesize_noisy(row, noise_sigma)
                )
            max_full = max(full_values)
            for prior_name, prior_score in prior_scores.items():
                prior_values = [
                    float(value) for value in prior_score["candidate_log_probs"]
                ]
                for alpha in alphas:
                    raw_combined = [
                        (1.0 + alpha) * full - alpha * prior
                        for full, prior in zip(full_values, prior_values)
                    ]
                    for beta in betas:
                        combined = list(raw_combined)
                        if beta > 0:
                            log_threshold = math.log(beta)
                            combined = [
                                value
                                if full >= max_full + log_threshold
                                else float("-inf")
                                for value, full in zip(combined, full_values)
                            ]
                        score = score_candidate_values(row, options, combined)
                        config = {
                            "prior": prior_name,
                            "alpha": alpha,
                            "plausibility_beta": beta,
                            "noise_sigma": noise_sigma if prior_name == "gaussian_noise" else None,
                        }
                        config_id = stable_hash(config)
                        result_id = f"{row['sample_uid']}::{config_id}"
                        if writer.has(result_id):
                            continue
                        writer.append(
                            {
                                **score,
                                "result_id": result_id,
                                "experiment": exp["name"],
                                "experiment_type": "contrastive_decoding",
                                "config_id": config_id,
                                "question_id": row["question_id"],
                                "question_form": "MC",
                                "condition": row["condition"],
                                "control": f"vcd_{prior_name}",
                                "prior": prior_name,
                                "alpha": alpha,
                                "plausibility_beta": beta,
                                "full_candidate_log_probs": full_values,
                                "prior_candidate_log_probs": prior_values,
                                "baseline_prediction": baseline["prediction"],
                                "baseline_correct": baseline["is_correct"],
                                "baseline_gold_margin": baseline["gold_margin"],
                                "margin_delta": (
                                    float(score["gold_margin"])
                                    - float(baseline["gold_margin"])
                                    if score.get("gold_margin") is not None
                                    and baseline.get("gold_margin") is not None
                                    else None
                                ),
                            }
                        )
        writer.finalize()

    def run_gated_label_correction(
        self, exp: Mapping[str, Any], profile: str
    ) -> None:
        """Keep label-token scoring and selectively invoke stronger correctors.

        The detector is label-free: normalized Full/Text head-output distance,
        Full/Text label-distribution JS divergence, and baseline uncertainty.
        Correctors are label-logit contrast, semantic permutation averaging,
        and an option-hidden evidence-first second pass.
        """
        writer = ResultWriter(self.output_dir, str(exp["name"]))

        def profile_value(name: str, default: Any) -> Any:
            return exp.get(f"{profile}_{name}", exp.get(name, default))

        limit = profile_value("limit", None)
        rows = self.rows_filtered(["MC"], exp.get("conditions", ["full"]), limit)
        alphas = [
            float(value)
            for value in profile_value("alphas", [0.05, 0.1, 0.25, 0.5, 1.0])
        ]
        detector_heads = parse_heads(exp.get("detector_heads", ["63:63", "63:45"]))
        random_detector_heads = parse_heads(
            exp.get("random_detector_heads", ["63:28", "63:29"])
        )
        detector_position = str(exp.get("detector_position", "last"))
        head_threshold = float(exp.get("head_threshold", 0.34670647978782654))
        js_quantile = float(exp.get("js_quantile", 0.8))
        margin_quantile = float(exp.get("margin_quantile", 0.2))
        permutation_count = int(profile_value("permutation_count", 3))
        run_evidence = bool(profile_value("run_evidence", True))
        evidence_prompt_key = str(exp.get("evidence_prompt_key", "MC_EVIDENCE_1"))
        evidence_max_new_tokens = int(exp.get("evidence_max_new_tokens", 96))

        selected_by_layer: dict[int, list[int]] = {}
        random_by_layer: dict[int, list[int]] = {}
        for layer_idx, head_idx in detector_heads:
            selected_by_layer.setdefault(layer_idx, []).append(head_idx)
        for layer_idx, head_idx in random_detector_heads:
            random_by_layer.setdefault(layer_idx, []).append(head_idx)
        capture_layers = sorted(set(selected_by_layer) | set(random_by_layer))

        def aggregate_signal(
            full_by_layer: Mapping[int, torch.Tensor],
            text_by_layer: Mapping[int, torch.Tensor],
            heads_by_layer: Mapping[int, Sequence[int]],
        ) -> float:
            values: list[float] = []
            for layer_idx, head_indices in heads_by_layer.items():
                full_heads = self._head_vectors(
                    full_by_layer[layer_idx], self.num_heads
                ).float()
                text_heads = self._head_vectors(
                    text_by_layer[layer_idx], self.num_heads
                ).float()
                for head_idx in head_indices:
                    if not 0 <= head_idx < self.num_heads:
                        raise ValueError(f"Detector head out of range: {layer_idx}:{head_idx}")
                    difference = torch.linalg.vector_norm(
                        full_heads[head_idx] - text_heads[head_idx]
                    )
                    denominator = torch.linalg.vector_norm(full_heads[head_idx]).clamp_min(1e-6)
                    values.append(float((difference / denominator).item()))
            return sum(values) / len(values) if values else 0.0

        payloads: list[dict[str, Any]] = []
        for row in tqdm(rows, desc=f"{exp['name']}:detector"):
            text_row = self.row_for_condition(row, "text_only") or self.synthesize_text_only(row)
            full_score = self.score_mc(row)
            text_score = self.score_mc(text_row)
            full_values = {
                layer_idx: self.capture_values(
                    row, "head", layer_idx, detector_position
                )
                for layer_idx in capture_layers
            }
            text_values = {
                layer_idx: self.capture_values(
                    text_row, "head", layer_idx, detector_position
                )
                for layer_idx in capture_layers
            }
            full_logits = [
                float(value) for value in full_score["candidate_log_probs"]
            ]
            text_logits = [
                float(value) for value in text_score["candidate_log_probs"]
            ]
            payloads.append(
                {
                    "row": dict(row),
                    "text_row": text_row,
                    "full": full_score,
                    "text": text_score,
                    "head_signal": aggregate_signal(
                        full_values, text_values, selected_by_layer
                    ),
                    "random_signal": aggregate_signal(
                        full_values, text_values, random_by_layer
                    ),
                    "label_js": js_divergence(full_logits, text_logits),
                }
            )

        def quantile(values: Sequence[float], q: float) -> float:
            if not values:
                return float("nan")
            return float(torch.quantile(torch.tensor(values, dtype=torch.float32), q).item())

        js_threshold = quantile(
            [float(payload["label_js"]) for payload in payloads], js_quantile
        )
        margin_threshold = quantile(
            [float(payload["full"]["top1_margin"]) for payload in payloads],
            margin_quantile,
        )
        selected_coverage = (
            sum(float(payload["head_signal"]) >= head_threshold for payload in payloads)
            / len(payloads)
            if payloads
            else 0.0
        )
        random_threshold = quantile(
            [float(payload["random_signal"]) for payload in payloads],
            max(0.0, min(1.0, 1.0 - selected_coverage)),
        )
        detector_summary = {
            "n": len(payloads),
            "detector_heads": detector_heads,
            "random_detector_heads": random_detector_heads,
            "head_threshold": head_threshold,
            "head_gate_coverage": selected_coverage,
            "js_quantile": js_quantile,
            "js_threshold": js_threshold,
            "margin_quantile": margin_quantile,
            "margin_threshold": margin_threshold,
            "random_matched_threshold": random_threshold,
        }
        with (writer.root / "detector_thresholds.json").open(
            "w", encoding="utf-8"
        ) as stream:
            json.dump(jsonable(detector_summary), stream, ensure_ascii=False, indent=2)

        def gates(payload: Mapping[str, Any]) -> dict[str, bool]:
            high_head = float(payload["head_signal"]) >= head_threshold
            high_js = float(payload["label_js"]) >= js_threshold
            low_margin = float(payload["full"]["top1_margin"]) <= margin_threshold
            return {
                "always": True,
                "head": high_head,
                "js": high_js,
                "head_or_js": high_head or high_js,
                "adaptive": high_head or high_js or low_margin,
                "random": float(payload["random_signal"]) >= random_threshold,
            }

        def append_result(
            payload: Mapping[str, Any],
            score: Mapping[str, Any],
            *,
            control: str,
            config: Mapping[str, Any],
            gate_name: str,
            gate_applied: bool,
            extra: Mapping[str, Any] | None = None,
        ) -> None:
            row = payload["row"]
            baseline = payload["full"]
            config_id = stable_hash(config)
            writer.append(
                {
                    **score,
                    "result_id": f"{row['sample_uid']}::{config_id}",
                    "experiment": exp["name"],
                    "experiment_type": "gated_label_correction",
                    "config_id": config_id,
                    "question_id": row["question_id"],
                    "question_form": "MC",
                    "condition": row["condition"],
                    "control": control,
                    "scoring_method": "label_token",
                    "detector": gate_name,
                    "gate_type": gate_name,
                    "gate_applied": gate_applied,
                    "head_visual_signal": payload["head_signal"],
                    "random_head_visual_signal": payload["random_signal"],
                    "full_text_label_js": payload["label_js"],
                    "head_threshold": head_threshold,
                    "js_threshold": js_threshold,
                    "margin_threshold": margin_threshold,
                    "random_threshold": random_threshold,
                    "baseline_prediction": baseline["prediction"],
                    "baseline_correct": baseline["is_correct"],
                    "baseline_gold_margin": baseline["gold_margin"],
                    "margin_delta": (
                        float(score["gold_margin"]) - float(baseline["gold_margin"])
                        if score.get("gold_margin") is not None
                        and baseline.get("gold_margin") is not None
                        else None
                    ),
                    **dict(extra or {}),
                }
            )

        gate_names = ["always", "head", "js", "head_or_js", "adaptive", "random"]
        for payload in tqdm(payloads, desc=f"{exp['name']}:correct"):
            row = payload["row"]
            options = parse_maybe_list(row.get("options_json"))
            full_logits = [
                float(value) for value in payload["full"]["candidate_log_probs"]
            ]
            text_logits = [
                float(value) for value in payload["text"]["candidate_log_probs"]
            ]
            active = gates(payload)
            baseline_config = {"method": "label_token_baseline"}
            append_result(
                payload,
                payload["full"],
                control="label_token_baseline",
                config=baseline_config,
                gate_name="none",
                gate_applied=False,
            )

            for alpha in alphas:
                contrast_logits = [
                    (1.0 + alpha) * full - alpha * text
                    for full, text in zip(full_logits, text_logits)
                ]
                for gate_name in gate_names:
                    applied = active[gate_name]
                    values = contrast_logits if applied else full_logits
                    score = score_candidate_values(row, options, values)
                    config = {
                        "method": "label_logit_contrast",
                        "alpha": alpha,
                        "gate": gate_name,
                        "head_threshold": head_threshold if "head" in gate_name else None,
                        "js_threshold": js_threshold if "js" in gate_name else None,
                        "margin_threshold": (
                            margin_threshold if gate_name == "adaptive" else None
                        ),
                        "random_threshold": (
                            random_threshold if gate_name == "random" else None
                        ),
                    }
                    append_result(
                        payload,
                        score,
                        control=f"label_contrast_{gate_name}",
                        config=config,
                        gate_name=gate_name,
                        gate_applied=applied,
                        extra={
                            "alpha": alpha,
                            "prior": "text_only",
                            "full_candidate_log_probs": full_logits,
                            "prior_candidate_log_probs": text_logits,
                        },
                    )

            permutation_gate_names = ["head", "head_or_js", "adaptive", "random"]
            if any(active[name] for name in permutation_gate_names):
                semantic_scores: list[list[float]] = []
                permutations: list[list[int]] = []
                for permutation_index in range(permutation_count):
                    permuted, permutation, _gold = self.permute_options(
                        row, permutation_index, self.seed
                    )
                    permuted_score = self.score_mc(permuted)
                    restored = [0.0] * len(options)
                    for new_index, old_index in enumerate(permutation):
                        restored[old_index] = float(
                            permuted_score["candidate_log_probs"][new_index]
                        )
                    semantic_scores.append(restored)
                    permutations.append(permutation)
                permutation_values = (
                    torch.tensor(semantic_scores, dtype=torch.float32)
                    .mean(dim=0)
                    .tolist()
                )
            else:
                permutation_values = full_logits
                permutations = []
            for gate_name in permutation_gate_names:
                applied = active[gate_name]
                values = permutation_values if applied else full_logits
                score = score_candidate_values(row, options, values)
                config = {
                    "method": "permutation_ensemble",
                    "gate": gate_name,
                    "permutation_count": permutation_count,
                    "head_threshold": head_threshold if "head" in gate_name else None,
                    "js_threshold": js_threshold if "js" in gate_name else None,
                    "margin_threshold": (
                        margin_threshold if gate_name == "adaptive" else None
                    ),
                    "random_threshold": random_threshold if gate_name == "random" else None,
                }
                append_result(
                    payload,
                    score,
                    control=f"permutation_{gate_name}",
                    config=config,
                    gate_name=gate_name,
                    gate_applied=applied,
                    extra={
                        "permutation_count": permutation_count,
                        "permutations_new_to_old": permutations if applied else [],
                    },
                )

            evidence_gate_names = ["head", "head_or_js", "adaptive"]
            evidence_text = ""
            evidence_score: dict[str, Any] | None = None
            if run_evidence and any(active[name] for name in evidence_gate_names):
                evidence_text = self.generate(
                    row,
                    prompt_key_override=evidence_prompt_key,
                    max_new_tokens_override=evidence_max_new_tokens,
                )
                evidence_row = deepcopy(dict(row))
                evidence_row["question"] = (
                    f"{row.get('question', '')}\n"
                    f"이미지에서 확인된 근거: {evidence_text}"
                )
                evidence_row["sample_uid"] = (
                    f"{row.get('sample_uid')}::evidence::{stable_hash(evidence_text, 8)}"
                )
                evidence_score = self.score_mc(evidence_row)
            for gate_name in evidence_gate_names:
                applied = run_evidence and active[gate_name]
                score = evidence_score if applied and evidence_score is not None else payload["full"]
                config = {
                    "method": "evidence_first",
                    "gate": gate_name,
                    "evidence_prompt_key": evidence_prompt_key,
                    "head_threshold": head_threshold if "head" in gate_name else None,
                    "js_threshold": js_threshold if "js" in gate_name else None,
                    "margin_threshold": (
                        margin_threshold if gate_name == "adaptive" else None
                    ),
                }
                append_result(
                    payload,
                    score,
                    control=f"evidence_{gate_name}",
                    config=config,
                    gate_name=gate_name,
                    gate_applied=bool(applied),
                    extra={
                        "evidence_text": evidence_text if applied else "",
                        "evidence_prompt_key": evidence_prompt_key,
                    },
                )
        writer.finalize()

    def run_detector_disambiguation(
        self, exp: Mapping[str, Any], profile: str
    ) -> None:
        """Separate head-detector information from ordinary output uncertainty.

        Calibration learns only thresholds and small diagnostic logistic models.
        Evaluation reuses the frozen calibration artifact.  Every requested layer
        is captured in the same Full/Text forward pair; all head-pair null tests
        are then computed on CPU without additional model calls.
        """
        writer = ResultWriter(self.output_dir, str(exp["name"]))

        def profile_value(name: str, default: Any) -> Any:
            return exp.get(f"{profile}_{name}", exp.get(name, default))

        mode = os.path.expandvars(str(exp.get("mode", "calibration"))).strip().lower()
        if mode not in {"calibration", "evaluation"}:
            raise ValueError(
                "detector_disambiguation mode must be 'calibration' or 'evaluation'"
            )
        limit = profile_value("limit", None)
        rows = self.rows_filtered(["MC"], exp.get("conditions", ["full"]), limit)
        configured_alpha = float(exp.get("alpha", 1.0))
        alpha_candidates = [
            float(value)
            for value in profile_value("alphas", [configured_alpha])
        ]
        if configured_alpha not in alpha_candidates:
            alpha_candidates.append(configured_alpha)
        alpha_candidates = sorted(set(alpha_candidates))
        alpha = configured_alpha
        detector_position = str(exp.get("detector_position", "last"))
        detector_heads = parse_heads(exp.get("detector_heads", ["63:63", "63:45"]))
        if not detector_heads:
            raise ValueError("detector_heads cannot be empty")
        selected_layers = {int(layer) for layer, _head in detector_heads}
        if len(selected_layers) != 1:
            raise ValueError("detector_heads must all come from one selected layer")
        selected_layer = next(iter(selected_layers))
        selected_head_indices = sorted({int(head) for _layer, head in detector_heads})
        if len(selected_head_indices) < 1:
            raise ValueError("At least one selected detector head is required")
        control_layers = [
            int(value) for value in exp.get("control_layers", [15, 31, 47, 55])
        ]
        capture_layers = sorted(set(control_layers + [selected_layer]))
        decoder, _decoder_path = decoder_layers(self.model)
        for layer_idx in capture_layers:
            if not 0 <= layer_idx < len(decoder):
                raise ValueError(f"Detector control layer out of range: {layer_idx}")
        for head_idx in selected_head_indices:
            if not 0 <= head_idx < self.num_heads:
                raise ValueError(f"Detector head out of range: {selected_layer}:{head_idx}")

        coverages = sorted(
            {
                float(value)
                for value in profile_value(
                    "coverages", [0.1, 0.2, 0.3, 0.427184466, 0.5, 0.6, 1.0]
                )
            }
        )
        if not coverages or any(not 0.0 < value <= 1.0 for value in coverages):
            raise ValueError("coverages must contain values in (0, 1]")
        primary_coverage = float(exp.get("primary_coverage", 0.42718446601941745))
        if not 0.0 < primary_coverage <= 1.0:
            raise ValueError("primary_coverage must be in (0, 1]")
        if all(abs(value - primary_coverage) > 1e-9 for value in coverages):
            coverages.append(primary_coverage)
            coverages.sort()
        pair_limit_raw = profile_value("pair_limit", None)
        pair_limit = int(pair_limit_raw) if pair_limit_raw not in (None, "") else None
        bootstrap_iterations = int(profile_value("bootstrap_iterations", 2000))
        logistic_l2 = float(exp.get("logistic_l2", 1.0))

        fixed_random_pairs: list[tuple[int, int]] = []
        for value in exp.get(
            "fixed_random_pairs", [[28, 29], [0, 1], [4, 17], [16, 20], [32, 48]]
        ):
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                raise ValueError("Each fixed_random_pairs item must contain two heads")
            pair = tuple(sorted((int(value[0]), int(value[1]))))
            if pair[0] == pair[1] or not 0 <= pair[0] < self.num_heads or not 0 <= pair[1] < self.num_heads:
                raise ValueError(f"Invalid fixed random head pair: {value}")
            if pair not in fixed_random_pairs:
                fixed_random_pairs.append(pair)

        calibration_path_text = os.path.expandvars(
            str(exp.get("calibration_path", ""))
        ).strip()
        calibration: dict[str, Any] | None = None
        if mode == "evaluation":
            if not calibration_path_text:
                raise ValueError("evaluation mode requires calibration_path")
            calibration_path = expand_path(calibration_path_text)
            if not calibration_path.is_file():
                raise FileNotFoundError(
                    f"Detector calibration artifact not found: {calibration_path}"
                )
            with calibration_path.open("r", encoding="utf-8") as stream:
                calibration = json.load(stream)
            expected = calibration.get("metadata", {})
            if int(expected.get("selected_layer", -1)) != selected_layer:
                raise ValueError("Calibration selected_layer does not match config")
            if [int(value) for value in expected.get("selected_heads", [])] != selected_head_indices:
                raise ValueError("Calibration selected_heads do not match config")
            if [int(value) for value in expected.get("capture_layers", [])] != capture_layers:
                raise ValueError("Calibration capture_layers do not match config")
            alpha = float(expected.get("alpha", configured_alpha))

        def coverage_key(value: float) -> str:
            return f"{float(value):.9f}"

        def threshold_at_coverage(values: Sequence[float], coverage: float) -> float:
            if not values:
                return float("nan")
            if coverage >= 1.0:
                return float("-inf")
            tensor = torch.tensor(list(values), dtype=torch.float64)
            return float(torch.quantile(tensor, 1.0 - float(coverage)).item())

        def mean_heads(values: Sequence[float], head_indices: Sequence[int]) -> float:
            return sum(float(values[index]) for index in head_indices) / len(head_indices)

        def pair_signal(payload: Mapping[str, Any], layer_idx: int, pair: tuple[int, int]) -> float:
            values = payload["head_signals"][str(layer_idx)]
            return 0.5 * (float(values[pair[0]]) + float(values[pair[1]]))

        signal_cache_path = writer.root / "detector_signals_cache.jsonl"
        cached_signals: dict[str, dict[str, Any]] = {}
        if signal_cache_path.exists():
            with signal_cache_path.open("r", encoding="utf-8") as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    cached_signals[str(record.get("sample_uid"))] = record

        payloads: list[dict[str, Any]] = []
        for row in tqdm(rows, desc=f"{exp['name']}:{mode}:signals"):
            sample_uid = str(row.get("sample_uid"))
            cache_record = cached_signals.get(sample_uid)
            cache_valid = bool(
                cache_record
                and cache_record.get("capture_layers") == capture_layers
                and cache_record.get("position") == detector_position
            )
            if cache_valid:
                full_score = dict(cache_record["full_score"])
                text_score = dict(cache_record["text_score"])
                head_signals = {
                    str(layer): [float(value) for value in values]
                    for layer, values in cache_record["head_signals"].items()
                }
            else:
                text_row = self.row_for_condition(row, "text_only") or self.synthesize_text_only(row)
                full_score, full_values = self.score_and_capture_head_values_many(
                    row, capture_layers, detector_position
                )
                text_score, text_values = self.score_and_capture_head_values_many(
                    text_row, capture_layers, detector_position
                )
                head_signals = {}
                for layer_idx in capture_layers:
                    full_heads = self._head_vectors(
                        full_values[layer_idx], self.num_heads
                    ).float()
                    text_heads = self._head_vectors(
                        text_values[layer_idx], self.num_heads
                    ).float()
                    numerator = torch.linalg.vector_norm(
                        full_heads - text_heads, dim=1
                    )
                    denominator = torch.linalg.vector_norm(
                        full_heads, dim=1
                    ).clamp_min(1e-6)
                    head_signals[str(layer_idx)] = [
                        float(value) for value in (numerator / denominator).tolist()
                    ]
                cache_record = {
                    "sample_uid": sample_uid,
                    "question_id": row.get("question_id"),
                    "capture_layers": capture_layers,
                    "position": detector_position,
                    "full_score": full_score,
                    "text_score": text_score,
                    "head_signals": head_signals,
                }
                with signal_cache_path.open("a", encoding="utf-8") as stream:
                    stream.write(
                        json.dumps(jsonable(cache_record), ensure_ascii=False) + "\n"
                    )

            options = parse_maybe_list(row.get("options_json"))
            full_logits = [float(value) for value in full_score["candidate_log_probs"]]
            text_logits = [float(value) for value in text_score["candidate_log_probs"]]
            contrast_logits = [
                (1.0 + alpha) * full - alpha * text
                for full, text in zip(full_logits, text_logits)
            ]
            contrast_score = score_candidate_values(row, options, contrast_logits)
            selected_signal = mean_heads(
                head_signals[str(selected_layer)], selected_head_indices
            )
            payloads.append(
                {
                    "row": dict(row),
                    "full": full_score,
                    "text": text_score,
                    "contrast": contrast_score,
                    "full_logits": full_logits,
                    "text_logits": text_logits,
                    "contrast_logits": contrast_logits,
                    "head_signals": head_signals,
                    "head": selected_signal,
                    "margin": -float(full_score["top1_margin"]),
                    "js": js_divergence(full_logits, text_logits),
                    "entropy": float(full_score["candidate_entropy"]),
                }
            )

        if not payloads:
            writer.finalize()
            return

        if mode == "calibration":
            alpha_selection_rows: list[dict[str, Any]] = []
            labels_available = all(
                isinstance(payload["full"].get("is_correct"), bool)
                for payload in payloads
            )
            for candidate_alpha in alpha_candidates:
                candidate_scores: list[dict[str, Any]] = []
                for payload in payloads:
                    options = parse_maybe_list(payload["row"].get("options_json"))
                    candidate_logits = [
                        (1.0 + candidate_alpha) * full - candidate_alpha * text
                        for full, text in zip(
                            payload["full_logits"], payload["text_logits"]
                        )
                    ]
                    candidate_scores.append(
                        score_candidate_values(
                            payload["row"], options, candidate_logits
                        )
                    )
                if labels_available:
                    baseline_correct = [
                        bool(payload["full"]["is_correct"])
                        for payload in payloads
                    ]
                    corrected = [
                        bool(score["is_correct"]) for score in candidate_scores
                    ]
                    repairs = sum(
                        (not before) and after
                        for before, after in zip(baseline_correct, corrected)
                    )
                    damage = sum(
                        before and (not after)
                        for before, after in zip(baseline_correct, corrected)
                    )
                    accuracy: float | str = sum(corrected) / len(corrected)
                else:
                    repairs = damage = 0
                    accuracy = ""
                margin_deltas = [
                    float(score["gold_margin"])
                    - float(payload["full"]["gold_margin"])
                    for payload, score in zip(payloads, candidate_scores)
                    if score.get("gold_margin") is not None
                    and payload["full"].get("gold_margin") is not None
                ]
                alpha_selection_rows.append(
                    {
                        "alpha": candidate_alpha,
                        "n": len(payloads),
                        "accuracy": accuracy,
                        "repairs": repairs,
                        "damage": damage,
                        "net_repairs": repairs - damage,
                        "mean_margin_delta": (
                            sum(margin_deltas) / len(margin_deltas)
                            if margin_deltas
                            else ""
                        ),
                        "scores": candidate_scores,
                    }
                )
            if labels_available:
                selected_alpha_row = max(
                    alpha_selection_rows,
                    key=lambda item: (
                        float(item["accuracy"]),
                        -int(item["damage"]),
                        -float(item["alpha"]),
                    ),
                )
                alpha = float(selected_alpha_row["alpha"])
            else:
                selected_alpha_row = next(
                    row
                    for row in alpha_selection_rows
                    if float(row["alpha"]) == configured_alpha
                )
                alpha = configured_alpha
            selected_scores = selected_alpha_row.pop("scores")
            for payload, selected_score in zip(payloads, selected_scores):
                payload["contrast_logits"] = [
                    (1.0 + alpha) * full - alpha * text
                    for full, text in zip(
                        payload["full_logits"], payload["text_logits"]
                    )
                ]
                payload["contrast"] = selected_score
            write_csv(
                writer.root / "alpha_selection.csv",
                [
                    {
                        **{key: value for key, value in row.items() if key != "scores"},
                        "selected": float(row["alpha"]) == alpha,
                    }
                    for row in alpha_selection_rows
                ],
            )

        def labels_for(name: str) -> list[bool] | None:
            if not payloads or any(
                not isinstance(payload["full"].get("is_correct"), bool)
                or not isinstance(payload["contrast"].get("is_correct"), bool)
                for payload in payloads
            ):
                return None
            if name == "error":
                return [not bool(payload["full"]["is_correct"]) for payload in payloads]
            if name == "repair":
                return [
                    (not bool(payload["full"]["is_correct"]))
                    and bool(payload["contrast"]["is_correct"])
                    for payload in payloads
                ]
            raise ValueError(f"Unknown detector label: {name}")

        error_labels = labels_for("error")
        repair_labels = labels_for("repair")

        def binary_auc(scores: Sequence[float], labels: Sequence[bool] | None) -> float | None:
            if labels is None:
                return None
            positives = [float(score) for score, label in zip(scores, labels) if label]
            negatives = [float(score) for score, label in zip(scores, labels) if not label]
            if not positives or not negatives:
                return None
            wins = 0.0
            for positive in positives:
                for negative in negatives:
                    if positive > negative:
                        wins += 1.0
                    elif positive == negative:
                        wins += 0.5
            return wins / (len(positives) * len(negatives))

        def average_precision(
            scores: Sequence[float], labels: Sequence[bool] | None
        ) -> float | None:
            if labels is None or not any(labels):
                return None
            ranked = sorted(
                zip((float(value) for value in scores), labels),
                key=lambda item: item[0],
                reverse=True,
            )
            hits = 0
            total = 0.0
            for rank, (_score, label) in enumerate(ranked, start=1):
                if label:
                    hits += 1
                    total += hits / rank
            return total / hits

        def pearson(values_a: Sequence[float], values_b: Sequence[float]) -> float | None:
            if len(values_a) != len(values_b) or len(values_a) < 2:
                return None
            a = torch.tensor(list(values_a), dtype=torch.float64)
            b = torch.tensor(list(values_b), dtype=torch.float64)
            a = a - a.mean()
            b = b - b.mean()
            denominator = torch.sqrt((a.square().sum()) * (b.square().sum()))
            if float(denominator.item()) <= 0.0:
                return None
            return float(((a * b).sum() / denominator).item())

        def fit_logistic(
            features: Sequence[Sequence[float]], labels: Sequence[bool] | None
        ) -> dict[str, Any] | None:
            if labels is None or not any(labels) or all(labels):
                return None
            x = torch.tensor(list(features), dtype=torch.float64)
            y = torch.tensor([float(value) for value in labels], dtype=torch.float64)
            means = x.mean(dim=0)
            scales = x.std(dim=0, unbiased=False).clamp_min(1e-6)
            z = (x - means) / scales
            weights = torch.zeros(z.shape[1], dtype=torch.float64, requires_grad=True)
            bias = torch.zeros((), dtype=torch.float64, requires_grad=True)
            optimizer = torch.optim.LBFGS(
                [weights, bias], lr=0.5, max_iter=200, line_search_fn="strong_wolfe"
            )

            def closure() -> torch.Tensor:
                optimizer.zero_grad()
                logits = z @ weights + bias
                loss = F.binary_cross_entropy_with_logits(logits, y)
                loss = loss + logistic_l2 * weights.square().sum() / len(labels)
                loss.backward()
                return loss

            optimizer.step(closure)
            return {
                "means": [float(value) for value in means.tolist()],
                "scales": [float(value) for value in scales.tolist()],
                "weights": [float(value) for value in weights.detach().tolist()],
                "bias": float(bias.detach().item()),
                "l2": logistic_l2,
            }

        def logistic_scores(
            model: Mapping[str, Any], features: Sequence[Sequence[float]]
        ) -> list[float]:
            means = [float(value) for value in model["means"]]
            scales = [float(value) for value in model["scales"]]
            weights = [float(value) for value in model["weights"]]
            bias = float(model["bias"])
            result: list[float] = []
            for row_features in features:
                logit = bias + sum(
                    weight * ((float(value) - mean) / scale)
                    for value, mean, scale, weight in zip(
                        row_features, means, scales, weights
                    )
                )
                if logit >= 0:
                    probability = 1.0 / (1.0 + math.exp(-logit))
                else:
                    exponential = math.exp(logit)
                    probability = exponential / (1.0 + exponential)
                result.append(probability)
            return result

        base_signals: dict[str, list[float]] = {
            "head": [float(payload["head"]) for payload in payloads],
            "margin": [float(payload["margin"]) for payload in payloads],
            "js": [float(payload["js"]) for payload in payloads],
            "entropy": [float(payload["entropy"]) for payload in payloads],
        }
        for layer_idx in capture_layers:
            base_signals[f"layer_{layer_idx}_selected_heads"] = [
                mean_heads(
                    payload["head_signals"][str(layer_idx)], selected_head_indices
                )
                for payload in payloads
            ]

        margin_values = base_signals["margin"]
        head_values = base_signals["head"]
        if mode == "calibration":
            margin_tensor = torch.tensor(margin_values, dtype=torch.float64)
            head_tensor = torch.tensor(head_values, dtype=torch.float64)
            margin_centered = margin_tensor - margin_tensor.mean()
            variance = float(margin_centered.square().sum().item())
            slope = (
                float(
                    (
                        margin_centered
                        * (head_tensor - head_tensor.mean())
                    ).sum().item()
                )
                / variance
                if variance > 0.0
                else 0.0
            )
            intercept = float(head_tensor.mean().item()) - slope * float(
                margin_tensor.mean().item()
            )
            residual_model = {"intercept": intercept, "slope": slope}
        else:
            assert calibration is not None
            residual_model = dict(calibration["residual_model"])
        base_signals["head_residual"] = [
            head
            - (
                float(residual_model["intercept"])
                + float(residual_model["slope"]) * margin
            )
            for head, margin in zip(head_values, margin_values)
        ]

        feature_sets: dict[str, list[list[float]]] = {
            "margin": [[payload["margin"]] for payload in payloads],
            "margin_head": [
                [payload["margin"], payload["head"]] for payload in payloads
            ],
            "generic": [
                [payload["margin"], payload["js"], payload["entropy"]]
                for payload in payloads
            ],
            "generic_head": [
                [
                    payload["margin"],
                    payload["js"],
                    payload["entropy"],
                    payload["head"],
                ]
                for payload in payloads
            ],
        }
        if mode == "calibration":
            logistic_models: dict[str, Any] = {}
            for target_name, labels in (
                ("error", error_labels),
                ("repair", repair_labels),
            ):
                for feature_name, features in feature_sets.items():
                    model = fit_logistic(features, labels)
                    if model is not None:
                        logistic_models[f"{target_name}_{feature_name}"] = {
                            **model,
                            "target": target_name,
                            "features": feature_name,
                        }
        else:
            assert calibration is not None
            logistic_models = dict(calibration.get("logistic_models", {}))
        for model_name, model in logistic_models.items():
            feature_name = str(model["features"])
            base_signals[model_name] = logistic_scores(
                model, feature_sets[feature_name]
            )

        all_pairs = list(combinations(range(self.num_heads), 2))
        selected_pair = (
            tuple(selected_head_indices)
            if len(selected_head_indices) == 2
            else None
        )
        if mode == "calibration":
            pairs_by_layer: dict[str, list[list[int]]] = {}
            for layer_idx in capture_layers:
                pairs = list(all_pairs)
                if pair_limit is not None and pair_limit < len(pairs):
                    rng = random.Random(self.seed + 1009 * layer_idx)
                    required = set(fixed_random_pairs)
                    if selected_pair is not None:
                        required.add(tuple(sorted(selected_pair)))
                    remaining = [pair for pair in pairs if pair not in required]
                    keep = max(0, pair_limit - len(required))
                    pairs = sorted(required) + rng.sample(remaining, keep)
                pairs_by_layer[str(layer_idx)] = [list(pair) for pair in pairs]
        else:
            assert calibration is not None
            pairs_by_layer = {
                str(layer): [list(pair) for pair in pairs]
                for layer, pairs in calibration["pairs_by_layer"].items()
            }

        if mode == "calibration":
            signal_thresholds = {
                name: {
                    coverage_key(coverage): threshold_at_coverage(values, coverage)
                    for coverage in coverages
                }
                for name, values in base_signals.items()
            }
            pair_thresholds: dict[str, dict[str, float]] = {}
            for layer_text, pairs in pairs_by_layer.items():
                layer_idx = int(layer_text)
                pair_thresholds[layer_text] = {}
                for pair_values in pairs:
                    pair = (int(pair_values[0]), int(pair_values[1]))
                    values = [pair_signal(payload, layer_idx, pair) for payload in payloads]
                    pair_thresholds[layer_text][f"{pair[0]}:{pair[1]}"] = (
                        threshold_at_coverage(values, primary_coverage)
                    )
            calibration = {
                "metadata": {
                    "version": 1,
                    "mode": "calibration",
                    "n": len(payloads),
                    "selected_layer": selected_layer,
                    "selected_heads": selected_head_indices,
                    "capture_layers": capture_layers,
                    "position": detector_position,
                    "alpha": alpha,
                    "primary_coverage": primary_coverage,
                    "coverages": coverages,
                    "seed": self.seed,
                },
                "residual_model": residual_model,
                "logistic_models": logistic_models,
                "signal_thresholds": signal_thresholds,
                "pairs_by_layer": pairs_by_layer,
                "pair_thresholds": pair_thresholds,
            }
            with (writer.root / "detector_calibration.json").open(
                "w", encoding="utf-8"
            ) as stream:
                json.dump(jsonable(calibration), stream, ensure_ascii=False, indent=2)
        else:
            assert calibration is not None
            signal_thresholds = dict(calibration["signal_thresholds"])
            pair_thresholds = dict(calibration["pair_thresholds"])
            with (writer.root / "calibration_used.json").open(
                "w", encoding="utf-8"
            ) as stream:
                json.dump(jsonable(calibration), stream, ensure_ascii=False, indent=2)

        def exact_mcnemar(repairs: int, damage: int) -> float:
            discordant = int(repairs) + int(damage)
            if discordant == 0:
                return 1.0
            smaller = min(int(repairs), int(damage))
            tail = sum(math.comb(discordant, index) for index in range(smaller + 1))
            return min(1.0, 2.0 * tail / (2**discordant))

        def policy_metrics(mask: Sequence[bool]) -> dict[str, Any]:
            applied = sum(bool(value) for value in mask)
            result: dict[str, Any] = {
                "n": len(payloads),
                "applied": applied,
                "coverage": applied / len(payloads) if payloads else 0.0,
            }
            if error_labels is None or repair_labels is None:
                return result
            final_correct = [
                bool(payload["contrast"]["is_correct"])
                if gate
                else bool(payload["full"]["is_correct"])
                for payload, gate in zip(payloads, mask)
            ]
            baseline_correct = [
                bool(payload["full"]["is_correct"]) for payload in payloads
            ]
            repairs = sum(
                (not before) and after
                for before, after in zip(baseline_correct, final_correct)
            )
            damage = sum(
                before and (not after)
                for before, after in zip(baseline_correct, final_correct)
            )
            errors_flagged = sum(
                gate and label for gate, label in zip(mask, error_labels)
            )
            repairable_flagged = sum(
                gate and label for gate, label in zip(mask, repair_labels)
            )
            error_total = sum(error_labels)
            repair_total = sum(repair_labels)
            result.update(
                {
                    "baseline_accuracy": sum(baseline_correct) / len(payloads),
                    "accuracy": sum(final_correct) / len(payloads),
                    "errors_flagged": errors_flagged,
                    "error_precision": errors_flagged / applied if applied else 0.0,
                    "error_recall": errors_flagged / error_total if error_total else None,
                    "repairable_flagged": repairable_flagged,
                    "repair_precision": repairable_flagged / applied if applied else 0.0,
                    "repair_recall": repairable_flagged / repair_total if repair_total else None,
                    "repairs": repairs,
                    "damage": damage,
                    "net_repairs": repairs - damage,
                    "utility_per_intervention": (
                        (repairs - damage) / applied if applied else 0.0
                    ),
                    "mcnemar_exact_p": exact_mcnemar(repairs, damage),
                }
            )
            return result

        def probabilistic_metrics(
            scores: Sequence[float], labels: Sequence[bool] | None
        ) -> tuple[float | None, float | None]:
            if labels is None:
                return None, None
            clipped = [min(1.0 - 1e-9, max(1e-9, float(value))) for value in scores]
            log_loss = -sum(
                math.log(probability if label else 1.0 - probability)
                for probability, label in zip(clipped, labels)
            ) / len(labels)
            brier = sum(
                (probability - float(label)) ** 2
                for probability, label in zip(clipped, labels)
            ) / len(labels)
            return log_loss, brier

        def rank_matched_mask(
            values: Sequence[float], coverage: float
        ) -> list[bool]:
            if not values:
                return []
            count = min(
                len(values), max(1, int(math.ceil(float(coverage) * len(values))))
            )
            selected = set(
                sorted(
                    range(len(values)),
                    key=lambda index: (-float(values[index]), index),
                )[:count]
            )
            return [index in selected for index in range(len(values))]

        detector_metric_rows: list[dict[str, Any]] = []
        for name, values in base_signals.items():
            error_log_loss = error_brier = repair_log_loss = repair_brier = None
            if name in logistic_models:
                target = str(logistic_models[name]["target"])
                if target == "error":
                    error_log_loss, error_brier = probabilistic_metrics(
                        values, error_labels
                    )
                elif target == "repair":
                    repair_log_loss, repair_brier = probabilistic_metrics(
                        values, repair_labels
                    )
            detector_metric_rows.append(
                {
                    "mode": mode,
                    "signal": name,
                    "n": len(values),
                    "n_errors": sum(error_labels) if error_labels is not None else "",
                    "n_repairable": (
                        sum(repair_labels) if repair_labels is not None else ""
                    ),
                    "error_auroc": binary_auc(values, error_labels),
                    "error_ap": average_precision(values, error_labels),
                    "repair_auroc": binary_auc(values, repair_labels),
                    "repair_ap": average_precision(values, repair_labels),
                    "error_log_loss": error_log_loss,
                    "error_brier": error_brier,
                    "repair_log_loss": repair_log_loss,
                    "repair_brier": repair_brier,
                    "pearson_with_margin_risk": pearson(values, margin_values),
                }
            )
        write_csv(writer.root / "detector_metrics.csv", detector_metric_rows)

        def get_signal_threshold(name: str, coverage: float) -> float:
            if name not in signal_thresholds:
                raise KeyError(f"No calibrated thresholds for signal {name!r}")
            key = coverage_key(coverage)
            if key not in signal_thresholds[name]:
                raise KeyError(f"No calibrated threshold for {name} at coverage {coverage}")
            return float(signal_thresholds[name][key])

        policy_definitions: list[dict[str, Any]] = []
        core_signal_names = [
            name
            for name in (
                "head",
                "margin",
                "js",
                "entropy",
                "head_residual",
                "error_margin",
                "error_margin_head",
                "error_generic",
                "error_generic_head",
                "repair_margin",
                "repair_margin_head",
                "repair_generic",
                "repair_generic_head",
            )
            if name in base_signals
        ]
        for coverage in coverages:
            for signal_name in core_signal_names:
                threshold = get_signal_threshold(signal_name, coverage)
                policy_definitions.append(
                    {
                        "name": f"{signal_name}_cov_{coverage_key(coverage)}",
                        "gate_type": signal_name,
                        "target_coverage": coverage,
                        "threshold": threshold,
                        "mask": [
                            value >= threshold for value in base_signals[signal_name]
                        ],
                    }
                )
                policy_definitions.append(
                    {
                        "name": f"{signal_name}_rankmatched_cov_{coverage_key(coverage)}",
                        "gate_type": f"{signal_name}_rankmatched",
                        "target_coverage": coverage,
                        "threshold": "rank_matched",
                        "mask": rank_matched_mask(
                            base_signals[signal_name], coverage
                        ),
                    }
                )

        primary_key = coverage_key(primary_coverage)
        head_threshold = get_signal_threshold("head", primary_coverage)
        margin_threshold = get_signal_threshold("margin", primary_coverage)
        head_mask = [value >= head_threshold for value in base_signals["head"]]
        margin_mask = [
            value >= margin_threshold for value in base_signals["margin"]
        ]
        policy_definitions.extend(
            [
                {
                    "name": "head_and_margin_primary",
                    "gate_type": "head_and_margin",
                    "target_coverage": primary_coverage,
                    "threshold": "",
                    "mask": [a and b for a, b in zip(head_mask, margin_mask)],
                },
                {
                    "name": "head_or_margin_primary",
                    "gate_type": "head_or_margin",
                    "target_coverage": primary_coverage,
                    "threshold": "",
                    "mask": [a or b for a, b in zip(head_mask, margin_mask)],
                },
            ]
        )
        for layer_idx in capture_layers:
            signal_name = f"layer_{layer_idx}_selected_heads"
            threshold = get_signal_threshold(signal_name, primary_coverage)
            policy_definitions.append(
                {
                    "name": f"{signal_name}_primary",
                    "gate_type": "same_heads_layer_control",
                    "target_coverage": primary_coverage,
                    "threshold": threshold,
                    "layer": layer_idx,
                    "mask": [
                        value >= threshold for value in base_signals[signal_name]
                    ],
                }
            )
        for pair in fixed_random_pairs:
            pair_key = f"{pair[0]}:{pair[1]}"
            if pair_key not in pair_thresholds[str(selected_layer)]:
                continue
            threshold = float(pair_thresholds[str(selected_layer)][pair_key])
            values = [pair_signal(payload, selected_layer, pair) for payload in payloads]
            policy_definitions.append(
                {
                    "name": f"random_pair_{pair[0]}_{pair[1]}_primary",
                    "gate_type": "random_pair",
                    "target_coverage": primary_coverage,
                    "threshold": threshold,
                    "heads": list(pair),
                    "layer": selected_layer,
                    "mask": [value >= threshold for value in values],
                }
            )

        def append_policy_result(
            payload: Mapping[str, Any], policy: Mapping[str, Any], applied: bool
        ) -> None:
            row = payload["row"]
            score = payload["contrast"] if applied else payload["full"]
            config = {
                "method": "detector_disambiguation",
                "mode": mode,
                "policy": policy["name"],
                "alpha": alpha,
                "threshold": policy.get("threshold"),
                "target_coverage": policy.get("target_coverage"),
                "layer": policy.get("layer"),
                "heads": policy.get("heads"),
            }
            config_id = stable_hash(config)
            writer.append(
                {
                    **score,
                    "result_id": f"{row['sample_uid']}::{config_id}",
                    "experiment": exp["name"],
                    "experiment_type": "detector_disambiguation",
                    "config_id": config_id,
                    "question_id": row["question_id"],
                    "question_form": "MC",
                    "condition": row["condition"],
                    "control": policy["name"],
                    "mode": mode,
                    "scoring_method": "label_token",
                    "gate_type": policy["gate_type"],
                    "gate_applied": bool(applied),
                    "gate_threshold": policy.get("threshold", ""),
                    "target_coverage": policy.get("target_coverage", ""),
                    "layer": policy.get("layer", selected_layer),
                    "heads": policy.get("heads", selected_head_indices),
                    "alpha": alpha,
                    "head_visual_signal": payload["head"],
                    "baseline_top1_margin": payload["full"]["top1_margin"],
                    "full_text_label_js": payload["js"],
                    "baseline_entropy": payload["entropy"],
                    "baseline_prediction": payload["full"]["prediction"],
                    "baseline_correct": payload["full"]["is_correct"],
                    "baseline_gold_margin": payload["full"]["gold_margin"],
                    "always_contrast_prediction": payload["contrast"]["prediction"],
                    "always_contrast_correct": payload["contrast"]["is_correct"],
                    "margin_delta": (
                        float(score["gold_margin"])
                        - float(payload["full"]["gold_margin"])
                        if score.get("gold_margin") is not None
                        and payload["full"].get("gold_margin") is not None
                        else None
                    ),
                }
            )

        baseline_policy = {
            "name": "baseline",
            "gate_type": "none",
            "target_coverage": 0.0,
            "threshold": "",
        }
        always_policy = {
            "name": "always_contrast",
            "gate_type": "always",
            "target_coverage": 1.0,
            "threshold": float("-inf"),
        }
        for payload in payloads:
            append_policy_result(payload, baseline_policy, False)
            append_policy_result(payload, always_policy, True)
        risk_coverage_rows: list[dict[str, Any]] = [
            {
                "mode": mode,
                "policy": "baseline",
                "gate_type": "none",
                "target_coverage": 0.0,
                "threshold": "",
                **policy_metrics([False] * len(payloads)),
            },
            {
                "mode": mode,
                "policy": "always_contrast",
                "gate_type": "always",
                "target_coverage": 1.0,
                "threshold": float("-inf"),
                **policy_metrics([True] * len(payloads)),
            },
        ]
        for policy in policy_definitions:
            mask = [bool(value) for value in policy["mask"]]
            for payload, applied in zip(payloads, mask):
                append_policy_result(payload, policy, applied)
            risk_coverage_rows.append(
                {
                    "mode": mode,
                    "policy": policy["name"],
                    "gate_type": policy["gate_type"],
                    "target_coverage": policy.get("target_coverage", ""),
                    "threshold": policy.get("threshold", ""),
                    "layer": policy.get("layer", ""),
                    "heads": policy.get("heads", ""),
                    **policy_metrics(mask),
                }
            )
        write_csv(writer.root / "risk_coverage.csv", risk_coverage_rows)

        pair_null_rows: list[dict[str, Any]] = []
        conditional_null_rows: list[dict[str, Any]] = []
        low_indices = [index for index, value in enumerate(margin_mask) if value]
        selected_low_count = sum(
            head_mask[index] and margin_mask[index] for index in range(len(payloads))
        )
        for layer_text, pairs in pairs_by_layer.items():
            layer_idx = int(layer_text)
            for pair_values in pairs:
                pair = (int(pair_values[0]), int(pair_values[1]))
                pair_key = f"{pair[0]}:{pair[1]}"
                values = [pair_signal(payload, layer_idx, pair) for payload in payloads]
                threshold = float(pair_thresholds[layer_text][pair_key])
                mask = [value >= threshold for value in values]
                metrics = policy_metrics(mask)
                rank_metrics = policy_metrics(
                    rank_matched_mask(values, primary_coverage)
                )
                is_selected = bool(
                    layer_idx == selected_layer
                    and selected_pair is not None
                    and tuple(pair) == tuple(sorted(selected_pair))
                )
                pair_null_rows.append(
                    {
                        "mode": mode,
                        "layer": layer_idx,
                        "head_a": pair[0],
                        "head_b": pair[1],
                        "is_selected_pair": is_selected,
                        "threshold": threshold,
                        "target_coverage": primary_coverage,
                        "error_auroc": binary_auc(values, error_labels),
                        "error_ap": average_precision(values, error_labels),
                        "repair_auroc": binary_auc(values, repair_labels),
                        "repair_ap": average_precision(values, repair_labels),
                        "pearson_with_margin_risk": pearson(values, margin_values),
                        **metrics,
                        **{
                            f"rank_{key}": value
                            for key, value in rank_metrics.items()
                        },
                    }
                )
                if layer_idx == selected_layer and selected_low_count > 0:
                    ranked_low = sorted(
                        low_indices, key=lambda index: values[index], reverse=True
                    )[:selected_low_count]
                    conditional_null_rows.append(
                        {
                            "mode": mode,
                            "layer": layer_idx,
                            "head_a": pair[0],
                            "head_b": pair[1],
                            "is_selected_pair": is_selected,
                            "low_margin_pool_n": len(low_indices),
                            "conditional_k": selected_low_count,
                            "errors_captured": (
                                sum(error_labels[index] for index in ranked_low)
                                if error_labels is not None
                                else ""
                            ),
                            "repairable_captured": (
                                sum(repair_labels[index] for index in ranked_low)
                                if repair_labels is not None
                                else ""
                            ),
                        }
                    )
        write_csv(writer.root / "head_pair_null.csv", pair_null_rows)
        write_csv(
            writer.root / "conditional_random_pair_null.csv", conditional_null_rows
        )

        strata_rows: list[dict[str, Any]] = []
        for head_flag in (True, False):
            for margin_flag in (True, False):
                indices = [
                    index
                    for index, (head_value, margin_value) in enumerate(
                        zip(head_mask, margin_mask)
                    )
                    if head_value == head_flag and margin_value == margin_flag
                ]
                strata_rows.append(
                    {
                        "mode": mode,
                        "head_gate": head_flag,
                        "margin_gate": margin_flag,
                        "n": len(indices),
                        "errors": (
                            sum(error_labels[index] for index in indices)
                            if error_labels is not None
                            else ""
                        ),
                        "repairable": (
                            sum(repair_labels[index] for index in indices)
                            if repair_labels is not None
                            else ""
                        ),
                        "error_rate": (
                            sum(error_labels[index] for index in indices) / len(indices)
                            if error_labels is not None and indices
                            else ""
                        ),
                        "repairable_rate": (
                            sum(repair_labels[index] for index in indices) / len(indices)
                            if repair_labels is not None and indices
                            else ""
                        ),
                    }
                )
        write_csv(writer.root / "head_margin_strata.csv", strata_rows)

        signal_rows: list[dict[str, Any]] = []
        for index, payload in enumerate(payloads):
            signal_rows.append(
                {
                    "mode": mode,
                    "sample_uid": payload["row"]["sample_uid"],
                    "question_id": payload["row"]["question_id"],
                    "baseline_prediction": payload["full"]["prediction"],
                    "baseline_correct": payload["full"]["is_correct"],
                    "contrast_prediction": payload["contrast"]["prediction"],
                    "contrast_correct": payload["contrast"]["is_correct"],
                    "repairable": repair_labels[index] if repair_labels is not None else "",
                    "head_signal": payload["head"],
                    "margin_risk": payload["margin"],
                    "js": payload["js"],
                    "entropy": payload["entropy"],
                    "head_residual": base_signals["head_residual"][index],
                    "head_signals_by_layer": payload["head_signals"],
                    "full_candidate_log_probs": payload["full_logits"],
                    "text_candidate_log_probs": payload["text_logits"],
                    **{
                        model_name: base_signals[model_name][index]
                        for model_name in logistic_models
                    },
                }
            )
        write_csv(writer.root / "detector_signals.csv", signal_rows)

        def percentile_against_null(
            selected_value: Any, null_values: Sequence[Any]
        ) -> float | None:
            if selected_value in (None, ""):
                return None
            numeric = [float(value) for value in null_values if value not in (None, "")]
            if not numeric:
                return None
            return sum(value <= float(selected_value) for value in numeric) / len(numeric)

        selected_rows = [row for row in pair_null_rows if row["is_selected_pair"]]
        same_layer_null = [
            row
            for row in pair_null_rows
            if row["layer"] == selected_layer and not row["is_selected_pair"]
        ]
        all_layer_null = [
            row for row in pair_null_rows if not row["is_selected_pair"]
        ]
        selected_row = selected_rows[0] if selected_rows else None
        conditional_selected = next(
            (row for row in conditional_null_rows if row["is_selected_pair"]), None
        )
        conditional_null = [
            row for row in conditional_null_rows if not row["is_selected_pair"]
        ]
        metric_lookup = {row["signal"]: row for row in detector_metric_rows}

        def delta_metric(new_name: str, old_name: str, metric: str) -> float | None:
            new_row = metric_lookup.get(new_name)
            old_row = metric_lookup.get(old_name)
            if not new_row or not old_row:
                return None
            new_value = new_row.get(metric)
            old_value = old_row.get(metric)
            if new_value in (None, "") or old_value in (None, ""):
                return None
            return float(new_value) - float(old_value)

        def bootstrap_delta(
            new_name: str,
            old_name: str,
            labels: Sequence[bool] | None,
            metric_name: str,
        ) -> dict[str, Any] | None:
            if (
                labels is None
                or new_name not in base_signals
                or old_name not in base_signals
                or bootstrap_iterations <= 0
            ):
                return None
            metric_function = (
                binary_auc if metric_name == "auroc" else average_precision
            )
            observed_new = metric_function(base_signals[new_name], labels)
            observed_old = metric_function(base_signals[old_name], labels)
            if observed_new is None or observed_old is None:
                return None
            rng = random.Random(
                self.seed
                + int(
                    stable_hash(
                        [new_name, old_name, metric_name, mode], length=8
                    ),
                    16,
                )
            )
            deltas: list[float] = []
            for _iteration in range(bootstrap_iterations):
                indices = [rng.randrange(len(labels)) for _ in range(len(labels))]
                sampled_labels = [bool(labels[index]) for index in indices]
                if not any(sampled_labels) or all(sampled_labels):
                    continue
                sampled_new = [base_signals[new_name][index] for index in indices]
                sampled_old = [base_signals[old_name][index] for index in indices]
                new_value = metric_function(sampled_new, sampled_labels)
                old_value = metric_function(sampled_old, sampled_labels)
                if new_value is not None and old_value is not None:
                    deltas.append(float(new_value) - float(old_value))
            if not deltas:
                return None
            deltas.sort()

            def empirical_quantile(q: float) -> float:
                index = min(
                    len(deltas) - 1,
                    max(0, int(round(q * (len(deltas) - 1)))),
                )
                return deltas[index]

            less_equal_zero = sum(value <= 0.0 for value in deltas) / len(deltas)
            greater_equal_zero = sum(value >= 0.0 for value in deltas) / len(deltas)
            return {
                "new": new_name,
                "old": old_name,
                "metric": metric_name,
                "observed_delta": float(observed_new) - float(observed_old),
                "ci95_low": empirical_quantile(0.025),
                "ci95_high": empirical_quantile(0.975),
                "two_sided_bootstrap_p": min(
                    1.0, 2.0 * min(less_equal_zero, greater_equal_zero)
                ),
                "iterations_requested": bootstrap_iterations,
                "iterations_used": len(deltas),
            }

        bootstrap_nested: list[dict[str, Any]] = []
        for target_name, labels in (
            ("error", error_labels),
            ("repair", repair_labels),
        ):
            for new_suffix, old_suffix in (
                ("margin_head", "margin"),
                ("generic_head", "generic"),
            ):
                for metric_name in ("auroc", "ap"):
                    result = bootstrap_delta(
                        f"{target_name}_{new_suffix}",
                        f"{target_name}_{old_suffix}",
                        labels,
                        metric_name,
                    )
                    if result is not None:
                        bootstrap_nested.append(result)

        report = {
            "mode": mode,
            "n": len(payloads),
            "selected_layer": selected_layer,
            "selected_heads": selected_head_indices,
            "primary_coverage": primary_coverage,
            "selected_pair": selected_row,
            "selected_pair_percentile_within_layer": (
                {
                    metric: percentile_against_null(
                        selected_row.get(metric),
                        [row.get(metric) for row in same_layer_null],
                    )
                    for metric in (
                        "error_auroc",
                        "error_ap",
                        "repair_auroc",
                        "repair_ap",
                        "net_repairs",
                        "utility_per_intervention",
                    )
                }
                if selected_row is not None
                else {}
            ),
            "selected_pair_percentile_all_captured_layers": (
                {
                    metric: percentile_against_null(
                        selected_row.get(metric),
                        [row.get(metric) for row in all_layer_null],
                    )
                    for metric in (
                        "error_auroc",
                        "error_ap",
                        "repair_auroc",
                        "repair_ap",
                        "net_repairs",
                        "utility_per_intervention",
                    )
                }
                if selected_row is not None
                else {}
            ),
            "conditional_selected_pair": conditional_selected,
            "conditional_selected_percentile": (
                {
                    metric: percentile_against_null(
                        conditional_selected.get(metric),
                        [row.get(metric) for row in conditional_null],
                    )
                    for metric in ("errors_captured", "repairable_captured")
                }
                if conditional_selected is not None
                else {}
            ),
            "nested_model_deltas": {
                "error_margin_plus_head_delta_auroc": delta_metric(
                    "error_margin_head", "error_margin", "error_auroc"
                ),
                "error_margin_plus_head_delta_ap": delta_metric(
                    "error_margin_head", "error_margin", "error_ap"
                ),
                "error_generic_plus_head_delta_auroc": delta_metric(
                    "error_generic_head", "error_generic", "error_auroc"
                ),
                "error_generic_plus_head_delta_ap": delta_metric(
                    "error_generic_head", "error_generic", "error_ap"
                ),
                "repair_margin_plus_head_delta_auroc": delta_metric(
                    "repair_margin_head", "repair_margin", "repair_auroc"
                ),
                "repair_margin_plus_head_delta_ap": delta_metric(
                    "repair_margin_head", "repair_margin", "repair_ap"
                ),
                "repair_generic_plus_head_delta_auroc": delta_metric(
                    "repair_generic_head", "repair_generic", "repair_auroc"
                ),
                "repair_generic_plus_head_delta_ap": delta_metric(
                    "repair_generic_head", "repair_generic", "repair_ap"
                ),
            },
            "bootstrap_nested_deltas": bootstrap_nested,
        }
        with (writer.root / "detector_report.json").open(
            "w", encoding="utf-8"
        ) as stream:
            json.dump(jsonable(report), stream, ensure_ascii=False, indent=2)
        writer.finalize()

    def run_semantic_gated_contrast(
        self, exp: Mapping[str, Any], profile: str
    ) -> None:
        """Calibrate then evaluate content ranking with selective visual contrast.

        Hyper-parameters and gate thresholds are selected only on a shuffled,
        disjoint calibration subset.  Evaluation records are emitted for the
        original label-token policy, semantic content baseline, best always-on
        contrast, best generic gate, and best L63 head-signal gate.
        """
        writer = ResultWriter(self.output_dir, str(exp["name"]))

        def profile_value(name: str, default: Any) -> Any:
            return exp.get(f"{profile}_{name}", exp.get(name, default))

        rows = self.rows_filtered(["MC"], exp.get("conditions", ["full"]), None)
        split_seed = int(exp.get("split_seed", self.seed))
        random.Random(split_seed).shuffle(rows)
        total_limit = profile_value("limit", None)
        if total_limit is not None:
            rows = rows[: int(total_limit)]
        calibration_limit = int(profile_value("calibration_limit", 24))
        if calibration_limit <= 0 or calibration_limit >= len(rows):
            raise ValueError(
                "semantic_gated_contrast requires 0 < calibration_limit < row count; "
                f"received calibration_limit={calibration_limit}, rows={len(rows)}"
            )
        calibration_rows = rows[:calibration_limit]
        evaluation_rows = rows[calibration_limit:]
        evaluation_limit = profile_value("evaluation_limit", None)
        if evaluation_limit is not None:
            evaluation_rows = evaluation_rows[: int(evaluation_limit)]

        prompt_key = str(exp.get("content_prompt_key", "MC_CONTENT_1"))
        normalization = str(exp.get("content_normalization", "mean"))
        candidate_batch_size = int(profile_value("candidate_batch_size", 1))
        continuation_prefix = str(exp.get("continuation_prefix", " "))
        priors = [str(value) for value in exp.get("priors", ["text_only", "blurred"])]
        alphas = [float(value) for value in profile_value("alphas", [0.25, 0.5, 1.0])]
        betas = [
            float(value)
            for value in profile_value("plausibility_betas", [0.0, 0.05])
        ]
        gate_types = [
            str(value)
            for value in profile_value(
                "gate_types",
                [
                    "always",
                    "disagree",
                    "low_margin",
                    "high_js",
                    "disagree_or_low_margin",
                    "high_head",
                    "high_head_and_low_margin",
                ],
            )
        ]
        margin_quantiles = [
            float(value) for value in profile_value("margin_quantiles", [0.2, 0.4])
        ]
        js_quantiles = [
            float(value) for value in profile_value("js_quantiles", [0.6, 0.8])
        ]
        head_quantiles = [
            float(value) for value in profile_value("head_quantiles", [0.6, 0.8])
        ]
        detector_heads = parse_heads(exp.get("detector_heads", []))
        detector_position = str(exp.get("detector_position", "last"))
        blur_radius = float(exp.get("blur_radius", 12.0))
        noise_sigma = float(exp.get("noise_sigma", 0.2))

        unsupported_priors = set(priors) - {"text_only", "blurred", "gaussian_noise"}
        if unsupported_priors:
            raise ValueError(f"Unsupported semantic contrast priors: {unsupported_priors}")

        def negative_row(row: Mapping[str, Any], prior: str) -> dict[str, Any]:
            if prior == "text_only":
                return self.row_for_condition(row, "text_only") or self.synthesize_text_only(row)
            if prior == "blurred":
                return self.synthesize_blurred(row, blur_radius)
            return self.synthesize_noisy(row, noise_sigma)

        def head_visual_signal(
            row: Mapping[str, Any], text_row: Mapping[str, Any]
        ) -> float | None:
            if not detector_heads:
                return None
            values: list[float] = []
            grouped: dict[int, list[int]] = {}
            for layer_idx, head_idx in detector_heads:
                grouped.setdefault(layer_idx, []).append(head_idx)
            for layer_idx, head_indices in grouped.items():
                full_heads = self._head_vectors(
                    self.capture_values(row, "head", layer_idx, detector_position),
                    self.num_heads,
                ).float()
                text_heads = self._head_vectors(
                    self.capture_values(text_row, "head", layer_idx, detector_position),
                    self.num_heads,
                ).float()
                for head_idx in head_indices:
                    if not 0 <= head_idx < self.num_heads:
                        raise ValueError(f"Detector head out of range: {layer_idx}:{head_idx}")
                    difference = torch.linalg.vector_norm(
                        full_heads[head_idx] - text_heads[head_idx]
                    )
                    denominator = torch.linalg.vector_norm(full_heads[head_idx]).clamp_min(1e-6)
                    values.append(float((difference / denominator).item()))
            return sum(values) / len(values) if values else None

        def build_payload(row: Mapping[str, Any]) -> dict[str, Any]:
            full_score = self.score_mc_content(
                row,
                prompt_key=prompt_key,
                normalization=normalization,
                candidate_batch_size=candidate_batch_size,
                continuation_prefix=continuation_prefix,
            )
            text_row = negative_row(row, "text_only")
            negative_scores: dict[str, dict[str, Any]] = {}
            negative_rows: dict[str, dict[str, Any]] = {}
            for prior in priors:
                candidate_row = text_row if prior == "text_only" else negative_row(row, prior)
                negative_rows[prior] = candidate_row
                negative_scores[prior] = self.score_mc_content(
                    candidate_row,
                    prompt_key=prompt_key,
                    normalization=normalization,
                    candidate_batch_size=candidate_batch_size,
                    continuation_prefix=continuation_prefix,
                )
            full_values = [float(value) for value in full_score["candidate_log_probs"]]
            js_by_prior = {
                prior: js_divergence(
                    full_values,
                    [float(value) for value in score["candidate_log_probs"]],
                )
                for prior, score in negative_scores.items()
            }
            return {
                "row": dict(row),
                "full": full_score,
                "negatives": negative_scores,
                "negative_rows": negative_rows,
                "js": js_by_prior,
                "head_signal": head_visual_signal(row, text_row),
            }

        calibration_payloads = [
            build_payload(row)
            for row in tqdm(calibration_rows, desc=f"{exp['name']}:calibration_scores")
        ]

        def quantile(values: Sequence[float], q: float) -> float:
            if not values:
                return float("nan")
            return float(torch.quantile(torch.tensor(values, dtype=torch.float32), q).item())

        margin_values = [float(item["full"]["top1_margin"]) for item in calibration_payloads]
        margin_thresholds = {q: quantile(margin_values, q) for q in margin_quantiles}
        js_thresholds = {
            prior: {
                q: quantile([float(item["js"][prior]) for item in calibration_payloads], q)
                for q in js_quantiles
            }
            for prior in priors
        }
        head_values = [
            float(item["head_signal"])
            for item in calibration_payloads
            if item.get("head_signal") is not None
        ]
        head_thresholds = {q: quantile(head_values, q) for q in head_quantiles}

        configs: list[dict[str, Any]] = []
        for prior in priors:
            for alpha in alphas:
                for beta in betas:
                    base = {
                        "prior": prior,
                        "alpha": alpha,
                        "plausibility_beta": beta,
                        "content_prompt_key": prompt_key,
                        "content_normalization": normalization,
                        "continuation_prefix": continuation_prefix,
                        "detector_heads": detector_heads,
                        "blur_radius": blur_radius if prior == "blurred" else None,
                    }
                    for gate_type in gate_types:
                        if gate_type in {"always", "disagree"}:
                            configs.append({**base, "gate_type": gate_type})
                        elif gate_type == "low_margin":
                            for q, threshold in margin_thresholds.items():
                                configs.append(
                                    {
                                        **base,
                                        "gate_type": gate_type,
                                        "margin_quantile": q,
                                        "margin_threshold": threshold,
                                    }
                                )
                        elif gate_type == "high_js":
                            for q, threshold in js_thresholds[prior].items():
                                configs.append(
                                    {
                                        **base,
                                        "gate_type": gate_type,
                                        "js_quantile": q,
                                        "js_threshold": threshold,
                                    }
                                )
                        elif gate_type == "disagree_or_low_margin":
                            for q, threshold in margin_thresholds.items():
                                configs.append(
                                    {
                                        **base,
                                        "gate_type": gate_type,
                                        "margin_quantile": q,
                                        "margin_threshold": threshold,
                                    }
                                )
                        elif gate_type == "high_head":
                            for q, threshold in head_thresholds.items():
                                configs.append(
                                    {
                                        **base,
                                        "gate_type": gate_type,
                                        "head_quantile": q,
                                        "head_threshold": threshold,
                                    }
                                )
                        elif gate_type == "high_head_and_low_margin":
                            for head_q, head_threshold in head_thresholds.items():
                                for margin_q, margin_threshold in margin_thresholds.items():
                                    configs.append(
                                        {
                                            **base,
                                            "gate_type": gate_type,
                                            "head_quantile": head_q,
                                            "head_threshold": head_threshold,
                                            "margin_quantile": margin_q,
                                            "margin_threshold": margin_threshold,
                                        }
                                    )
                        else:
                            raise ValueError(f"Unsupported gate_type: {gate_type}")

        def gate_applies(payload: Mapping[str, Any], config: Mapping[str, Any]) -> bool:
            gate_type = str(config["gate_type"])
            if gate_type == "always":
                return True
            prior = str(config["prior"])
            full = payload["full"]
            negative = payload["negatives"][prior]
            disagree = int(full["predicted_index"]) != int(negative["predicted_index"])
            low_margin = float(full["top1_margin"]) <= float(
                config.get("margin_threshold", float("-inf"))
            )
            high_js = float(payload["js"][prior]) >= float(
                config.get("js_threshold", float("inf"))
            )
            head_signal = payload.get("head_signal")
            high_head = head_signal is not None and float(head_signal) >= float(
                config.get("head_threshold", float("inf"))
            )
            if gate_type == "disagree":
                return disagree
            if gate_type == "low_margin":
                return low_margin
            if gate_type == "high_js":
                return high_js
            if gate_type == "disagree_or_low_margin":
                return disagree or low_margin
            if gate_type == "high_head":
                return high_head
            if gate_type == "high_head_and_low_margin":
                return high_head and low_margin
            raise ValueError(f"Unsupported gate_type: {gate_type}")

        def apply_config(
            payload: Mapping[str, Any], config: Mapping[str, Any]
        ) -> tuple[dict[str, Any], bool]:
            row = payload["row"]
            options = parse_maybe_list(row.get("options_json"))
            full_values = [float(value) for value in payload["full"]["candidate_log_probs"]]
            applied = gate_applies(payload, config)
            combined = list(full_values)
            if applied:
                prior_values = [
                    float(value)
                    for value in payload["negatives"][str(config["prior"])][
                        "candidate_log_probs"
                    ]
                ]
                alpha = float(config["alpha"])
                combined = [
                    (1.0 + alpha) * full - alpha * prior
                    for full, prior in zip(full_values, prior_values)
                ]
                beta = float(config.get("plausibility_beta", 0.0))
                if beta > 0:
                    max_full = max(full_values)
                    threshold = max_full + math.log(beta)
                    combined = [
                        value if full >= threshold else float("-inf")
                        for value, full in zip(combined, full_values)
                    ]
            return score_candidate_values(row, options, combined), applied

        calibration_metrics: list[dict[str, Any]] = []
        for config in configs:
            correct = repairs = damage = applied_count = 0
            margins: list[float] = []
            for payload in calibration_payloads:
                score, applied = apply_config(payload, config)
                correct += int(score.get("is_correct") is True)
                repairs += int(
                    score.get("is_correct") is True
                    and payload["full"].get("is_correct") is False
                )
                damage += int(
                    score.get("is_correct") is False
                    and payload["full"].get("is_correct") is True
                )
                applied_count += int(applied)
                if score.get("gold_margin") is not None:
                    margins.append(float(score["gold_margin"]))
            calibration_metrics.append(
                {
                    **config,
                    "config_id": stable_hash(config),
                    "n": len(calibration_payloads),
                    "correct": correct,
                    "accuracy": correct / len(calibration_payloads),
                    "repairs": repairs,
                    "damage": damage,
                    "net_repairs": repairs - damage,
                    "intervention_rate": applied_count / len(calibration_payloads),
                    "mean_gold_margin": sum(margins) / len(margins) if margins else None,
                }
            )

        def metric_key(item: Mapping[str, Any]) -> tuple[float, float, float, float]:
            return (
                float(item["correct"]),
                -float(item["damage"]),
                float(item["net_repairs"]),
                float(item.get("mean_gold_margin") or float("-inf")),
            )

        def best_where(predicate: Callable[[Mapping[str, Any]], bool]) -> dict[str, Any] | None:
            eligible = [item for item in calibration_metrics if predicate(item)]
            return max(eligible, key=metric_key) if eligible else None

        selected_named: list[tuple[str, dict[str, Any]]] = []
        selections = {
            "always": best_where(lambda item: item["gate_type"] == "always"),
            "gated": best_where(
                lambda item: item["gate_type"] not in {"always", "high_head", "high_head_and_low_margin"}
            ),
            "head_gated": best_where(
                lambda item: item["gate_type"] in {"high_head", "high_head_and_low_margin"}
            ),
        }
        for name, selected in selections.items():
            if selected is not None:
                selected_named.append((name, selected))

        calibration_baseline_correct = sum(
            int(payload["full"].get("is_correct") is True)
            for payload in calibration_payloads
        )
        selection_payload = {
            "split_seed": split_seed,
            "calibration_question_ids": [row["question_id"] for row in calibration_rows],
            "evaluation_question_ids": [row["question_id"] for row in evaluation_rows],
            "content_baseline": {
                "correct": calibration_baseline_correct,
                "n": len(calibration_payloads),
                "accuracy": calibration_baseline_correct / len(calibration_payloads),
            },
            "thresholds": {
                "margin": margin_thresholds,
                "js": js_thresholds,
                "head": head_thresholds,
            },
            "selected": selections,
            "all_calibration_metrics": sorted(
                calibration_metrics, key=metric_key, reverse=True
            ),
        }
        with (writer.root / "calibration_selection.json").open(
            "w", encoding="utf-8"
        ) as stream:
            json.dump(jsonable(selection_payload), stream, ensure_ascii=False, indent=2)

        label_config_id = stable_hash({"scoring": "label_token"})
        content_config_id = stable_hash(
            {
                "scoring": "content_likelihood",
                "prompt_key": prompt_key,
                "normalization": normalization,
                "continuation_prefix": continuation_prefix,
            }
        )
        expected_config_ids = [
            label_config_id,
            content_config_id,
            *[str(selected["config_id"]) for _name, selected in selected_named],
        ]
        for row in tqdm(evaluation_rows, desc=f"{exp['name']}:evaluation"):
            expected_result_ids = [
                f"{row['sample_uid']}::{config_id}" for config_id in expected_config_ids
            ]
            if all(writer.has(result_id) for result_id in expected_result_ids):
                continue
            payload = build_payload(row)
            row = payload["row"]
            label_score = self.score_mc(row)
            content_score = payload["full"]
            common = {
                "experiment": exp["name"],
                "experiment_type": "semantic_gated_contrast",
                "question_id": row["question_id"],
                "question_form": "MC",
                "condition": row["condition"],
                "split": "evaluation",
                "scoring_method": "content_likelihood",
                "content_prompt_key": prompt_key,
                "content_normalization": normalization,
                "original_label_prediction": label_score["prediction"],
                "original_label_correct": label_score["is_correct"],
                "content_baseline_prediction": content_score["prediction"],
                "content_baseline_correct": content_score["is_correct"],
                "content_baseline_gold_margin": content_score["gold_margin"],
                "head_visual_signal": payload.get("head_signal"),
            }
            writer.append(
                {
                    **label_score,
                    **common,
                    "result_id": f"{row['sample_uid']}::{label_config_id}",
                    "config_id": label_config_id,
                    "control": "label_token_baseline",
                    "scoring_method": "label_token",
                    "baseline_correct": label_score["is_correct"],
                    "baseline_gold_margin": label_score["gold_margin"],
                    "margin_delta": 0.0,
                    "gate_applied": False,
                }
            )
            writer.append(
                {
                    **content_score,
                    **common,
                    "result_id": f"{row['sample_uid']}::{content_config_id}",
                    "config_id": content_config_id,
                    "control": "content_likelihood_baseline",
                    "baseline_correct": label_score["is_correct"],
                    "baseline_gold_margin": None,
                    "margin_delta": None,
                    "gate_applied": False,
                }
            )
            for selection_name, selected in selected_named:
                score, applied = apply_config(payload, selected)
                config_id = str(selected["config_id"])
                prior = str(selected["prior"])
                gate_type = str(selected["gate_type"])
                gate_threshold = selected.get(
                    "head_threshold",
                    selected.get("js_threshold", selected.get("margin_threshold")),
                )
                writer.append(
                    {
                        **score,
                        **common,
                        "result_id": f"{row['sample_uid']}::{config_id}",
                        "config_id": config_id,
                        "control": f"semantic_contrast_{selection_name}",
                        "prior": prior,
                        "alpha": selected["alpha"],
                        "plausibility_beta": selected["plausibility_beta"],
                        "gate_type": gate_type,
                        "gate_threshold": gate_threshold,
                        "gate_applied": applied,
                        "full_prior_js": payload["js"][prior],
                        "prior_prediction": payload["negatives"][prior]["prediction"],
                        "prior_candidate_log_probs": payload["negatives"][prior][
                            "candidate_log_probs"
                        ],
                        "baseline_prediction": content_score["prediction"],
                        "baseline_correct": content_score["is_correct"],
                        "baseline_gold_margin": content_score["gold_margin"],
                        "margin_delta": (
                            float(score["gold_margin"])
                            - float(content_score["gold_margin"])
                            if score.get("gold_margin") is not None
                            and content_score.get("gold_margin") is not None
                            else None
                        ),
                    }
                )
        writer.finalize()

    def run_mismatch_activation_export(
        self, exp: Mapping[str, Any], profile: str
    ) -> None:
        """Export length-matched Full/Blurred/Shuffled activations once.

        Calibration captures every requested layer.  Evaluation can read the
        frozen CPU calibration artifact and capture only the selected layers.
        Each question is stored independently so an interrupted GPU run resumes
        without recomputing completed examples.
        """
        writer = ResultWriter(self.output_dir, str(exp["name"]))

        def profile_value(name: str, default: Any) -> Any:
            return exp.get(f"{profile}_{name}", exp.get(name, default))

        mode = os.path.expandvars(str(exp.get("mode", "calibration"))).strip().lower()
        if mode not in {"calibration", "evaluation"}:
            raise ValueError(
                "mismatch_activation_export mode must be calibration or evaluation"
            )
        limit_raw = profile_value("limit", None)
        limit = int(limit_raw) if limit_raw not in (None, "") else None
        position = str(exp.get("position", "last"))
        if position != "last":
            raise ValueError(
                "Mismatch export currently requires position='last' so the exact "
                "prompt-position checks remain interpretable"
            )
        blur_radius = float(exp.get("blur_radius", 12.0))
        shuffle_seed = int(exp.get("shuffle_seed", self.seed))
        canonical_size_raw = exp.get("canonical_image_size")
        canonical_image_size = (
            [int(value) for value in canonical_size_raw]
            if isinstance(canonical_size_raw, (list, tuple))
            else None
        )
        image_fit_mode = str(exp.get("image_fit_mode", "stretch"))
        strict_length_match = bool(exp.get("strict_length_match", True))
        prefer_manifest_shuffled = bool(
            exp.get("prefer_manifest_shuffled", True)
        )
        reuse_aligned_export_text = os.path.expandvars(
            str(exp.get("reuse_aligned_export", ""))
        ).strip()
        capture_text_confound = bool(
            exp.get(
                f"{mode}_capture_text_confound",
                exp.get("capture_text_confound", mode == "calibration"),
            )
        )
        permutation_count = int(
            exp.get(
                f"{mode}_permutation_sensitivity_count",
                profile_value(
                    "permutation_sensitivity_count", 1 if mode == "calibration" else 0
                ),
            )
        )
        if permutation_count < 0:
            raise ValueError("permutation_sensitivity_count must be non-negative")

        model_layers, _ = decoder_layers(self.model)
        configured_layers = exp.get("layers", "all")
        calibration_path_text = os.path.expandvars(
            str(exp.get("calibration_path", ""))
        ).strip()
        frozen_calibration: dict[str, Any] | None = None
        if mode == "evaluation":
            if not calibration_path_text:
                raise ValueError("Evaluation mismatch export requires calibration_path")
            calibration_path = expand_path(calibration_path_text)
            if not calibration_path.is_file():
                raise FileNotFoundError(
                    f"Mismatch calibration artifact not found: {calibration_path}"
                )
            with calibration_path.open("r", encoding="utf-8") as stream:
                frozen_calibration = json.load(stream)
            selected_groups = frozen_calibration.get("selected_head_groups", {})
            selected_pairs = {
                (int(layer), int(head))
                for values in selected_groups.values()
                for layer, head in values
            }
            if not selected_pairs:
                raise ValueError("Calibration artifact contains no selected heads")
            layer_indices = sorted({layer for layer, _head in selected_pairs})
        elif configured_layers in (None, "all"):
            layer_indices = list(range(len(model_layers)))
        else:
            layer_indices = sorted({int(value) for value in configured_layers})
        if not layer_indices:
            raise ValueError("Mismatch export has no layers to capture")
        for layer_idx in layer_indices:
            if not 0 <= layer_idx < len(model_layers):
                raise ValueError(f"Mismatch export layer out of range: {layer_idx}")

        def load_cached_payload(path: Path) -> dict[str, Any]:
            try:
                return torch.load(path, map_location="cpu", weights_only=False)
            except TypeError:
                return torch.load(path, map_location="cpu")

        reused_payload_paths: dict[str, Path] = {}
        reuse_aligned_export: Path | None = None
        if reuse_aligned_export_text:
            reuse_aligned_export = expand_path(reuse_aligned_export_text)
            reuse_manifest_path = reuse_aligned_export / "export_manifest.json"
            if not reuse_manifest_path.is_file():
                raise FileNotFoundError(
                    f"Missing aligned export manifest: {reuse_manifest_path}"
                )
            reuse_manifest = json.loads(
                reuse_manifest_path.read_text(encoding="utf-8")
            )
            reuse_cache_dir = Path(str(reuse_manifest.get("cache_dir", "")))
            if not reuse_cache_dir.is_dir():
                reuse_candidates = sorted(
                    reuse_aligned_export.glob("activation_cache_*")
                )
                if not reuse_candidates:
                    raise FileNotFoundError(
                        f"No aligned activation cache under {reuse_aligned_export}"
                    )
                reuse_cache_dir = reuse_candidates[-1]
            for reuse_path in sorted(reuse_cache_dir.glob("*.pt")):
                reuse_payload = load_cached_payload(reuse_path)
                reuse_question_id = str(reuse_payload.get("question_id"))
                if reuse_question_id in reused_payload_paths:
                    raise RuntimeError(
                        f"Duplicate question in aligned cache: {reuse_question_id}"
                    )
                reused_payload_paths[reuse_question_id] = reuse_path
            if not reused_payload_paths:
                raise FileNotFoundError(
                    f"No aligned activation shards in {reuse_cache_dir}"
                )

        legacy_heads = parse_heads(exp.get("legacy_heads", ["63:63", "63:45"]))
        legacy_heads = [
            (layer, head)
            for layer, head in legacy_heads
            if 0 <= layer < len(model_layers) and 0 <= head < self.num_heads
        ]
        storage_name = str(exp.get("storage_dtype", "float16")).lower()
        storage_dtypes = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        if storage_name not in storage_dtypes:
            raise ValueError(f"Unsupported activation storage dtype: {storage_name}")
        storage_dtype = storage_dtypes[storage_name]
        export_config = {
            "version": 2,
            "mode": mode,
            "layers": layer_indices,
            "num_heads": self.num_heads,
            "position": position,
            "strict_length_match": strict_length_match,
            "blur_radius": blur_radius,
            "shuffle_seed": shuffle_seed,
            "canonical_image_size": canonical_image_size,
            "image_fit_mode": image_fit_mode,
            "prefer_manifest_shuffled": prefer_manifest_shuffled,
            "reuse_aligned_export": (
                str(reuse_aligned_export) if reuse_aligned_export is not None else ""
            ),
            "capture_text_confound": capture_text_confound,
            "permutation_sensitivity_count": permutation_count,
            "storage_dtype": storage_name,
        }
        export_config_id = stable_hash(export_config, 16)
        cache_dir = writer.root / f"activation_cache_{export_config_id}"
        cache_dir.mkdir(parents=True, exist_ok=True)

        expanded = self.expanded_counterfactual_rows(
            ["full", "blurred", "shuffled", "text_only"],
            limit,
            blur_radius=blur_radius,
            shuffle_seed=shuffle_seed,
            match_image_size=True,
            canonical_image_size=canonical_image_size,
            image_fit_mode=image_fit_mode,
            prefer_manifest_shuffled=prefer_manifest_shuffled,
        )
        grouped: dict[str, dict[str, dict[str, Any]]] = {}
        for row in expanded:
            grouped.setdefault(str(row.get("question_id")), {})[
                norm_condition(row.get("condition"))
            ] = row
        required_conditions = {"full", "blurred", "shuffled", "text_only"}
        incomplete = {
            question_id: sorted(required_conditions - set(rows_by_condition))
            for question_id, rows_by_condition in grouped.items()
            if required_conditions - set(rows_by_condition)
        }
        if incomplete:
            preview = list(incomplete.items())[:5]
            raise ValueError(f"Incomplete mismatch condition groups: {preview}")

        def tensor_digest(value: torch.Tensor | None) -> str | None:
            if value is None:
                return None
            cpu = value.detach().cpu().contiguous()
            return hashlib.sha1(cpu.numpy().tobytes()).hexdigest()

        def input_signature(prepared: PreparedInput) -> dict[str, Any]:
            inputs = prepared.inputs
            last_positions = torch.nonzero(
                prepared.masks["last"][0], as_tuple=False
            ).flatten()
            signature: dict[str, Any] = {
                "input_length": int(inputs["input_ids"].shape[-1]),
                "attention_valid": int(inputs["attention_mask"].sum().item()),
                "image_token_count": int(prepared.masks["image"].sum().item()),
                "last_position": int(last_positions[-1].item()),
                "input_ids_digest": tensor_digest(inputs.get("input_ids")),
                "mm_token_type_ids_digest": tensor_digest(
                    inputs.get("mm_token_type_ids")
                ),
                "position_ids_digest": tensor_digest(inputs.get("position_ids")),
            }
            grid = inputs.get("image_grid_thw")
            signature["image_grid_thw"] = (
                grid.detach().cpu().tolist()
                if isinstance(grid, torch.Tensor)
                else None
            )
            return signature

        def capture_row(
            row: Mapping[str, Any], *, keep_activation: bool
        ) -> tuple[dict[str, Any], torch.Tensor | None, dict[str, Any]]:
            prepared = self.prepare(row)
            capture = MultiLayerHeadCapture(
                self.model, prepared, layer_indices, position
            )
            score = self._score_prepared(prepared, capture)
            missing = sorted(set(layer_indices) - set(capture.values))
            if missing:
                raise RuntimeError(
                    f"Mismatch capture hooks did not run for layers: {missing}"
                )
            vectors = torch.stack(
                [
                    self.pooled_vector(capture.values[layer_idx]).float()
                    for layer_idx in layer_indices
                ]
            )
            if vectors.shape[-1] % self.num_heads:
                raise RuntimeError(
                    f"Captured width {vectors.shape[-1]} is not divisible by "
                    f"{self.num_heads} heads"
                )
            heads = vectors.reshape(
                len(layer_indices), self.num_heads, vectors.shape[-1] // self.num_heads
            )
            signature = input_signature(prepared)
            self.mc_cache[str(row.get("sample_uid"))] = dict(score)
            del prepared, capture, vectors
            if keep_activation:
                return score, heads.to(dtype=storage_dtype).contiguous(), signature
            return score, heads, signature

        def paired_distances(
            source: torch.Tensor, target: torch.Tensor
        ) -> dict[str, torch.Tensor]:
            source = source.float()
            target = target.float()
            source_norm = torch.linalg.vector_norm(source, dim=-1)
            target_norm = torch.linalg.vector_norm(target, dim=-1)
            symmetric_l2 = torch.linalg.vector_norm(source - target, dim=-1) / (
                0.5 * (source_norm + target_norm)
            ).clamp_min(1e-6)
            cosine = 1.0 - F.cosine_similarity(source, target, dim=-1)
            return {
                "symmetric_l2": symmetric_l2.to(torch.float16),
                "cosine": cosine.to(torch.float16),
                "source_norm": source_norm.to(torch.float16),
                "target_norm": target_norm.to(torch.float16),
            }

        signature_keys = (
            "input_length",
            "attention_valid",
            "image_token_count",
            "last_position",
            "input_ids_digest",
            "mm_token_type_ids_digest",
            "position_ids_digest",
            "image_grid_thw",
        )

        for question_id in tqdm(sorted(grouped), desc=f"{exp['name']}:{mode}"):
            rows_by_condition = grouped[question_id]
            result_id = f"{question_id}::{export_config_id}"
            cache_path = cache_dir / f"{stable_hash(question_id, 20)}.pt"
            if writer.has(result_id) and cache_path.is_file():
                continue

            reused_payload: dict[str, Any] | None = None
            if reuse_aligned_export is not None:
                reused_path = reused_payload_paths.get(question_id)
                if reused_path is None:
                    raise RuntimeError(
                        f"Aligned activation cache has no question {question_id}"
                    )
                reused_payload = load_cached_payload(reused_path)
                reused_layers = [int(value) for value in reused_payload["layers"]]
                if reused_layers != layer_indices:
                    raise RuntimeError(
                        "Aligned cache layers do not match the frozen calibration: "
                        f"cached={reused_layers}, requested={layer_indices}"
                    )
                if int(reused_payload["num_heads"]) != self.num_heads:
                    raise RuntimeError(
                        "Aligned cache num_heads does not match the loaded model"
                    )
                full_score = dict(reused_payload["scores"]["full"])
                blurred_score = dict(reused_payload["scores"]["blurred"])
                text_score = dict(reused_payload["scores"]["text_only"])
                full_heads = reused_payload["activations"]["full"].to(
                    dtype=storage_dtype
                )
                blurred_heads = reused_payload["activations"]["blurred"].to(
                    dtype=storage_dtype
                )
                full_signature = dict(reused_payload["signatures"]["full"])
                blurred_signature = dict(reused_payload["signatures"]["blurred"])
                text_signature = dict(reused_payload["signatures"]["text_only"])
                text_heads = None
            else:
                full_score, full_heads, full_signature = capture_row(
                    rows_by_condition["full"], keep_activation=True
                )
                blurred_score, blurred_heads, blurred_signature = capture_row(
                    rows_by_condition["blurred"], keep_activation=True
                )
            shuffled_score, shuffled_heads, shuffled_signature = capture_row(
                rows_by_condition["shuffled"], keep_activation=True
            )
            assert full_heads is not None
            assert blurred_heads is not None
            assert shuffled_heads is not None

            mismatches: dict[str, dict[str, Any]] = {}
            for condition, signature in (
                ("blurred", blurred_signature),
                ("shuffled", shuffled_signature),
            ):
                different = {
                    key: {"full": full_signature.get(key), condition: signature.get(key)}
                    for key in signature_keys
                    if full_signature.get(key) != signature.get(key)
                }
                if different:
                    mismatches[condition] = different
            if mismatches and strict_length_match:
                raise RuntimeError(
                    f"Length/grid matching failed for question {question_id}: "
                    f"{json.dumps(mismatches, ensure_ascii=False)}"
                )

            if reused_payload is None:
                text_score, text_heads, text_signature = capture_row(
                    rows_by_condition["text_only"],
                    keep_activation=capture_text_confound,
                )
            distances: dict[str, Any] = {
                "full_blurred": (
                    reused_payload["distances"]["full_blurred"]
                    if reused_payload is not None
                    else paired_distances(full_heads, blurred_heads)
                ),
                "full_shuffled": paired_distances(full_heads, shuffled_heads),
            }
            if capture_text_confound and text_heads is not None:
                distances["full_text"] = paired_distances(full_heads, text_heads)
            elif (
                capture_text_confound
                and reused_payload is not None
                and "full_text" in reused_payload.get("distances", {})
            ):
                distances["full_text"] = reused_payload["distances"]["full_text"]

            permutation_distances: list[torch.Tensor] = []
            permutation_signatures: list[dict[str, Any]] = []
            for permutation_index in range(1, permutation_count + 1):
                permuted, _permutation, _gold = self.permute_options(
                    rows_by_condition["full"], permutation_index, self.seed
                )
                _score, permuted_heads, permuted_signature = capture_row(
                    permuted, keep_activation=False
                )
                assert permuted_heads is not None
                permutation_distances.append(
                    paired_distances(full_heads, permuted_heads)["symmetric_l2"]
                )
                permutation_signatures.append(permuted_signature)
            if permutation_distances:
                distances["option_permutation"] = torch.stack(
                    permutation_distances
                ).mean(dim=0)
            elif (
                reused_payload is not None
                and "option_permutation" in reused_payload.get("distances", {})
            ):
                distances["option_permutation"] = reused_payload["distances"][
                    "option_permutation"
                ]

            payload = {
                "version": 2,
                "config_id": export_config_id,
                "question_id": question_id,
                "layers": layer_indices,
                "num_heads": self.num_heads,
                "head_dim": int(full_heads.shape[-1]),
                "activations": {
                    "full": full_heads,
                    "blurred": blurred_heads,
                    "shuffled": shuffled_heads,
                },
                "scores": {
                    "full": full_score,
                    "blurred": blurred_score,
                    "shuffled": shuffled_score,
                    "text_only": text_score,
                },
                "signatures": {
                    "full": full_signature,
                    "blurred": blurred_signature,
                    "shuffled": shuffled_signature,
                    "text_only": text_signature,
                    "option_permutations": permutation_signatures,
                },
                "signature_mismatches": mismatches,
                "distances": distances,
                "metadata": {
                    "full_image_name": rows_by_condition["full"].get(
                        "input_image_name"
                    ),
                    "shuffled_image_name": rows_by_condition["shuffled"].get(
                        "input_image_name"
                    ),
                    "shuffled_from_question_id": rows_by_condition["shuffled"].get(
                        "shuffled_from_question_id"
                    ),
                    "synthetic_shuffled": bool(
                        rows_by_condition["shuffled"].get("synthetic_condition")
                    ),
                    "shuffle_seed": shuffle_seed,
                    "reused_aligned_conditions": reused_payload is not None,
                },
            }
            temporary_path = cache_path.with_suffix(".tmp")
            torch.save(payload, temporary_path)
            os.replace(temporary_path, cache_path)
            writer.append(
                {
                    "result_id": result_id,
                    "experiment": exp["name"],
                    "experiment_type": "mismatch_activation_export",
                    "config_id": export_config_id,
                    "question_id": question_id,
                    "question_form": "MC",
                    "condition": "paired_counterfactuals",
                    "control": "length_matched_activation_export",
                    "cache_file": str(cache_path),
                    "length_match_ok": not bool(mismatches),
                    "full_correct": full_score.get("is_correct"),
                    "blurred_correct": blurred_score.get("is_correct"),
                    "shuffled_correct": shuffled_score.get("is_correct"),
                    "text_only_correct": text_score.get("is_correct"),
                    "full_input_length": full_signature["input_length"],
                    "visual_token_count": full_signature["image_token_count"],
                    "image_grid_thw": full_signature["image_grid_thw"],
                    "shuffle_seed": shuffle_seed,
                    "shuffled_image_name": rows_by_condition["shuffled"].get(
                        "input_image_name"
                    ),
                    "shuffled_from_question_id": rows_by_condition["shuffled"].get(
                        "shuffled_from_question_id"
                    ),
                    "synthetic_shuffled": bool(
                        rows_by_condition["shuffled"].get("synthetic_condition")
                    ),
                    "reused_aligned_conditions": reused_payload is not None,
                }
            )
            del full_heads, blurred_heads, shuffled_heads, text_heads, payload

        manifest = {
            **export_config,
            "config_id": export_config_id,
            "cache_dir": str(cache_dir),
            "n_questions": len(grouped),
            "legacy_heads": [list(value) for value in legacy_heads],
            "calibration_path": calibration_path_text if mode == "evaluation" else "",
            "selection_uses_correctness_labels": False,
            "notes": [
                "Full/Blurred/Shuffled share identical prompt length, visual grid, and last position.",
                "Full/Text distance is exported only as a confound diagnostic and is never used for head selection.",
                "Mismatch detector labels are condition labels; correctness labels are used only to evaluate fallback utility.",
            ],
        }
        with (writer.root / "export_manifest.json").open(
            "w", encoding="utf-8"
        ) as stream:
            json.dump(jsonable(manifest), stream, ensure_ascii=False, indent=2)
        writer.finalize()

    def run_permutation_ensemble(self, exp: Mapping[str, Any], profile: str) -> None:
        writer = ResultWriter(self.output_dir, str(exp["name"]))
        limit = exp.get(f"{profile}_limit", exp.get("limit"))
        if bool(exp.get("expand_counterfactuals", False)):
            rows = self.expanded_counterfactual_rows(
                exp.get("conditions", ["full"]),
                limit,
                blur_radius=float(exp.get("blur_radius", 12.0)),
                shuffle_seed=int(exp.get("shuffle_seed", self.seed)),
                match_image_size=True,
                canonical_image_size=exp.get("canonical_image_size"),
                image_fit_mode=str(exp.get("image_fit_mode", "stretch")),
            )
        else:
            rows = self.rows_filtered(["MC"], exp.get("conditions", ["full"]), limit)
        counts = [
            int(value)
            for value in exp.get(
                f"{profile}_permutation_counts",
                exp.get("permutation_counts", [3, 6]),
            )
        ]
        for row in tqdm(rows, desc=str(exp["name"])):
            options = parse_maybe_list(row.get("options_json"))
            baseline = self.score_mc(row)
            for count in counts:
                semantic_scores: list[list[float]] = []
                permutations: list[list[int]] = []
                for permutation_index in range(count):
                    permuted, permutation, _gold = self.permute_options(
                        row, permutation_index, self.seed
                    )
                    result = self.score_mc(permuted)
                    restored = [0.0] * len(options)
                    for new_index, old_index in enumerate(permutation):
                        restored[old_index] = float(result["candidate_log_probs"][new_index])
                    semantic_scores.append(restored)
                    permutations.append(permutation)
                averaged = torch.tensor(semantic_scores).mean(dim=0).tolist()
                score = score_candidate_values(row, options, averaged)
                config = {
                    "permutation_count": count,
                    "aggregation": "mean_logprob",
                    "canonical_image_size": exp.get("canonical_image_size"),
                    "image_fit_mode": exp.get("image_fit_mode", "stretch"),
                }
                config_id = stable_hash(config)
                result_id = f"{row['sample_uid']}::{config_id}"
                if writer.has(result_id):
                    continue
                writer.append(
                    {
                        **score,
                        "result_id": result_id,
                        "experiment": exp["name"],
                        "experiment_type": "permutation_ensemble",
                        "config_id": config_id,
                        "question_id": row["question_id"],
                        "question_form": "MC",
                        "condition": row["condition"],
                        "control": "permutation_ensemble",
                        "permutation_count": count,
                        "permutations_new_to_old": permutations,
                        "baseline_prediction": baseline["prediction"],
                        "baseline_correct": baseline["is_correct"],
                        "baseline_gold_margin": baseline["gold_margin"],
                        "margin_delta": (
                            float(score["gold_margin"]) - float(baseline["gold_margin"])
                            if score.get("gold_margin") is not None
                            and baseline.get("gold_margin") is not None
                            else None
                        ),
                    }
                )
        writer.finalize()

    def run_always_on_evidence_first(
        self, exp: Mapping[str, Any], profile: str
    ) -> None:
        """Generate option-hidden visual evidence, then answer in a second pass."""
        writer = ResultWriter(self.output_dir, str(exp["name"]))
        limit_raw = exp.get(f"{profile}_limit", exp.get("limit"))
        limit = int(limit_raw) if limit_raw not in (None, "") else None
        conditions = exp.get("conditions", ["full", "blurred", "shuffled"])
        if bool(exp.get("expand_counterfactuals", True)):
            rows = self.expanded_counterfactual_rows(
                conditions,
                limit,
                blur_radius=float(exp.get("blur_radius", 12.0)),
                shuffle_seed=int(exp.get("shuffle_seed", self.seed)),
                match_image_size=True,
                canonical_image_size=exp.get("canonical_image_size"),
                image_fit_mode=str(exp.get("image_fit_mode", "stretch")),
            )
        else:
            rows = self.rows_filtered(["MC"], conditions, limit)
        evidence_prompt_key = str(exp.get("evidence_prompt_key", "MC_EVIDENCE_1"))
        evidence_max_new_tokens = int(
            exp.get(
                f"{profile}_evidence_max_new_tokens",
                exp.get("evidence_max_new_tokens", 96),
            )
        )
        config = {
            "method": "always_on_evidence_first",
            "evidence_prompt_key": evidence_prompt_key,
            "evidence_max_new_tokens": evidence_max_new_tokens,
            "options_hidden_during_evidence": True,
            "canonical_image_size": exp.get("canonical_image_size"),
            "image_fit_mode": exp.get("image_fit_mode", "stretch"),
        }
        config_id = stable_hash(config)
        for row in tqdm(rows, desc=str(exp["name"])):
            result_id = f"{row['sample_uid']}::{config_id}"
            if writer.has(result_id):
                continue
            baseline = self.score_mc(row)
            evidence_text = self.generate(
                row,
                prompt_key_override=evidence_prompt_key,
                max_new_tokens_override=evidence_max_new_tokens,
            )
            evidence_row = deepcopy(dict(row))
            evidence_row["question"] = (
                f"{row.get('question')}\n\n"
                f"이미지에서 먼저 확인한 근거: {evidence_text}"
            )
            evidence_row["sample_uid"] = (
                f"{row.get('sample_uid')}::always_evidence::"
                f"{stable_hash(evidence_text, 10)}"
            )
            score = self.score_mc(evidence_row)
            normalized_evidence = re.sub(r"\s+", " ", evidence_text).strip().lower()
            semantic_options = [
                re.sub(r"\s+", " ", strip_option_label(option)).strip().lower()
                for option in parse_maybe_list(row.get("options_json"))
            ]
            exact_option_matches = [
                index
                for index, option in enumerate(semantic_options)
                if len(option) >= 4 and option in normalized_evidence
            ]
            answer_label_pattern = re.compile(
                r"(?:정답|답)\s*(?:은|는|:)?\s*([0-9]+|[A-Za-z])",
                flags=re.IGNORECASE,
            )
            label_leak_match = answer_label_pattern.search(evidence_text)
            writer.append(
                {
                    **score,
                    "result_id": result_id,
                    "experiment": exp["name"],
                    "experiment_type": "always_on_evidence_first",
                    "config_id": config_id,
                    "question_id": row["question_id"],
                    "question_form": "MC",
                    "condition": row["condition"],
                    "control": "always_on_evidence_first",
                    "scoring_method": "label_token",
                    "evidence_text": evidence_text,
                    "options_hidden_during_evidence": True,
                    "evidence_exact_option_match_indices": exact_option_matches,
                    "evidence_exact_option_match": bool(exact_option_matches),
                    "evidence_answer_label_leak": bool(label_leak_match),
                    "evidence_answer_label": (
                        label_leak_match.group(1) if label_leak_match else ""
                    ),
                    "baseline_prediction": baseline["prediction"],
                    "baseline_correct": baseline["is_correct"],
                    "baseline_gold_margin": baseline["gold_margin"],
                    "margin_delta": (
                        float(score["gold_margin"])
                        - float(baseline["gold_margin"])
                        if score.get("gold_margin") is not None
                        and baseline.get("gold_margin") is not None
                        else None
                    ),
                }
            )
        writer.finalize()

    def run_label_bias_calibration(self, exp: Mapping[str, Any], profile: str) -> None:
        """Estimate a label-position prior on a disjoint calibration subset."""
        writer = ResultWriter(self.output_dir, str(exp["name"]))
        rows = self.rows_filtered(["MC"], exp.get("conditions", ["full"]), None)
        calibration_limit = int(
            exp.get(f"{profile}_calibration_limit", exp.get("calibration_limit", 24))
        )
        evaluation_limit = exp.get(f"{profile}_limit", exp.get("limit"))
        permutation_count = int(
            exp.get(
                f"{profile}_calibration_permutations",
                exp.get("calibration_permutations", 3),
            )
        )
        gammas = [
            float(value)
            for value in exp.get(f"{profile}_gammas", exp.get("gammas", [0.5, 1.0]))
        ]
        calibration_rows = rows[:calibration_limit]
        evaluation_rows = rows[calibration_limit:]
        if evaluation_limit is not None:
            evaluation_rows = evaluation_rows[: int(evaluation_limit)]
        bias_samples: dict[int, list[list[float]]] = {}
        for row in tqdm(calibration_rows, desc=f"{exp['name']}:calibrate"):
            option_count = len(parse_maybe_list(row.get("options_json")))
            for permutation_index in range(permutation_count):
                permuted, _permutation, _gold = self.permute_options(
                    row, permutation_index + 1, self.seed
                )
                values = [
                    float(value)
                    for value in self.score_mc(permuted)["candidate_log_probs"]
                ]
                center = sum(values) / len(values)
                bias_samples.setdefault(option_count, []).append(
                    [value - center for value in values]
                )
        bias_by_count = {
            count: torch.tensor(samples).mean(dim=0).tolist()
            for count, samples in bias_samples.items()
        }
        with (writer.root / "estimated_label_bias.json").open("w", encoding="utf-8") as stream:
            json.dump(bias_by_count, stream, ensure_ascii=False, indent=2)
        for row in tqdm(evaluation_rows, desc=f"{exp['name']}:evaluate"):
            options = parse_maybe_list(row.get("options_json"))
            bias = bias_by_count.get(len(options))
            if bias is None:
                continue
            baseline = self.score_mc(row)
            baseline_values = [float(value) for value in baseline["candidate_log_probs"]]
            for gamma in gammas:
                corrected = [
                    value - gamma * float(prior)
                    for value, prior in zip(baseline_values, bias)
                ]
                score = score_candidate_values(row, options, corrected)
                config = {
                    "gamma": gamma,
                    "calibration_limit": calibration_limit,
                    "calibration_permutations": permutation_count,
                    "option_count": len(options),
                }
                config_id = stable_hash(config)
                result_id = f"{row['sample_uid']}::{config_id}"
                if writer.has(result_id):
                    continue
                writer.append(
                    {
                        **score,
                        "result_id": result_id,
                        "experiment": exp["name"],
                        "experiment_type": "label_bias_calibration",
                        "config_id": config_id,
                        "question_id": row["question_id"],
                        "question_form": "MC",
                        "condition": row["condition"],
                        "control": "label_bias_correction",
                        "gamma": gamma,
                        "estimated_label_bias": bias,
                        "calibration_question_ids": [
                            item["question_id"] for item in calibration_rows
                        ],
                        "baseline_prediction": baseline["prediction"],
                        "baseline_correct": baseline["is_correct"],
                        "baseline_gold_margin": baseline["gold_margin"],
                        "margin_delta": (
                            float(score["gold_margin"]) - float(baseline["gold_margin"])
                            if score.get("gold_margin") is not None
                            and baseline.get("gold_margin") is not None
                            else None
                        ),
                    }
                )
        writer.finalize()

    def run_symbol_invariant_vhr(self, exp: Mapping[str, Any], profile: str) -> None:
        """Reinforce heads that are image-sensitive but permutation-stable."""
        writer = ResultWriter(self.output_dir, str(exp["name"]))
        limit = exp.get(f"{profile}_limit", exp.get("limit"))
        rows = self.rows_filtered(["MC"], exp.get("conditions", ["full"]), limit)
        layer_idx = int(exp.get("layer", 22))
        position = str(exp.get("selection_position", "last"))
        permutation_count = int(
            exp.get(f"{profile}_permutations", exp.get("permutations", 2))
        )
        factors = [
            float(value)
            for value in exp.get(
                f"{profile}_scale_factors", exp.get("scale_factors", [1.1, 1.2])
            )
        ]
        for row in tqdm(rows, desc=str(exp["name"])):
            baseline = self.score_mc(row)
            full_heads = self._head_vectors(
                self.capture_values(row, "head", layer_idx, position), self.num_heads
            )
            text_heads = self._head_vectors(
                self.capture_values(
                    self.synthesize_text_only(row), "head", layer_idx, position
                ),
                self.num_heads,
            )
            vhd = torch.linalg.vector_norm(full_heads - text_heads, dim=-1)
            permutation_distances: list[torch.Tensor] = []
            for permutation_index in range(1, permutation_count + 1):
                permuted, _permutation, _gold = self.permute_options(
                    row, permutation_index, self.seed
                )
                permuted_heads = self._head_vectors(
                    self.capture_values(permuted, "head", layer_idx, position),
                    self.num_heads,
                )
                denominator = torch.linalg.vector_norm(full_heads, dim=-1).clamp_min(1e-6)
                permutation_distances.append(
                    torch.linalg.vector_norm(permuted_heads - full_heads, dim=-1)
                    / denominator
                )
            symbol_sensitivity = torch.stack(permutation_distances).mean(dim=0)
            vhd_threshold = torch.median(vhd)
            symbol_threshold = torch.median(symbol_sensitivity)
            selected = [
                (layer_idx, head)
                for head in range(self.num_heads)
                if float(vhd[head]) > float(vhd_threshold)
                and float(symbol_sensitivity[head]) < float(symbol_threshold)
            ]
            for factor in factors:
                config = {
                    "layer": layer_idx,
                    "factor": factor,
                    "permutations": permutation_count,
                    "rule": "vhd_above_median_and_symbol_sensitivity_below_median",
                }
                config_id = stable_hash(config)
                result_id = f"{row['sample_uid']}::{config_id}"
                if writer.has(result_id):
                    continue

                def factory(
                    prepared: PreparedInput,
                    selected: list[tuple[int, int]] = selected,
                    factor: float = factor,
                ) -> HeadOutputScaler:
                    return HeadOutputScaler(
                        self.model,
                        prepared,
                        selected,
                        factor - 1.0,
                        self.num_heads,
                        "first_token",
                        "last",
                    )

                score = self.score_mc(row, factory, cache_baseline=False)
                writer.append(
                    {
                        **score,
                        "result_id": result_id,
                        "experiment": exp["name"],
                        "experiment_type": "symbol_invariant_vhr",
                        "config_id": config_id,
                        "question_id": row["question_id"],
                        "question_form": "MC",
                        "condition": row["condition"],
                        "control": "symbol_invariant_vhr",
                        "layer": layer_idx,
                        "factor": factor,
                        "alpha": factor - 1.0,
                        "selected_heads": selected,
                        "vhd": vhd.tolist(),
                        "symbol_sensitivity_proxy": symbol_sensitivity.tolist(),
                        "permutations": permutation_count,
                        "baseline_prediction": baseline["prediction"],
                        "baseline_correct": baseline["is_correct"],
                        "baseline_gold_margin": baseline["gold_margin"],
                        "margin_delta": (
                            float(score["gold_margin"]) - float(baseline["gold_margin"])
                            if score.get("gold_margin") is not None
                            and baseline.get("gold_margin") is not None
                            else None
                        ),
                    }
                )
        writer.finalize()

    def run_global_layer_discovery(self, exp: Mapping[str, Any], profile: str) -> None:
        """Scan all layers efficiently, then causally test only ranked candidates.

        Candidate selection is label-free: mean Full/Text VHD is discounted by
        option-permutation sensitivity. Correct/wrong labels are reported only as
        diagnostics and never enter the ranking score.
        """
        writer = ResultWriter(self.output_dir, str(exp["name"]))
        model_layers, _ = decoder_layers(self.model)
        configured_layers = exp.get("layers", "all")
        if configured_layers == "all" or configured_layers is None:
            layer_indices = list(range(len(model_layers)))
        else:
            layer_indices = [int(value) for value in configured_layers]
        scan_limit = exp.get(f"{profile}_scan_limit", exp.get("scan_limit"))
        scan_rows = self.rows_filtered(
            ["MC"], exp.get("conditions", ["full"]), scan_limit
        )
        position = str(exp.get("selection_position", "last"))
        permutation_count = int(
            exp.get(
                f"{profile}_scan_permutations",
                exp.get("scan_permutations", 1),
            )
        )
        scan_config = {
            "layers": layer_indices,
            "position": position,
            "permutations": permutation_count,
            "metric": "mean_vhd_discounted_by_permutation_sensitivity",
        }
        scan_config_id = stable_hash(scan_config)

        for row in tqdm(scan_rows, desc=f"{exp['name']}:representation_scan"):
            expected_ids = {
                layer_idx: f"{row['sample_uid']}::{scan_config_id}::layer_{layer_idx}"
                for layer_idx in layer_indices
            }
            if all(writer.has(result_id) for result_id in expected_ids.values()):
                continue
            baseline, full_values = self.score_and_capture_head_values_many(
                row, layer_indices, position
            )
            text_values = self.capture_head_values_many(
                self.synthesize_text_only(row), layer_indices, position
            )
            permuted_values: list[dict[int, torch.Tensor]] = []
            for permutation_index in range(1, permutation_count + 1):
                permuted, _permutation, _gold = self.permute_options(
                    row, permutation_index, self.seed
                )
                permuted_values.append(
                    self.capture_head_values_many(permuted, layer_indices, position)
                )
            for layer_idx in layer_indices:
                result_id = expected_ids[layer_idx]
                if writer.has(result_id):
                    continue
                full_heads = self._head_vectors(
                    full_values[layer_idx], self.num_heads
                )
                text_heads = self._head_vectors(
                    text_values[layer_idx], self.num_heads
                )
                vhd = torch.linalg.vector_norm(full_heads - text_heads, dim=-1)
                full_norm = torch.linalg.vector_norm(full_heads, dim=-1)
                text_norm = torch.linalg.vector_norm(text_heads, dim=-1)
                if permuted_values:
                    permutation_distances = []
                    denominator = full_norm.clamp_min(1e-6)
                    for captured in permuted_values:
                        permuted_heads = self._head_vectors(
                            captured[layer_idx], self.num_heads
                        )
                        permutation_distances.append(
                            torch.linalg.vector_norm(
                                permuted_heads - full_heads, dim=-1
                            )
                            / denominator
                        )
                    symbol_sensitivity = torch.stack(permutation_distances).mean(dim=0)
                else:
                    symbol_sensitivity = torch.zeros_like(vhd)
                writer.append(
                    {
                        "result_id": result_id,
                        "experiment": exp["name"],
                        "experiment_type": "global_layer_discovery",
                        "config_id": f"{scan_config_id}_L{layer_idx}",
                        "question_id": row["question_id"],
                        "question_form": "MC",
                        "condition": row["condition"],
                        "control": "representation_scan",
                        "layer": layer_idx,
                        "position": position,
                        "is_correct": baseline["is_correct"],
                        "gold_margin": baseline["gold_margin"],
                        "baseline_correct": baseline["is_correct"],
                        "vhd": vhd.tolist(),
                        "full_head_norm": full_norm.tolist(),
                        "text_head_norm": text_norm.tolist(),
                        "symbol_sensitivity_proxy": symbol_sensitivity.tolist(),
                    }
                )

        representation_records = [
            record
            for record in writer.records.values()
            if record.get("control") == "representation_scan"
            and int(record.get("layer", -1)) in layer_indices
            and str(record.get("config_id", "")).startswith(scan_config_id)
        ]
        records_by_layer: dict[int, list[Mapping[str, Any]]] = {}
        for record in representation_records:
            records_by_layer.setdefault(int(record["layer"]), []).append(record)
        ranking_rows: list[dict[str, Any]] = []
        head_rankings: dict[int, list[int]] = {}
        for layer_idx in layer_indices:
            records = records_by_layer.get(layer_idx, [])
            if not records:
                continue
            vhd_matrix = torch.tensor([record["vhd"] for record in records]).float()
            symbol_matrix = torch.tensor(
                [record["symbol_sensitivity_proxy"] for record in records]
            ).float()
            text_norm_matrix = torch.tensor(
                [record["text_head_norm"] for record in records]
            ).float()
            mean_vhd = vhd_matrix.mean(dim=0)
            mean_symbol = symbol_matrix.mean(dim=0)
            mean_text_norm = text_norm_matrix.mean(dim=0)
            negative = (
                mean_vhd > mean_vhd.mean() + mean_vhd.std(unbiased=False)
            ) & (
                mean_text_norm
                > mean_text_norm.mean() + mean_text_norm.std(unbiased=False)
            )
            head_score = mean_vhd / (1.0 + mean_symbol)
            head_score[negative] = 0.0
            ordered_heads = torch.argsort(head_score, descending=True).tolist()
            head_rankings[layer_idx] = [int(value) for value in ordered_heads]
            top_count = max(1, math.ceil(self.num_heads * float(exp.get("top_fraction", 0.25))))
            top_heads = ordered_heads[:top_count]
            layer_score = float(head_score[top_heads].mean().item())
            correct_mask = torch.tensor(
                [record.get("is_correct") is True for record in records],
                dtype=torch.bool,
            )
            wrong_mask = ~correct_mask
            correct_vhd = (
                float(vhd_matrix[correct_mask].mean().item())
                if bool(correct_mask.any())
                else float("nan")
            )
            wrong_vhd = (
                float(vhd_matrix[wrong_mask].mean().item())
                if bool(wrong_mask.any())
                else float("nan")
            )
            ranking_rows.append(
                {
                    "layer": layer_idx,
                    "n_samples": len(records),
                    "layer_score_label_free": layer_score,
                    "mean_vhd": float(mean_vhd.mean().item()),
                    "mean_symbol_sensitivity": float(mean_symbol.mean().item()),
                    "mean_vhd_baseline_correct_diagnostic": correct_vhd,
                    "mean_vhd_baseline_wrong_diagnostic": wrong_vhd,
                    "top_heads": ordered_heads[: int(exp.get("report_top_heads", 8))],
                    "top_head_scores": [
                        float(head_score[head].item())
                        for head in ordered_heads[: int(exp.get("report_top_heads", 8))]
                    ],
                    "negative_sensitivity_heads": torch.nonzero(
                        negative, as_tuple=False
                    ).flatten().tolist(),
                    "mean_vhd_by_head": mean_vhd.tolist(),
                    "mean_symbol_sensitivity_by_head": mean_symbol.tolist(),
                }
            )
        ranking_rows.sort(
            key=lambda record: float(record["layer_score_label_free"]), reverse=True
        )
        write_csv(writer.root / "layer_ranking.csv", ranking_rows)

        candidate_count = int(
            exp.get(
                f"{profile}_candidate_layer_count",
                exp.get("candidate_layer_count", 5),
            )
        )
        minimum_gap = int(exp.get("minimum_layer_gap", 2))
        candidate_layers: list[int] = []
        for record in ranking_rows:
            layer_idx = int(record["layer"])
            if all(abs(layer_idx - existing) >= minimum_gap for existing in candidate_layers):
                candidate_layers.append(layer_idx)
            if len(candidate_layers) >= candidate_count:
                break
        selected_head_count = int(exp.get("selected_head_count", 4))
        candidate_heads = {
            layer_idx: head_rankings[layer_idx][:selected_head_count]
            for layer_idx in candidate_layers
        }
        selection_payload = {
            "selection_uses_labels": False,
            "selection_metric": "mean_vhd/(1+mean_permutation_sensitivity)",
            "candidate_layers": candidate_layers,
            "candidate_heads": candidate_heads,
            "minimum_layer_gap": minimum_gap,
            "scan_question_ids": [row["question_id"] for row in scan_rows],
        }
        with (writer.root / "selected_candidates.json").open(
            "w", encoding="utf-8"
        ) as stream:
            json.dump(selection_payload, stream, ensure_ascii=False, indent=2)

        causal_limit = exp.get(f"{profile}_causal_limit", exp.get("causal_limit"))
        causal_rows = scan_rows if causal_limit is None else scan_rows[: int(causal_limit)]
        factors = [
            float(value)
            for value in exp.get(
                f"{profile}_scale_factors", exp.get("scale_factors", [1.1, 1.2])
            )
        ]
        random_set_count = int(
            exp.get(
                f"{profile}_random_set_count", exp.get("random_set_count", 2)
            )
        )
        head_sets_by_layer: dict[int, list[tuple[str, list[int]]]] = {}
        for layer_idx in candidate_layers:
            selected = candidate_heads[layer_idx]
            head_sets = [("ranked_top", selected)]
            available = [
                head for head in range(self.num_heads) if head not in set(selected)
            ]
            for random_index in range(random_set_count):
                rng = random.Random(f"{self.seed}:{layer_idx}:{random_index}")
                head_sets.append(
                    (
                        f"random_{random_index + 1}",
                        sorted(rng.sample(available, selected_head_count)),
                    )
                )
            head_sets_by_layer[layer_idx] = head_sets

        for row in tqdm(causal_rows, desc=f"{exp['name']}:causal_scan"):
            baseline = self.score_mc(row)
            for layer_idx in candidate_layers:
                for head_set_name, heads in head_sets_by_layer[layer_idx]:
                    selected_pairs = [(layer_idx, head) for head in heads]
                    for factor in factors:
                        config = {
                            "layer": layer_idx,
                            "head_set": head_set_name,
                            "heads": heads,
                            "factor": factor,
                            "selection_config": scan_config_id,
                        }
                        config_id = stable_hash(config)
                        result_id = f"{row['sample_uid']}::{config_id}"
                        if writer.has(result_id):
                            continue

                        def factory(
                            prepared: PreparedInput,
                            selected_pairs: list[tuple[int, int]] = selected_pairs,
                            factor: float = factor,
                        ) -> HeadOutputScaler:
                            return HeadOutputScaler(
                                self.model,
                                prepared,
                                selected_pairs,
                                factor - 1.0,
                                self.num_heads,
                                "first_token",
                                "last",
                            )

                        score = self.score_mc(row, factory, cache_baseline=False)
                        writer.append(
                            {
                                **score,
                                "result_id": result_id,
                                "experiment": exp["name"],
                                "experiment_type": "global_layer_discovery",
                                "config_id": config_id,
                                "question_id": row["question_id"],
                                "question_form": "MC",
                                "condition": row["condition"],
                                "control": "candidate_causal_scan",
                                "layer": layer_idx,
                                "head_set": head_set_name,
                                "heads": heads,
                                "factor": factor,
                                "alpha": factor - 1.0,
                                "baseline_prediction": baseline["prediction"],
                                "baseline_correct": baseline["is_correct"],
                                "baseline_gold_margin": baseline["gold_margin"],
                                "margin_delta": (
                                    float(score["gold_margin"])
                                    - float(baseline["gold_margin"])
                                    if score.get("gold_margin") is not None
                                    and baseline.get("gold_margin") is not None
                                    else None
                                ),
                            }
                        )
        writer.finalize()

    def run_backpatch(self, exp: Mapping[str, Any], profile: str) -> None:
        writer = ResultWriter(self.output_dir, str(exp["name"]))
        limit = exp.get(f"{profile}_limit", exp.get("limit"))
        rows = self.rows_filtered(["MC"], exp.get("conditions", ["full"]), limit)
        source_layers = [int(layer) for layer in exp.get("source_layers", [31, 35])]
        destination_layers = [int(layer) for layer in exp.get("destination_layers", [8, 16, 23])]
        positions = list(exp.get("positions", ["image"]))
        strengths = [float(value) for value in exp.get("strengths", [1.0])]

        for row in tqdm(rows, desc=str(exp["name"])):
            baseline = self.score_mc(row)
            for position in positions:
                for source_layer in source_layers:
                    source_values = self.capture_values(
                        row, "residual", source_layer, position
                    )
                    if source_values.shape[0] == 0:
                        continue
                    for destination_layer in destination_layers:
                        if source_layer <= destination_layer:
                            continue
                        for strength in strengths:
                            config = {
                                "source_layer": source_layer,
                                "destination_layer": destination_layer,
                                "position": position,
                                "strength": strength,
                            }
                            config_id = stable_hash(config)
                            result_id = f"{row['sample_uid']}::{config_id}"
                            if writer.has(result_id):
                                continue

                            def factory(
                                prepared: PreparedInput,
                                source_values: torch.Tensor = source_values,
                            ) -> ComponentPatcher:
                                return ComponentPatcher(
                                    self.model,
                                    prepared,
                                    "residual",
                                    destination_layer,
                                    position,
                                    source_values,
                                    strength,
                                    self.num_heads,
                                )

                            score = self.score_mc(row, factory, cache_baseline=False)
                            record = {
                                **score,
                                "result_id": result_id,
                                "experiment": exp["name"],
                                "experiment_type": "backpatch",
                                "config_id": config_id,
                                "question_id": row["question_id"],
                                "question_form": "MC",
                                "condition": row["condition"],
                                "source_layer": source_layer,
                                "destination_layer": destination_layer,
                                "position": position,
                                "strength": strength,
                                "baseline_correct": baseline["is_correct"],
                                "baseline_prediction": baseline["prediction"],
                                "baseline_gold_margin": baseline["gold_margin"],
                                "margin_delta": (
                                    float(score["gold_margin"])
                                    - float(baseline["gold_margin"])
                                    if score.get("gold_margin") is not None
                                    and baseline.get("gold_margin") is not None
                                    else None
                                ),
                            }
                            writer.append(record)
        writer.finalize()

    def run_activation_export(self, exp: Mapping[str, Any], profile: str) -> None:
        writer = ResultWriter(self.output_dir, str(exp["name"]))
        limit = exp.get(f"{profile}_limit", exp.get("limit"))
        forms = exp.get("forms")
        conditions = [norm_condition(value) for value in exp.get("conditions", [])]
        rows = self.rows_filtered(forms, conditions, None)
        if "text_only" in conditions and not any(
            row["condition"] == "text_only" for row in rows
        ):
            full_rows = self.rows_filtered(forms, ["full"], None)
            rows.extend(self.synthesize_text_only(row) for row in full_rows)
        if limit is not None and conditions:
            # Avoid a condition-sorted manifest filling the export with only Full rows.
            quota = max(1, math.ceil(int(limit) / len(conditions)))
            buckets = {
                condition: [row for row in rows if row["condition"] == condition][:quota]
                for condition in conditions
            }
            rows = [
                row
                for condition in conditions
                for row in buckets[condition]
            ][: int(limit)]
        elif limit is not None:
            rows = rows[: int(limit)]
        layers = [int(layer) for layer in exp.get("layers", [22, 31, 35])]
        components = list(exp.get("components", ["head", "residual"]))
        positions = list(exp.get("positions", ["last", "query", "image"]))
        vectors: list[torch.Tensor] = []
        metadata: list[dict[str, Any]] = []
        for row in tqdm(rows, desc=str(exp["name"])):
            for component in components:
                for layer_idx in layers:
                    for position in positions:
                        values = self.capture_values(row, component, layer_idx, position)
                        if values.numel() == 0:
                            continue
                        vector = self.pooled_vector(values).to(torch.float32)
                        vectors.append(vector)
                        record = {
                            "result_id": stable_hash(
                                [row["sample_uid"], component, layer_idx, position], 20
                            ),
                            "experiment": exp["name"],
                            "experiment_type": "activation_export",
                            "config_id": stable_hash([component, layer_idx, position]),
                            "question_id": row["question_id"],
                            "question_form": row["question_form"],
                            "condition": row["condition"],
                            "component": component,
                            "layer": layer_idx,
                            "position": position,
                            "vector_index": len(vectors) - 1,
                            "reference": row["reference"],
                            "task_type": row.get("task_type", ""),
                            "corpus_name": row.get("corpus_name", ""),
                        }
                        metadata.append(record)
                        writer.append(record)
        tensor_path = writer.root / "activations.pt"
        if vectors:
            dimensions = {int(vector.numel()) for vector in vectors}
            payload: dict[str, Any]
            if len(dimensions) == 1:
                payload = {"vectors": torch.stack(vectors), "metadata": metadata}
            else:
                payload = {"vectors": vectors, "metadata": metadata}
            torch.save(payload, tensor_path)
        writer.finalize()

    def run_experiment(self, exp: Mapping[str, Any], profile: str) -> None:
        experiment_type = str(exp["type"])
        print(f"\n=== {exp['name']} ({experiment_type}) ===")
        dispatch = {
            "generation_scaling": self.run_generation_scaling,
            "option_permutation": self.run_option_permutation,
            "patching_sweep": self.run_patching_sweep,
            "visual_contrast_reinforcement": self.run_visual_contrast_reinforcement,
            "source_control": self.run_source_control,
            "label_swap_control": self.run_label_swap_control,
            "bridge": self.run_bridge,
            "dynamic_vhd": self.run_dynamic_vhd,
            "faithful_vhr": self.run_faithful_vhr,
            "contrastive_decoding": self.run_contrastive_decoding,
            "gated_label_correction": self.run_gated_label_correction,
            "detector_disambiguation": self.run_detector_disambiguation,
            "semantic_gated_contrast": self.run_semantic_gated_contrast,
            "mismatch_activation_export": self.run_mismatch_activation_export,
            "permutation_ensemble": self.run_permutation_ensemble,
            "always_on_evidence_first": self.run_always_on_evidence_first,
            "label_bias_calibration": self.run_label_bias_calibration,
            "symbol_invariant_vhr": self.run_symbol_invariant_vhr,
            "global_layer_discovery": self.run_global_layer_discovery,
            "backpatch": self.run_backpatch,
            "activation_export": self.run_activation_export,
        }
        if experiment_type not in dispatch:
            raise ValueError(
                f"Unsupported experiment type {experiment_type!r}; "
                f"available={sorted(dispatch)}"
            )
        dispatch[experiment_type](exp, profile)


def resolve_dtype(name: str) -> torch.dtype:
    name = name.lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16", "half"}:
        return torch.float16
    if name in {"fp32", "float32", "float"}:
        return torch.float32
    if name == "auto":
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16 if torch.cuda.is_available() else torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def parse_image_root_overrides(values: Sequence[str]) -> dict[str, Path]:
    roots: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--image-root must be KEY=PATH")
        key, path = value.split("=", 1)
        roots[norm_condition(key)] = expand_path(path)
    return roots


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    if not isinstance(config, dict):
        raise ValueError("Config must be a JSON object")
    return config


def normalize_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Accept both the readable nested config and the legacy flat layout."""
    result = dict(config)
    model = config.get("model") if isinstance(config.get("model"), Mapping) else {}
    data = config.get("data") if isinstance(config.get("data"), Mapping) else {}
    aliases = {
        "model_id": model.get("id"),
        "model_family": model.get("family"),
        "dtype": model.get("dtype"),
        "device_map": model.get("device_map"),
        "attn_implementation": model.get("attn_implementation"),
        "trust_remote_code": model.get("trust_remote_code"),
        "num_heads": model.get("num_heads"),
        "input_data": data.get("input_data"),
        "image_roots": data.get("image_roots"),
    }
    for key, value in aliases.items():
        if key not in result and value is not None:
            result[key] = value
    return result


def literature_all_preset(
    model_id: str, num_heads: int | str = "auto", model_family: str = "auto"
) -> dict[str, Any]:
    """Self-contained preset used by run_literature_all.sh (no JSON required)."""
    top4 = ["22:4", "22:17", "22:16", "22:20"]
    return {
        "model_id": model_id,
        "model_family": normalize_model_family(model_family),
        "dtype": "bfloat16",
        "device_map": "auto",
        "attn_implementation": "sdpa",
        "trust_remote_code": True,
        "num_heads": num_heads if str(num_heads).lower() == "auto" else int(num_heads),
        "input_data": "INPUT_DATA_REQUIRED",
        "image_roots": {},
        "output_dir": "OUTPUT_DIR_REQUIRED",
        "seed": 17,
        "activation_cache_gb": 1.0,
        "prompt_keys": {"MC": "MC_1", "SA": "SA_1", "LA": "LA_1"},
        "experiments": [
            {
                "name": "01_permutation_baseline",
                "type": "option_permutation",
                "profiles": ["quick", "full"],
                "conditions": ["full"],
                "heads": [],
                "alpha": 0.0,
                "quick_limit": 30,
                "quick_permutations": 2,
                "full_permutations": 5,
            },
            {
                "name": "02_permutation_fixed_l22_top4",
                "type": "option_permutation",
                "profiles": ["quick", "full"],
                "conditions": ["full"],
                "heads": top4,
                "alpha": 0.2,
                "phase": "first_token",
                "position": "last",
                "quick_limit": 30,
                "quick_permutations": 2,
                "full_permutations": 5,
            },
            {
                "name": "03_l22_vhr_and_suppression",
                "type": "faithful_vhr",
                "profiles": ["quick", "full"],
                "conditions": ["full"],
                "layers": [22],
                "actions": ["vhr_reinforce", "vhd_suppress"],
                "quick_limit": 30,
                "quick_scale_factors": [1.1, 1.2],
                "full_scale_factors": [1.05, 1.1, 1.2, 1.5],
                "quick_suppression_factors": [0.5],
                "full_suppression_factors": [0.25, 0.5, 0.75],
            },
            {
                "name": "04_vcd_logit_contrast",
                "type": "contrastive_decoding",
                "profiles": ["quick", "full"],
                "conditions": ["full"],
                "priors": ["text_only", "gaussian_noise"],
                "noise_sigma": 0.2,
                "quick_limit": 50,
                "quick_alphas": [1.0],
                "full_alphas": [0.5, 1.0, 2.0],
                "quick_plausibility_betas": [0.1],
                "full_plausibility_betas": [0.0, 0.1],
            },
            {
                "name": "05_permutation_ensemble",
                "type": "permutation_ensemble",
                "profiles": ["quick", "full"],
                "conditions": ["full"],
                "quick_limit": 30,
                "quick_permutation_counts": [3],
                "full_permutation_counts": [3, 6],
            },
            {
                "name": "06_label_prior_calibration",
                "type": "label_bias_calibration",
                "profiles": ["quick", "full"],
                "conditions": ["full"],
                "quick_calibration_limit": 12,
                "full_calibration_limit": 50,
                "quick_calibration_permutations": 2,
                "full_calibration_permutations": 3,
                "quick_limit": 30,
                "quick_gammas": [1.0],
                "full_gammas": [0.5, 1.0, 1.5],
            },
            {
                "name": "07_l22_symbol_invariant_vhr",
                "type": "symbol_invariant_vhr",
                "profiles": ["quick", "full"],
                "conditions": ["full"],
                "layer": 22,
                "quick_limit": 20,
                "quick_permutations": 2,
                "full_permutations": 4,
                "quick_scale_factors": [1.1, 1.2],
                "full_scale_factors": [1.05, 1.1, 1.2, 1.5],
            },
            {
                "name": "08_global_layer_discovery",
                "type": "global_layer_discovery",
                "profiles": ["quick", "full"],
                "conditions": ["full"],
                "layers": "all",
                "selection_position": "last",
                "top_fraction": 0.25,
                "selected_head_count": 4,
                "minimum_layer_gap": 2,
                "quick_scan_limit": 30,
                "quick_scan_permutations": 1,
                "quick_candidate_layer_count": 3,
                "quick_causal_limit": 20,
                "quick_random_set_count": 1,
                "quick_scale_factors": [0.5, 1.1],
                "full_scan_permutations": 1,
                "full_candidate_layer_count": 5,
                "full_causal_limit": 120,
                "full_random_set_count": 2,
                "full_scale_factors": [0.5, 1.1],
            },
        ],
    }


def validate_config_only(config: Mapping[str, Any], profile: str) -> None:
    required = ["model_id", "input_data", "output_dir", "experiments"]
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Missing config keys: {missing}")
    if not isinstance(config["experiments"], list):
        raise ValueError("experiments must be a list")
    names: set[str] = set()
    supported_types = {
        "generation_scaling",
        "option_permutation",
        "patching_sweep",
        "visual_contrast_reinforcement",
        "source_control",
        "label_swap_control",
        "bridge",
        "dynamic_vhd",
        "faithful_vhr",
        "contrastive_decoding",
        "gated_label_correction",
        "detector_disambiguation",
        "semantic_gated_contrast",
        "mismatch_activation_export",
        "permutation_ensemble",
        "always_on_evidence_first",
        "label_bias_calibration",
        "symbol_invariant_vhr",
        "global_layer_discovery",
        "backpatch",
        "activation_export",
    }
    for exp in config["experiments"]:
        if not isinstance(exp, dict) or "name" not in exp or "type" not in exp:
            raise ValueError("Each experiment requires name and type")
        if exp["name"] in names:
            raise ValueError(f"Duplicate experiment name: {exp['name']}")
        names.add(str(exp["name"]))
        if exp["type"] not in supported_types:
            raise ValueError(
                f"Unsupported experiment type {exp['type']!r} in {exp['name']!r}"
            )
        profiles = exp.get("profiles", ["quick", "full"])
        if profile not in profiles:
            continue
    print(f"Config validation passed: {len(config['experiments'])} experiments")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Optional suite JSON configuration")
    parser.add_argument(
        "--preset", choices=("literature-all",), default=None,
        help="Use an embedded experiment preset instead of a JSON config",
    )
    parser.add_argument(
        "--model-id", default=None,
        help="Override the config model id, or set the model for an embedded preset",
    )
    parser.add_argument(
        "--model-family",
        choices=tuple(sorted(MODEL_FAMILIES)),
        default=None,
        help="Model architecture for a preset (default: infer from model id)",
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=None,
        help="Override query-head count; default is model auto-detection",
    )
    parser.add_argument("--profile", choices=("quick", "full"), default="quick")
    parser.add_argument("--input-data", default=None, help="Override config input_data")
    parser.add_argument(
        "--image-root",
        action="append",
        default=[],
        metavar="KEY=PATH",
        help="Override/add an image root; repeatable",
    )
    parser.add_argument("--output-dir", default=None, help="Override config output_dir")
    parser.add_argument(
        "--only",
        default=None,
        help="Comma-separated experiment names to run",
    )
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=None,
        help="Override shuffle_seed for mismatch activation export experiments",
    )
    parser.add_argument(
        "--force-synthetic-shuffled",
        action="store_true",
        help="Ignore manifest-provided shuffled rows and synthesize a new pairing",
    )
    parser.add_argument(
        "--reuse-aligned-export",
        default=None,
        help="Reuse Full/Blurred/Text-only tensors and score only the new Shuffled input",
    )
    parser.add_argument("--validate-config", action="store_true")
    args = parser.parse_args()

    if bool(args.config) == bool(args.preset):
        raise ValueError("Specify exactly one of --config or --preset")
    if args.config:
        config_path = expand_path(args.config)
        config = normalize_config(load_config(config_path))
        config_dir = config_path.parent
    else:
        config = literature_all_preset(
            args.model_id or "NCSOFT/VARCO-VISION-2.0-14B",
            args.num_heads if args.num_heads is not None else "auto",
            args.model_family or "auto",
        )
        config_dir = Path.cwd()
    if args.model_id:
        config["model_id"] = args.model_id
    if args.model_family:
        config["model_family"] = args.model_family
    if args.num_heads is not None:
        config["num_heads"] = args.num_heads
    for experiment in config.get("experiments", []):
        if experiment.get("type") != "mismatch_activation_export":
            continue
        if args.shuffle_seed is not None:
            experiment["shuffle_seed"] = args.shuffle_seed
        if args.force_synthetic_shuffled:
            experiment["prefer_manifest_shuffled"] = False
        if args.reuse_aligned_export:
            experiment["reuse_aligned_export"] = args.reuse_aligned_export
    validate_config_only(config, args.profile)
    input_data = expand_path(args.input_data or config["input_data"], config_dir)
    output_dir = expand_path(args.output_dir or config["output_dir"], config_dir)
    image_roots = {
        norm_condition(key): expand_path(value, config_dir)
        for key, value in config.get("image_roots", {}).items()
    }
    image_roots.update(parse_image_root_overrides(args.image_root))
    if not image_roots:
        raise ValueError("At least one image root is required")
    if args.validate_config:
        print(f"input_data={input_data} (exists={input_data.is_file()})")
        for key, root in sorted(image_roots.items()):
            print(f"image_root[{key}]={root} (exists={root.is_dir()})")
        print(f"output_dir={output_dir}")
        if not input_data.is_file():
            raise FileNotFoundError(f"Input data does not exist: {input_data}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_input(input_data)
    unsupported = sorted(
        {row["question_form"] for row in rows} - {"MC", "SA", "LA"}
    )
    if unsupported:
        raise ValueError(f"Unsupported question forms: {unsupported}")

    dtype = resolve_dtype(str(config.get("dtype", "auto")))
    model_id = str(config["model_id"])
    model_family = infer_model_family(
        model_id, args.model_family or config.get("model_family", "auto")
    )
    attn_implementation = str(config.get("attn_implementation", "sdpa"))
    print(f"input_data={input_data}")
    print(f"rows={len(rows)}")
    print(f"output_dir={output_dir}")
    print(f"model={model_id} (family={model_family})")
    print(f"dtype={dtype}, attention={attn_implementation}")
    image_store = ImageStore(image_roots)
    print(f"image_roots={image_roots}")
    print(f"image_counts={image_store.counts()}")

    model = load_vlm_model(
        model_id,
        model_family,
        dtype=dtype,
        attn_implementation=attn_implementation,
        device_map=config.get("device_map", "auto"),
        trust_remote_code=bool(config.get("trust_remote_code", False)),
    )
    processor = AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=bool(config.get("trust_remote_code", False)),
    )
    layers, decoder_path = decoder_layers(model)
    configured_num_heads = config.get("num_heads", "auto")
    if configured_num_heads in (None, "", "auto", 0, "0"):
        num_heads = infer_num_attention_heads(model, layers)
        head_source = "auto"
    else:
        num_heads = int(configured_num_heads)
        head_source = "config"
    print(
        f"model loaded: decoder={decoder_path}, layers={len(layers)}, "
        f"query_heads={num_heads} ({head_source})"
    )

    prompt_keys = {
        "MC": str(config.get("prompt_keys", {}).get("MC", "MC_1")),
        "SA": str(config.get("prompt_keys", {}).get("SA", "SA_1")),
        "LA": str(config.get("prompt_keys", {}).get("LA", "LA_1")),
    }
    runner = ExperimentRunner(
        model=model,
        processor=processor,
        rows=rows,
        image_store=image_store,
        output_dir=output_dir,
        prompt_keys=prompt_keys,
        num_heads=num_heads,
        dtype=dtype,
        max_new_tokens=config.get("max_new_tokens"),
        seed=int(config.get("seed", 17)),
        activation_cache_gb=float(config.get("activation_cache_gb", 1.0)),
    )
    print(f"image_token_ids={sorted(runner.image_token_ids)}")

    only = set(args.only.split(",")) if args.only else None
    experiments = []
    for exp in config["experiments"]:
        profiles = exp.get("profiles", ["quick", "full"])
        if args.profile not in profiles:
            continue
        if only is not None and exp["name"] not in only:
            continue
        experiments.append(exp)
    if not experiments:
        raise ValueError("No experiments selected")
    l22_specific = {
        "02_permutation_fixed_l22_top4",
        "03_l22_vhr_and_suppression",
        "07_l22_symbol_invariant_vhr",
    }
    selected_l22 = sorted(
        str(exp["name"]) for exp in experiments if str(exp["name"]) in l22_specific
    )
    if model_family != "llava_onevision" and selected_l22:
        print(
            "WARNING: fixed VARCO L22/head findings are model-specific and should not "
            f"be interpreted as Qwen discoveries: {selected_l22}. Run "
            "08_global_layer_discovery first and create a Qwen-specific config."
        )

    completed: list[str] = []
    try:
        for exp in experiments:
            runner.run_experiment(exp, args.profile)
            completed.append(str(exp["name"]))
            runner.clear_activation_cache()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    except torch.cuda.OutOfMemoryError as exc:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise RuntimeError(
            "CUDA OOM. Use profile=quick, lower per-experiment limits, or run "
            "experiments separately with --only."
        ) from exc
    finally:
        manifest = {
            "profile": args.profile,
            "input_data": str(input_data),
            "model_id": model_id,
            "model_family": model_family,
            "decoder_path": decoder_path,
            "num_layers": len(layers),
            "num_attention_heads": num_heads,
            "completed_experiments": completed,
            "selected_experiments": [str(exp["name"]) for exp in experiments],
        }
        with (output_dir / "run_manifest.json").open("w", encoding="utf-8") as stream:
            json.dump(manifest, stream, ensure_ascii=False, indent=2)
    print(f"Completed: {completed}")


if __name__ == "__main__":
    main()
