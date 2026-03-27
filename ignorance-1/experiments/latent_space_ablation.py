#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.losses.alignment import ignorance_penalty, normalized_mse_loss, paired_alignment_loss
from src.losses.sigreg import collapse_detected, isotropic_score, sigreg_loss
from src.models.jepa import JEPAModel
from src.training.phase4 import _proxy_config
from src.utils.data import sample_ood_queries, set_seed
from src.utils.retrieval import VectorIndex


RESULTS_TSV = ROOT / "artifacts" / "results.tsv"
RUNS_DIR = ROOT / "artifacts" / "runs"

KNOWN_QUERIES = [
    "Sort a numeric list ascending and return the result.",
    "How can I order an array of integers from smallest to largest?",
    "Read each line from a text file and strip whitespace.",
    "I want to load a file and trim every line in it.",
    "Parse a json string into a javascript object.",
    "Convert this JSON text into a JS variable.",
    "Fetch JSON from an HTTP endpoint in python.",
    "Group dictionaries by a key in python.",
]

OOD_QUERIES = [
    "What is the weather in Tokyo today?",
    "Who was the first president of the United States?",
    "Explain photosynthesis in one sentence.",
    "Why is the sky blue?",
    "Name three planets in the solar system.",
    "How do bees make honey?",
]

PARAPHRASE_PAIRS = [
    ("Sort a numeric list ascending and return the result.", "How can I order an array of integers from smallest to largest?"),
    ("Read each line from a text file and strip whitespace.", "I want to load a file and trim every line in it."),
    ("Parse a json string into a javascript object.", "Convert this JSON text into a JS variable."),
]


@dataclass(frozen=True)
class TaskFamily:
    name: str
    prompts: tuple[str, ...]
    codes: tuple[str, ...]
    hard_negatives: tuple[str, ...]


@dataclass(frozen=True)
class LatentExperiment:
    name: str
    seed: int
    patch_size: int
    tokenizer_mode: str
    annotation_prob: float
    use_hard_negatives: bool
    temperature: float = 0.07
    embedding_weight: float = 0.65
    mse_weight: float = 0.15
    sigreg_weight: float = 0.05
    ood_weight: float = 0.2
    vicreg_weight: float = 0.0
    use_extended_families: bool = False
    paraphrase_weight: float = 0.0
    use_stronger_family_negatives: bool = False
    family_holdout_eval: bool = False
    prompt_augmentation: bool = False
    lr: float = 5e-4
    warmup_steps: int = 0
    ramp_regularizers: bool = False
    use_mined_negatives: bool = False
    use_virtual_nulls: bool = False
    use_multisource_families: bool = False


TASK_FAMILIES: tuple[TaskFamily, ...] = (
    TaskFamily(
        name="sort_numbers",
        prompts=(
            "Sort a numeric list ascending and return the result.",
            "How can I order an array of integers from smallest to largest?",
            "Return the numbers in increasing order without mutating the input.",
        ),
        codes=(
            "def solve(values):\n    return sorted(values)\n",
            "ordered = sorted(numbers)\n",
        ),
        hard_negatives=(
            "def solve(values):\n    return sorted(values, reverse=True)\n",
            "total = sum(values)\n",
        ),
    ),
    TaskFamily(
        name="read_trim_file",
        prompts=(
            "Read each line from a text file and strip whitespace.",
            "I want to load a file and trim every line in it.",
            "Open a file and collect trimmed rows.",
        ),
        codes=(
            "with open(path) as handle:\n    lines = [line.strip() for line in handle]\n",
            "rows = []\nwith open(file_path) as handle:\n    rows = [line.strip() for line in handle]\n",
        ),
        hard_negatives=(
            "with open(path) as handle:\n    blob = handle.read()\n",
            "with open(path, 'w') as handle:\n    handle.write(text)\n",
        ),
    ),
    TaskFamily(
        name="parse_json_js",
        prompts=(
            "Parse a json string into a javascript object.",
            "Convert this JSON text into a JS variable.",
            "Deserialize JSON text into a JavaScript value.",
        ),
        codes=(
            "const parsed = JSON.parse(payload);\n",
            "const value = JSON.parse(body);\n",
        ),
        hard_negatives=(
            "const text = JSON.stringify(payload);\n",
            "const parsed = Number(payload);\n",
        ),
    ),
    TaskFamily(
        name="fetch_json_python",
        prompts=(
            "Fetch JSON from an HTTP endpoint in python.",
            "Make a GET request and parse the response body as JSON.",
            "Call an API and deserialize the JSON payload in python.",
        ),
        codes=(
            "response = requests.get(url)\ndata = response.json()\n",
            "payload = requests.get(endpoint).json()\n",
        ),
        hard_negatives=(
            "response = requests.post(url, json=payload)\n",
            "data = json.loads(text)\n",
        ),
    ),
    TaskFamily(
        name="group_dicts_python",
        prompts=(
            "Group dictionaries by a key in python.",
            "Bucket rows by one field and collect them in lists.",
            "Build a mapping from key values to matching records.",
        ),
        codes=(
            "groups = {}\nfor row in rows:\n    groups.setdefault(row.get('kind'), []).append(row)\n",
            "buckets = {}\nfor item in items:\n    buckets.setdefault(item['type'], []).append(item)\n",
        ),
        hard_negatives=(
            "ordered = sorted(rows, key=lambda row: row.get('kind'))\n",
            "counts = {}\nfor row in rows:\n    counts[row.get('kind')] = counts.get(row.get('kind'), 0) + 1\n",
        ),
    ),
)

EXTENDED_TASK_FAMILIES: tuple[TaskFamily, ...] = TASK_FAMILIES + (
    TaskFamily(
        name="filter_active_rows",
        prompts=(
            "Filter a list of dictionaries to only active entries.",
            "Keep only rows whose active flag is true.",
            "Return items marked as active.",
        ),
        codes=(
            "active = [row for row in rows if row.get('active')]\n",
            "filtered = [item for item in items if item['active']]\n",
        ),
        hard_negatives=(
            "inactive = [row for row in rows if not row.get('active')]\n",
            "ordered = sorted(rows, key=lambda row: row.get('active'))\n",
        ),
    ),
    TaskFamily(
        name="startswith_check",
        prompts=(
            "Check whether a string starts with a prefix in JavaScript.",
            "Test if text begins with the requested token.",
            "Return whether an input string has a given prefix.",
        ),
        codes=(
            "const hasPrefix = text.startsWith(prefix);\n",
            "const matches = value.startsWith(token);\n",
        ),
        hard_negatives=(
            "const hasSuffix = text.endsWith(prefix);\n",
            "const matches = value.includes(token);\n",
        ),
    ),
    TaskFamily(
        name="flatten_nested_lists",
        prompts=(
            "Flatten a nested python list into a single list.",
            "Collapse a list of lists into one flat sequence.",
            "Take nested arrays and return all values in one list.",
        ),
        codes=(
            "flat = [value for group in nested for value in group]\n",
            "items = [item for chunk in chunks for item in chunk]\n",
        ),
        hard_negatives=(
            "pairs = list(zip(*nested))\n",
            "nested = [list(group) for group in flat]\n",
        ),
    ),
)

MULTISOURCE_TASK_FAMILIES: tuple[TaskFamily, ...] = EXTENDED_TASK_FAMILIES + (
    TaskFamily(
        name="sql_filter_rows",
        prompts=(
            "Filter SQL rows where status is active.",
            "Write a SQL query that keeps only active records.",
            "Return active users from a table with SQL.",
        ),
        codes=(
            "SELECT * FROM users WHERE status = 'active';\n",
            "SELECT id, name FROM accounts WHERE active = TRUE;\n",
        ),
        hard_negatives=(
            "SELECT * FROM users ORDER BY status;\n",
            "UPDATE users SET status = 'active';\n",
        ),
    ),
    TaskFamily(
        name="bash_find_files",
        prompts=(
            "Find all log files in a directory with bash.",
            "List every .log file recursively from the shell.",
            "Use bash to search for log files under a path.",
        ),
        codes=(
            "find \"$root\" -type f -name '*.log'\n",
            "find . -type f | grep '\\.log$'\n",
        ),
        hard_negatives=(
            "find \"$root\" -type d -name '*.log'\n",
            "ls *.log\n",
        ),
    ),
    TaskFamily(
        name="html_link_extract",
        prompts=(
            "Extract href attributes from HTML anchors in Python.",
            "Parse HTML and collect link targets.",
            "Return all anchor href values from markup.",
        ),
        codes=(
            "links = [tag.get('href') for tag in soup.find_all('a')]\n",
            "hrefs = [anchor['href'] for anchor in document.select('a[href]')]\n",
        ),
        hard_negatives=(
            "titles = [tag.text for tag in soup.find_all('a')]\n",
            "images = [img.get('src') for img in soup.find_all('img')]\n",
        ),
    ),
)

VIRTUAL_NULL_SNIPPETS: tuple[str, ...] = (
    "def unsupported_task(*args, **kwargs):\n    raise NotImplementedError('unknown task')\n",
    "result = None\n# intentionally unrelated placeholder\n",
    "function unknownIntent() {\n  return null;\n}\n",
    "SELECT NULL AS unsupported_result;\n",
    "echo 'no matching capability'\n",
)


def variance_covariance_loss(z: torch.Tensor, variance_floor: float = 1.0) -> torch.Tensor:
    if z.ndim != 2 or z.shape[0] < 2:
        return z.new_tensor(0.0)

    z = z.float()
    centered = z - z.mean(dim=0, keepdim=True)
    std = torch.sqrt(centered.var(dim=0, unbiased=False) + 1e-4)
    var_loss = F.relu(variance_floor - std).mean()

    cov = centered.T @ centered / max(centered.shape[0] - 1, 1)
    off_diag_mask = ~torch.eye(cov.shape[0], dtype=torch.bool, device=cov.device)
    cov_loss = cov[off_diag_mask].pow(2).mean()
    return (var_loss + cov_loss).to(z.dtype)


def ensure_results_header() -> None:
    RESULTS_TSV.parent.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    if RESULTS_TSV.exists():
        return
    RESULTS_TSV.write_text(
        "run_id\tstatus\tdevice\tphase_score\tphase1_pass\tphase2_pass\tphase3_pass\tphase4_pass\t"
        "without_retrieval\twith_retrieval\tretrieval_gap\tplanning_success\tscaling_improvement\tdescription\n"
    )


def parse_results_table() -> list[dict[str, str]]:
    if not RESULTS_TSV.exists():
        return []
    rows: list[dict[str, str]] = []
    with RESULTS_TSV.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            cleaned = {(key or "").strip(): (value or "").strip() for key, value in row.items()}
            if cleaned.get("run_id"):
                rows.append(cleaned)
    return rows


def next_run_index(history_rows: list[dict[str, str]]) -> int:
    max_run = 0
    for row in history_rows:
        prefix = row.get("run_id", "").split("-", 1)[0]
        if prefix.isdigit():
            max_run = max(max_run, int(prefix))
    return max_run + 1


def slugify(name: str) -> str:
    cleaned = []
    for char in name.lower():
        cleaned.append(char if char.isalnum() else "-")
    slug = "".join(cleaned)
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug.strip("-")


def append_result(run_id: str, device: str, metrics: dict[str, float | bool], description: str) -> None:
    phase_score = (
        max(float(metrics["dense_gap"]), 0.0)
        + max(float(metrics["known_paraphrase_similarity"]), 0.0)
        + max(float(metrics["code_isotropy"]), 0.0)
        + float(bool(metrics["dense_gap_pass"]))
        + float(bool(metrics["cohesion_pass"]))
        + float(bool(metrics["isotropy_pass"]))
        + float(bool(not metrics["collapse_detected"]))
    )
    row = (
        f"{run_id}\tok\t{device}\t{phase_score:.3f}\t"
        f"{int(bool(metrics['dense_gap_pass']))}\t"
        f"{int(bool(metrics['cohesion_pass']))}\t"
        f"{int(bool(metrics['isotropy_pass']))}\t"
        f"{int(bool(not metrics['collapse_detected']))}\t"
        f"{float(metrics['dense_gap']):.3f}\t"
        f"{float(metrics['known_paraphrase_similarity']):.3f}\t"
        f"{float(metrics['ood_gap']):.3f}\t"
        f"{float(metrics['code_isotropy']):.3f}\t"
        f"{float(metrics['code_avg_similarity']):.3f}\t"
        f"{description}\n"
    )
    with RESULTS_TSV.open("a") as handle:
        handle.write(row)


class QueueBuffer:
    def __init__(self, size: int, dim: int, device: str):
        self.buffer = torch.empty(0, dim, device=device)
        self.size = size

    def push(self, x: torch.Tensor) -> None:
        self.buffer = torch.cat([x.detach(), self.buffer], dim=0)[: self.size]

    def get(self) -> torch.Tensor:
        return self.buffer


class FlexibleTokenizer:
    def __init__(self, vocab_size: int = 4096, mode: str = "word_hash"):
        self.vocab_size = vocab_size
        self.mode = mode
        self.pad_id = 0

    def _hash_piece(self, piece: str) -> int:
        return (int(hashlib.md5(piece.encode()).hexdigest()[:8], 16) % (self.vocab_size - 1)) + 1

    def encode(self, text: str, seq_len: int) -> list[int]:
        lowered = text.lower()
        if self.mode == "byte_hash":
            tokens = [self._hash_piece(f"b:{byte}") for byte in lowered.encode("utf-8")]
        elif self.mode == "char_ngram":
            collapsed = re.sub(r"\s+", " ", lowered).strip()
            padded = f" {collapsed} " if collapsed else " "
            ngrams = [padded[idx : idx + 3] for idx in range(max(len(padded) - 2, 1))]
            if not ngrams:
                ngrams = [padded]
            tokens = [self._hash_piece(f"c:{gram}") for gram in ngrams]
        else:
            clean_text = re.sub(r"[^\w\s]", " ", lowered)
            pieces = [piece for piece in clean_text.split() if piece]
            tokens = [self._hash_piece(f"w:{piece}") for piece in pieces]
        if len(tokens) >= seq_len:
            return tokens[:seq_len]
        return tokens + [self.pad_id] * (seq_len - len(tokens))

    def batch_encode(self, texts: list[str], seq_len: int, device: str) -> torch.Tensor:
        return torch.tensor([self.encode(text, seq_len) for text in texts], dtype=torch.long, device=device)


def annotate_code(prompt: str, code: str, annotation_prob: float) -> str:
    if random.random() > annotation_prob:
        return code
    return f"# task: {prompt.strip().rstrip('.')}\n{code}"


def augmented_prompt_pool(family: TaskFamily) -> list[str]:
    prompts = list(family.prompts)
    augmented: list[str] = []
    for prompt in prompts:
        core = prompt.strip().rstrip(".")
        augmented.extend(
            [
                prompt,
                f"How do I {core.lower()}?",
                f"Need code to {core.lower()}.",
                f"Write a short snippet that can {core.lower()}.",
                f"Implement this task: {core}.",
                f"Please help me {core.lower()}.",
            ]
        )
    deduped: list[str] = []
    seen: set[str] = set()
    for prompt in augmented:
        if prompt not in seen:
            deduped.append(prompt)
            seen.add(prompt)
    return deduped


def build_training_pairs(
    repeats: int,
    annotation_prob: float,
    families: tuple[TaskFamily, ...],
    holdout_last_prompt: bool = False,
    prompt_augmentation: bool = False,
) -> list[tuple[str, str, str, list[str]]]:
    pairs: list[tuple[str, str, str, list[str]]] = []
    for _ in range(repeats):
        for family in families:
            if prompt_augmentation:
                base_prompts = TaskFamily(
                    name=family.name,
                    prompts=tuple(family.prompts[:-1]) if holdout_last_prompt and len(family.prompts) > 1 else family.prompts,
                    codes=family.codes,
                    hard_negatives=family.hard_negatives,
                )
                prompt_pool = tuple(augmented_prompt_pool(base_prompts))
            else:
                prompt_pool = family.prompts[:-1] if holdout_last_prompt and len(family.prompts) > 1 else family.prompts
            prompt = random.choice(prompt_pool)
            code = random.choice(family.codes)
            pairs.append((prompt, annotate_code(prompt, code, annotation_prob), family.name, list(family.hard_negatives)))
    random.shuffle(pairs)
    return pairs


def build_index_snippets(families: tuple[TaskFamily, ...]) -> list[str]:
    snippets = []
    for family in families:
        snippets.extend(family.codes)
    return sorted(set(snippets))


def build_negative_candidates(
    batch_pairs: list[tuple[str, str, str, list[str]]],
    family_lookup: dict[str, TaskFamily],
    include_family_candidates: bool,
    use_virtual_nulls: bool,
) -> list[str]:
    candidates: list[str] = []
    if include_family_candidates:
        batch_family_names = {family_name for _, _, family_name, _ in batch_pairs}
        for family_name, family in family_lookup.items():
            if family_name in batch_family_names:
                continue
            candidates.extend(family.codes)
            candidates.extend(family.hard_negatives)
    if use_virtual_nulls:
        candidates.extend(VIRTUAL_NULL_SNIPPETS)
    return sorted(set(candidates))


def encode_mined_negative_pool(
    tokenizer: FlexibleTokenizer,
    config,
    device: str,
    model: JEPAModel,
    candidate_texts: list[str],
    query_latents: torch.Tensor,
    top_k: int = 8,
) -> torch.Tensor:
    if not candidate_texts:
        return torch.empty(0, config.embed_dim, device=device)

    candidate_ids = tokenizer.batch_encode(candidate_texts, config.max_seq_len, device)
    with torch.no_grad():
        candidate_latents = model.encode(candidate_ids)
        query_norm = F.normalize(query_latents.detach().float(), dim=-1)
        candidate_norm = F.normalize(candidate_latents.float(), dim=-1)
        similarities = query_norm @ candidate_norm.T
        hardest = similarities.max(dim=0).values
        limit = min(top_k, candidate_latents.shape[0])
        top_indices = torch.topk(hardest, k=limit).indices
    return candidate_latents[top_indices].detach()


def build_holdout_queries(families: tuple[TaskFamily, ...]) -> list[str]:
    queries: list[str] = []
    for family in families:
        if family.prompts:
            queries.append(family.prompts[-1])
    return queries


def avg(rows: list[tuple[str, float]], wanted: str) -> float:
    values = [value for label, value in rows if label == wanted]
    return sum(values) / max(len(values), 1)


def candidate_queue() -> list[LatentExperiment]:
    recipes = [
        LatentExperiment("latent baseline wordhash patch32", 42, 32, "word_hash", 1.0, False),
        LatentExperiment("latent patch16 wordhash", 42, 16, "word_hash", 1.0, False),
        LatentExperiment("latent bytehash patch32", 42, 32, "byte_hash", 1.0, False),
        LatentExperiment("latent bytehash patch16", 42, 16, "byte_hash", 1.0, False),
        LatentExperiment("latent bytehash patch16 confirm seed43", 43, 16, "byte_hash", 1.0, False),
        LatentExperiment("latent bytehash patch16 annotation half", 42, 16, "byte_hash", 0.5, False),
        LatentExperiment("latent bytehash patch16 annotation half confirm seed43", 43, 16, "byte_hash", 0.5, False),
        LatentExperiment("latent bytehash patch16 hard negatives", 42, 16, "byte_hash", 0.5, True),
        LatentExperiment("latent hard negatives confirm seed43", 43, 16, "byte_hash", 0.5, True),
        LatentExperiment("latent bytehash patch16 lowtemp align", 42, 16, "byte_hash", 1.0, False, temperature=0.03, embedding_weight=0.9, mse_weight=0.05),
        LatentExperiment("latent bytehash patch16 lowtemp align confirm seed43", 43, 16, "byte_hash", 1.0, False, temperature=0.03, embedding_weight=0.9, mse_weight=0.05),
        LatentExperiment("latent bytehash patch16 strong sigreg", 42, 16, "byte_hash", 1.0, False, sigreg_weight=0.15),
        LatentExperiment("latent bytehash patch16 strong sigreg confirm seed43", 43, 16, "byte_hash", 1.0, False, sigreg_weight=0.15),
        LatentExperiment("latent bytehash patch16 lowtemp plus sigreg", 42, 16, "byte_hash", 1.0, False, temperature=0.03, embedding_weight=0.9, mse_weight=0.05, sigreg_weight=0.15),
        LatentExperiment("latent bytehash patch16 lowtemp plus sigreg confirm seed43", 43, 16, "byte_hash", 1.0, False, temperature=0.03, embedding_weight=0.9, mse_weight=0.05, sigreg_weight=0.15),
        LatentExperiment("latent bytehash patch16 strong ood", 42, 16, "byte_hash", 1.0, False, ood_weight=0.4),
        LatentExperiment("latent bytehash patch16 strong ood confirm seed43", 43, 16, "byte_hash", 1.0, False, ood_weight=0.4),
        LatentExperiment("latent extended vicreg", 42, 16, "byte_hash", 1.0, True, ood_weight=0.4, vicreg_weight=0.1, use_extended_families=True),
        LatentExperiment("latent extended vicreg confirm seed43", 43, 16, "byte_hash", 1.0, True, ood_weight=0.4, vicreg_weight=0.1, use_extended_families=True),
        LatentExperiment("latent extended vicreg no annotation", 42, 16, "byte_hash", 0.5, True, ood_weight=0.4, vicreg_weight=0.1, use_extended_families=True),
        LatentExperiment("latent extended vicreg no annotation confirm seed43", 43, 16, "byte_hash", 0.5, True, ood_weight=0.4, vicreg_weight=0.1, use_extended_families=True),
        LatentExperiment("latent extended vicreg lowtemp", 42, 16, "byte_hash", 1.0, True, temperature=0.03, embedding_weight=0.9, mse_weight=0.05, ood_weight=0.4, vicreg_weight=0.1, use_extended_families=True),
        LatentExperiment("latent extended vicreg lowtemp confirm seed43", 43, 16, "byte_hash", 1.0, True, temperature=0.03, embedding_weight=0.9, mse_weight=0.05, ood_weight=0.4, vicreg_weight=0.1, use_extended_families=True),
        LatentExperiment("latent extended tuned balanced", 42, 16, "byte_hash", 1.0, True, ood_weight=0.3, vicreg_weight=0.2, use_extended_families=True),
        LatentExperiment("latent extended tuned balanced confirm seed43", 43, 16, "byte_hash", 1.0, True, ood_weight=0.3, vicreg_weight=0.2, use_extended_families=True),
        LatentExperiment("latent extended tuned balanced confirm seed44", 44, 16, "byte_hash", 1.0, True, ood_weight=0.3, vicreg_weight=0.2, use_extended_families=True),
        LatentExperiment("latent extended tuned strong vicreg", 42, 16, "byte_hash", 1.0, True, ood_weight=0.25, vicreg_weight=0.25, use_extended_families=True),
        LatentExperiment("latent extended tuned strong vicreg confirm seed43", 43, 16, "byte_hash", 1.0, True, ood_weight=0.25, vicreg_weight=0.25, use_extended_families=True),
        LatentExperiment("latent extended tuned strong vicreg confirm seed44", 44, 16, "byte_hash", 1.0, True, ood_weight=0.25, vicreg_weight=0.25, use_extended_families=True),
        LatentExperiment("latent extended tuned strong vicreg confirm seed45", 45, 16, "byte_hash", 1.0, True, ood_weight=0.25, vicreg_weight=0.25, use_extended_families=True),
        LatentExperiment("latent extended tuned strong vicreg confirm seed46", 46, 16, "byte_hash", 1.0, True, ood_weight=0.25, vicreg_weight=0.25, use_extended_families=True),
        LatentExperiment("latent extended tuned strong vicreg annot75", 42, 16, "byte_hash", 0.75, True, ood_weight=0.25, vicreg_weight=0.25, use_extended_families=True),
        LatentExperiment("latent extended tuned strong vicreg annot75 confirm seed43", 43, 16, "byte_hash", 0.75, True, ood_weight=0.25, vicreg_weight=0.25, use_extended_families=True),
        LatentExperiment("latent extended tuned strong vicreg annot75 confirm seed44", 44, 16, "byte_hash", 0.75, True, ood_weight=0.25, vicreg_weight=0.25, use_extended_families=True),
        LatentExperiment("latent strong vicreg paraphrase", 42, 16, "byte_hash", 1.0, True, ood_weight=0.25, vicreg_weight=0.25, use_extended_families=True, paraphrase_weight=0.2),
        LatentExperiment("latent strong vicreg paraphrase confirm seed43", 43, 16, "byte_hash", 1.0, True, ood_weight=0.25, vicreg_weight=0.25, use_extended_families=True, paraphrase_weight=0.2),
        LatentExperiment("latent strong vicreg paraphrase confirm seed44", 44, 16, "byte_hash", 1.0, True, ood_weight=0.25, vicreg_weight=0.25, use_extended_families=True, paraphrase_weight=0.2),
        LatentExperiment("latent strong vicreg annot75 paraphrase", 42, 16, "byte_hash", 0.75, True, ood_weight=0.25, vicreg_weight=0.25, use_extended_families=True, paraphrase_weight=0.2),
        LatentExperiment("latent strong vicreg annot75 paraphrase confirm seed43", 43, 16, "byte_hash", 0.75, True, ood_weight=0.25, vicreg_weight=0.25, use_extended_families=True, paraphrase_weight=0.2),
        LatentExperiment("latent strong vicreg annot75 paraphrase confirm seed44", 44, 16, "byte_hash", 0.75, True, ood_weight=0.25, vicreg_weight=0.25, use_extended_families=True, paraphrase_weight=0.2),
        LatentExperiment("latent next charngram patch16", 42, 16, "char_ngram", 1.0, True, ood_weight=0.25, vicreg_weight=0.25, use_extended_families=True),
        LatentExperiment("latent next charngram patch8", 42, 8, "char_ngram", 1.0, True, ood_weight=0.25, vicreg_weight=0.25, use_extended_families=True),
        LatentExperiment("latent next stronger negatives", 42, 16, "byte_hash", 1.0, True, ood_weight=0.25, vicreg_weight=0.25, use_extended_families=True, use_stronger_family_negatives=True),
        LatentExperiment("latent next charngram stronger negatives", 42, 16, "char_ngram", 1.0, True, ood_weight=0.25, vicreg_weight=0.25, use_extended_families=True, use_stronger_family_negatives=True),
        LatentExperiment("latent next charngram stronger negatives annot75", 42, 16, "char_ngram", 0.75, True, ood_weight=0.25, vicreg_weight=0.25, use_extended_families=True, use_stronger_family_negatives=True),
        LatentExperiment("latent next charngram patch16 holdout", 42, 16, "char_ngram", 1.0, True, ood_weight=0.25, vicreg_weight=0.25, use_extended_families=True, family_holdout_eval=True),
        LatentExperiment("latent next charngram patch16 holdout confirm seed43", 43, 16, "char_ngram", 1.0, True, ood_weight=0.25, vicreg_weight=0.25, use_extended_families=True, family_holdout_eval=True),
        LatentExperiment("latent next charngram patch16 holdout confirm seed44", 44, 16, "char_ngram", 1.0, True, ood_weight=0.25, vicreg_weight=0.25, use_extended_families=True, family_holdout_eval=True),
        LatentExperiment("latent next dataaug charngram holdout", 42, 16, "char_ngram", 1.0, True, ood_weight=0.25, vicreg_weight=0.25, use_extended_families=True, family_holdout_eval=True, prompt_augmentation=True),
        LatentExperiment("latent next dataaug charngram holdout confirm seed43", 43, 16, "char_ngram", 1.0, True, ood_weight=0.25, vicreg_weight=0.25, use_extended_families=True, family_holdout_eval=True, prompt_augmentation=True),
        LatentExperiment("latent next dataaug charngram holdout confirm seed44", 44, 16, "char_ngram", 1.0, True, ood_weight=0.25, vicreg_weight=0.25, use_extended_families=True, family_holdout_eval=True, prompt_augmentation=True),
        LatentExperiment("latent next dataaug charngram annot75 holdout", 42, 16, "char_ngram", 0.75, True, ood_weight=0.25, vicreg_weight=0.25, use_extended_families=True, family_holdout_eval=True, prompt_augmentation=True),
        LatentExperiment("latent next dataaug charngram annot75 holdout confirm seed43", 43, 16, "char_ngram", 0.75, True, ood_weight=0.25, vicreg_weight=0.25, use_extended_families=True, family_holdout_eval=True, prompt_augmentation=True),
        LatentExperiment("latent next dataaug charngram annot75 holdout confirm seed44", 44, 16, "char_ngram", 0.75, True, ood_weight=0.25, vicreg_weight=0.25, use_extended_families=True, family_holdout_eval=True, prompt_augmentation=True),
        LatentExperiment("latent next dataaug paraphrase charngram holdout", 42, 16, "char_ngram", 1.0, True, ood_weight=0.25, vicreg_weight=0.25, use_extended_families=True, family_holdout_eval=True, prompt_augmentation=True, paraphrase_weight=0.2),
        LatentExperiment("latent next dataaug paraphrase charngram holdout confirm seed43", 43, 16, "char_ngram", 1.0, True, ood_weight=0.25, vicreg_weight=0.25, use_extended_families=True, family_holdout_eval=True, prompt_augmentation=True, paraphrase_weight=0.2),
        LatentExperiment("latent next dataaug paraphrase charngram holdout confirm seed44", 44, 16, "char_ngram", 1.0, True, ood_weight=0.25, vicreg_weight=0.25, use_extended_families=True, family_holdout_eval=True, prompt_augmentation=True, paraphrase_weight=0.2),
        LatentExperiment("latent next dataaug paraphrase charngram annot75 holdout", 42, 16, "char_ngram", 0.75, True, ood_weight=0.25, vicreg_weight=0.25, use_extended_families=True, family_holdout_eval=True, prompt_augmentation=True, paraphrase_weight=0.2),
        LatentExperiment("latent next dataaug paraphrase charngram annot75 holdout confirm seed43", 43, 16, "char_ngram", 0.75, True, ood_weight=0.25, vicreg_weight=0.25, use_extended_families=True, family_holdout_eval=True, prompt_augmentation=True, paraphrase_weight=0.2),
        LatentExperiment("latent next dataaug paraphrase charngram annot75 holdout confirm seed44", 44, 16, "char_ngram", 0.75, True, ood_weight=0.25, vicreg_weight=0.25, use_extended_families=True, family_holdout_eval=True, prompt_augmentation=True, paraphrase_weight=0.2),
        LatentExperiment("latent next dataaug paraphrase charngram annot67 holdout", 42, 16, "char_ngram", 0.67, True, ood_weight=0.25, vicreg_weight=0.25, use_extended_families=True, family_holdout_eval=True, prompt_augmentation=True, paraphrase_weight=0.2),
        LatentExperiment("latent next dataaug paraphrase charngram annot67 holdout confirm seed43", 43, 16, "char_ngram", 0.67, True, ood_weight=0.25, vicreg_weight=0.25, use_extended_families=True, family_holdout_eval=True, prompt_augmentation=True, paraphrase_weight=0.2),
        LatentExperiment("latent next dataaug paraphrase charngram annot67 holdout confirm seed44", 44, 16, "char_ngram", 0.67, True, ood_weight=0.25, vicreg_weight=0.25, use_extended_families=True, family_holdout_eval=True, prompt_augmentation=True, paraphrase_weight=0.2),
        LatentExperiment("latent next dataaug paraphrase charngram annot75 strongnegs holdout", 42, 16, "char_ngram", 0.75, True, ood_weight=0.25, vicreg_weight=0.25, use_extended_families=True, use_stronger_family_negatives=True, family_holdout_eval=True, prompt_augmentation=True, paraphrase_weight=0.2),
        LatentExperiment("latent next dataaug paraphrase charngram annot75 strongnegs holdout confirm seed43", 43, 16, "char_ngram", 0.75, True, ood_weight=0.25, vicreg_weight=0.25, use_extended_families=True, use_stronger_family_negatives=True, family_holdout_eval=True, prompt_augmentation=True, paraphrase_weight=0.2),
        LatentExperiment("latent next dataaug paraphrase charngram annot75 strongnegs holdout confirm seed44", 44, 16, "char_ngram", 0.75, True, ood_weight=0.25, vicreg_weight=0.25, use_extended_families=True, use_stronger_family_negatives=True, family_holdout_eval=True, prompt_augmentation=True, paraphrase_weight=0.2),
        LatentExperiment("latent next dataaug paraphrase10 charngram annot75 holdout", 42, 16, "char_ngram", 0.75, True, ood_weight=0.25, vicreg_weight=0.25, use_extended_families=True, family_holdout_eval=True, prompt_augmentation=True, paraphrase_weight=0.1),
        LatentExperiment("latent next dataaug paraphrase10 charngram annot75 holdout confirm seed43", 43, 16, "char_ngram", 0.75, True, ood_weight=0.25, vicreg_weight=0.25, use_extended_families=True, family_holdout_eval=True, prompt_augmentation=True, paraphrase_weight=0.1),
        LatentExperiment("latent next dataaug paraphrase10 charngram annot75 holdout confirm seed44", 44, 16, "char_ngram", 0.75, True, ood_weight=0.25, vicreg_weight=0.25, use_extended_families=True, family_holdout_eval=True, prompt_augmentation=True, paraphrase_weight=0.1),
        LatentExperiment("latent next dataaug paraphrase10 vic20 charngram annot75 holdout", 42, 16, "char_ngram", 0.75, True, ood_weight=0.25, vicreg_weight=0.2, use_extended_families=True, family_holdout_eval=True, prompt_augmentation=True, paraphrase_weight=0.1),
        LatentExperiment("latent next dataaug paraphrase10 vic20 charngram annot75 holdout confirm seed43", 43, 16, "char_ngram", 0.75, True, ood_weight=0.25, vicreg_weight=0.2, use_extended_families=True, family_holdout_eval=True, prompt_augmentation=True, paraphrase_weight=0.1),
        LatentExperiment("latent next dataaug paraphrase10 vic20 charngram annot75 holdout confirm seed44", 44, 16, "char_ngram", 0.75, True, ood_weight=0.25, vicreg_weight=0.2, use_extended_families=True, family_holdout_eval=True, prompt_augmentation=True, paraphrase_weight=0.1),
        LatentExperiment("latent next rampreg charngram annot75 holdout", 42, 16, "char_ngram", 0.75, True, ood_weight=0.25, vicreg_weight=0.25, use_extended_families=True, family_holdout_eval=True, prompt_augmentation=True, paraphrase_weight=0.2, ramp_regularizers=True),
        LatentExperiment("latent next rampreg charngram annot75 holdout confirm seed43", 43, 16, "char_ngram", 0.75, True, ood_weight=0.25, vicreg_weight=0.25, use_extended_families=True, family_holdout_eval=True, prompt_augmentation=True, paraphrase_weight=0.2, ramp_regularizers=True),
        LatentExperiment("latent next rampreg charngram annot75 holdout confirm seed44", 44, 16, "char_ngram", 0.75, True, ood_weight=0.25, vicreg_weight=0.25, use_extended_families=True, family_holdout_eval=True, prompt_augmentation=True, paraphrase_weight=0.2, ramp_regularizers=True),
        LatentExperiment("latent next warmup rampreg charngram annot75 holdout", 42, 16, "char_ngram", 0.75, True, ood_weight=0.25, vicreg_weight=0.25, use_extended_families=True, family_holdout_eval=True, prompt_augmentation=True, paraphrase_weight=0.2, warmup_steps=60, ramp_regularizers=True),
        LatentExperiment("latent next warmup rampreg charngram annot75 holdout confirm seed43", 43, 16, "char_ngram", 0.75, True, ood_weight=0.25, vicreg_weight=0.25, use_extended_families=True, family_holdout_eval=True, prompt_augmentation=True, paraphrase_weight=0.2, warmup_steps=60, ramp_regularizers=True),
        LatentExperiment("latent next warmup rampreg charngram annot75 holdout confirm seed44", 44, 16, "char_ngram", 0.75, True, ood_weight=0.25, vicreg_weight=0.25, use_extended_families=True, family_holdout_eval=True, prompt_augmentation=True, paraphrase_weight=0.2, warmup_steps=60, ramp_regularizers=True),
        LatentExperiment("latent next warmup120 rampreg charngram annot75 holdout", 42, 16, "char_ngram", 0.75, True, ood_weight=0.25, vicreg_weight=0.25, use_extended_families=True, family_holdout_eval=True, prompt_augmentation=True, paraphrase_weight=0.2, warmup_steps=120, ramp_regularizers=True),
        LatentExperiment("latent next warmup120 rampreg charngram annot75 holdout confirm seed43", 43, 16, "char_ngram", 0.75, True, ood_weight=0.25, vicreg_weight=0.25, use_extended_families=True, family_holdout_eval=True, prompt_augmentation=True, paraphrase_weight=0.2, warmup_steps=120, ramp_regularizers=True),
        LatentExperiment("latent next warmup120 rampreg charngram annot75 holdout confirm seed44", 44, 16, "char_ngram", 0.75, True, ood_weight=0.25, vicreg_weight=0.25, use_extended_families=True, family_holdout_eval=True, prompt_augmentation=True, paraphrase_weight=0.2, warmup_steps=120, ramp_regularizers=True),
        LatentExperiment("latent next lowlr warmup rampreg charngram annot75 holdout", 42, 16, "char_ngram", 0.75, True, ood_weight=0.25, vicreg_weight=0.25, use_extended_families=True, family_holdout_eval=True, prompt_augmentation=True, paraphrase_weight=0.2, lr=3e-4, warmup_steps=60, ramp_regularizers=True),
        LatentExperiment("latent next lowlr warmup rampreg charngram annot75 holdout confirm seed43", 43, 16, "char_ngram", 0.75, True, ood_weight=0.25, vicreg_weight=0.25, use_extended_families=True, family_holdout_eval=True, prompt_augmentation=True, paraphrase_weight=0.2, lr=3e-4, warmup_steps=60, ramp_regularizers=True),
        LatentExperiment("latent next lowlr warmup rampreg charngram annot75 holdout confirm seed44", 44, 16, "char_ngram", 0.75, True, ood_weight=0.25, vicreg_weight=0.25, use_extended_families=True, family_holdout_eval=True, prompt_augmentation=True, paraphrase_weight=0.2, lr=3e-4, warmup_steps=60, ramp_regularizers=True),
        LatentExperiment("latent next lowlr warmup80 rampreg charngram annot75 holdout", 42, 16, "char_ngram", 0.75, True, ood_weight=0.25, vicreg_weight=0.25, use_extended_families=True, family_holdout_eval=True, prompt_augmentation=True, paraphrase_weight=0.2, lr=3e-4, warmup_steps=80, ramp_regularizers=True),
        LatentExperiment("latent next lowlr warmup80 rampreg charngram annot75 holdout confirm seed43", 43, 16, "char_ngram", 0.75, True, ood_weight=0.25, vicreg_weight=0.25, use_extended_families=True, family_holdout_eval=True, prompt_augmentation=True, paraphrase_weight=0.2, lr=3e-4, warmup_steps=80, ramp_regularizers=True),
        LatentExperiment("latent next lowlr warmup80 rampreg charngram annot75 holdout confirm seed44", 44, 16, "char_ngram", 0.75, True, ood_weight=0.25, vicreg_weight=0.25, use_extended_families=True, family_holdout_eval=True, prompt_augmentation=True, paraphrase_weight=0.2, lr=3e-4, warmup_steps=80, ramp_regularizers=True),
        LatentExperiment("latent next lowlr warmup80 rampreg charngram annot75 holdout confirm seed45", 45, 16, "char_ngram", 0.75, True, ood_weight=0.25, vicreg_weight=0.25, use_extended_families=True, family_holdout_eval=True, prompt_augmentation=True, paraphrase_weight=0.2, lr=3e-4, warmup_steps=80, ramp_regularizers=True),
        LatentExperiment("latent next lowlr warmup80 rampreg charngram annot67 holdout confirm seed45", 45, 16, "char_ngram", 0.67, True, ood_weight=0.25, vicreg_weight=0.25, use_extended_families=True, family_holdout_eval=True, prompt_augmentation=True, paraphrase_weight=0.2, lr=3e-4, warmup_steps=80, ramp_regularizers=True),
        LatentExperiment("latent next lowlr warmup80 rampreg charngram annot67 holdout confirm seed46", 46, 16, "char_ngram", 0.67, True, ood_weight=0.25, vicreg_weight=0.25, use_extended_families=True, family_holdout_eval=True, prompt_augmentation=True, paraphrase_weight=0.2, lr=3e-4, warmup_steps=80, ramp_regularizers=True),
        LatentExperiment("latent next lowlr warmup80 rampreg charngram annot67 holdout confirm seed47", 47, 16, "char_ngram", 0.67, True, ood_weight=0.25, vicreg_weight=0.25, use_extended_families=True, family_holdout_eval=True, prompt_augmentation=True, paraphrase_weight=0.2, lr=3e-4, warmup_steps=80, ramp_regularizers=True),
        LatentExperiment("latent next lowerlr warmup80 rampreg charngram annot67 holdout confirm seed47", 47, 16, "char_ngram", 0.67, True, ood_weight=0.25, vicreg_weight=0.25, use_extended_families=True, family_holdout_eval=True, prompt_augmentation=True, paraphrase_weight=0.2, lr=2.5e-4, warmup_steps=80, ramp_regularizers=True),
        LatentExperiment("latent next lowerlr warmup80 rampreg charngram annot67 holdout confirm seed48", 48, 16, "char_ngram", 0.67, True, ood_weight=0.25, vicreg_weight=0.25, use_extended_families=True, family_holdout_eval=True, prompt_augmentation=True, paraphrase_weight=0.2, lr=2.5e-4, warmup_steps=80, ramp_regularizers=True),
        LatentExperiment("latent next lowlr warmup80 strongnegs rampreg charngram annot67 holdout confirm seed47", 47, 16, "char_ngram", 0.67, True, ood_weight=0.25, vicreg_weight=0.25, use_extended_families=True, use_stronger_family_negatives=True, family_holdout_eval=True, prompt_augmentation=True, paraphrase_weight=0.2, lr=3e-4, warmup_steps=80, ramp_regularizers=True),
        LatentExperiment("latent next lowlr warmup80 strongnegs rampreg charngram annot67 holdout confirm seed48", 48, 16, "char_ngram", 0.67, True, ood_weight=0.25, vicreg_weight=0.25, use_extended_families=True, use_stronger_family_negatives=True, family_holdout_eval=True, prompt_augmentation=True, paraphrase_weight=0.2, lr=3e-4, warmup_steps=80, ramp_regularizers=True),
        LatentExperiment("latent next lowlr warmup80 ood30 rampreg charngram annot67 holdout confirm seed47", 47, 16, "char_ngram", 0.67, True, ood_weight=0.30, vicreg_weight=0.25, use_extended_families=True, family_holdout_eval=True, prompt_augmentation=True, paraphrase_weight=0.2, lr=3e-4, warmup_steps=80, ramp_regularizers=True),
        LatentExperiment("latent next lowlr warmup80 ood30 rampreg charngram annot67 holdout confirm seed48", 48, 16, "char_ngram", 0.67, True, ood_weight=0.30, vicreg_weight=0.25, use_extended_families=True, family_holdout_eval=True, prompt_augmentation=True, paraphrase_weight=0.2, lr=3e-4, warmup_steps=80, ramp_regularizers=True),
        LatentExperiment("latent next lowlr warmup80 strongnegs rampreg charngram annot75 holdout confirm seed45", 45, 16, "char_ngram", 0.75, True, ood_weight=0.25, vicreg_weight=0.25, use_extended_families=True, use_stronger_family_negatives=True, family_holdout_eval=True, prompt_augmentation=True, paraphrase_weight=0.2, lr=3e-4, warmup_steps=80, ramp_regularizers=True),
        LatentExperiment("latent next lowlr warmup80 strongnegs rampreg charngram annot75 holdout confirm seed46", 46, 16, "char_ngram", 0.75, True, ood_weight=0.25, vicreg_weight=0.25, use_extended_families=True, use_stronger_family_negatives=True, family_holdout_eval=True, prompt_augmentation=True, paraphrase_weight=0.2, lr=3e-4, warmup_steps=80, ramp_regularizers=True),
        LatentExperiment("latent next lowlr warmup80 ood30 rampreg charngram annot75 holdout confirm seed45", 45, 16, "char_ngram", 0.75, True, ood_weight=0.30, vicreg_weight=0.25, use_extended_families=True, family_holdout_eval=True, prompt_augmentation=True, paraphrase_weight=0.2, lr=3e-4, warmup_steps=80, ramp_regularizers=True),
        LatentExperiment("latent next lowlr warmup80 ood30 rampreg charngram annot75 holdout confirm seed46", 46, 16, "char_ngram", 0.75, True, ood_weight=0.30, vicreg_weight=0.25, use_extended_families=True, family_holdout_eval=True, prompt_augmentation=True, paraphrase_weight=0.2, lr=3e-4, warmup_steps=80, ramp_regularizers=True),
        LatentExperiment("latent next lowlr warmup80 paraphrase18 rampreg charngram annot75 holdout confirm seed45", 45, 16, "char_ngram", 0.75, True, ood_weight=0.25, vicreg_weight=0.25, use_extended_families=True, family_holdout_eval=True, prompt_augmentation=True, paraphrase_weight=0.18, lr=3e-4, warmup_steps=80, ramp_regularizers=True),
        LatentExperiment("latent next lowlr warmup80 paraphrase18 rampreg charngram annot75 holdout confirm seed46", 46, 16, "char_ngram", 0.75, True, ood_weight=0.25, vicreg_weight=0.25, use_extended_families=True, family_holdout_eval=True, prompt_augmentation=True, paraphrase_weight=0.18, lr=3e-4, warmup_steps=80, ramp_regularizers=True),
        LatentExperiment("latent next lowlr warmup80 paraphrase18 rampreg charngram annot75 holdout confirm seed47", 47, 16, "char_ngram", 0.75, True, ood_weight=0.25, vicreg_weight=0.25, use_extended_families=True, family_holdout_eval=True, prompt_augmentation=True, paraphrase_weight=0.18, lr=3e-4, warmup_steps=80, ramp_regularizers=True),
        LatentExperiment("latent next lowlr warmup80 vic20 rampreg charngram annot75 holdout confirm seed45", 45, 16, "char_ngram", 0.75, True, ood_weight=0.25, vicreg_weight=0.2, use_extended_families=True, family_holdout_eval=True, prompt_augmentation=True, paraphrase_weight=0.2, lr=3e-4, warmup_steps=80, ramp_regularizers=True),
        LatentExperiment("latent next lowlr warmup80 vic20 rampreg charngram annot75 holdout confirm seed46", 46, 16, "char_ngram", 0.75, True, ood_weight=0.25, vicreg_weight=0.2, use_extended_families=True, family_holdout_eval=True, prompt_augmentation=True, paraphrase_weight=0.2, lr=3e-4, warmup_steps=80, ramp_regularizers=True),
        LatentExperiment("latent next lowlr warmup80 vic20 rampreg charngram annot75 holdout confirm seed47", 47, 16, "char_ngram", 0.75, True, ood_weight=0.25, vicreg_weight=0.2, use_extended_families=True, family_holdout_eval=True, prompt_augmentation=True, paraphrase_weight=0.2, lr=3e-4, warmup_steps=80, ramp_regularizers=True),
        LatentExperiment("latent next lowerlr warmup rampreg charngram annot75 holdout", 42, 16, "char_ngram", 0.75, True, ood_weight=0.25, vicreg_weight=0.25, use_extended_families=True, family_holdout_eval=True, prompt_augmentation=True, paraphrase_weight=0.2, lr=2.5e-4, warmup_steps=60, ramp_regularizers=True),
        LatentExperiment("latent next lowerlr warmup rampreg charngram annot75 holdout confirm seed43", 43, 16, "char_ngram", 0.75, True, ood_weight=0.25, vicreg_weight=0.25, use_extended_families=True, family_holdout_eval=True, prompt_augmentation=True, paraphrase_weight=0.2, lr=2.5e-4, warmup_steps=60, ramp_regularizers=True),
        LatentExperiment("latent next lowerlr warmup rampreg charngram annot75 holdout confirm seed44", 44, 16, "char_ngram", 0.75, True, ood_weight=0.25, vicreg_weight=0.25, use_extended_families=True, family_holdout_eval=True, prompt_augmentation=True, paraphrase_weight=0.2, lr=2.5e-4, warmup_steps=60, ramp_regularizers=True),
        LatentExperiment("latent next lowerlr warmup rampreg charngram annot75 holdout confirm seed45", 45, 16, "char_ngram", 0.75, True, ood_weight=0.25, vicreg_weight=0.25, use_extended_families=True, family_holdout_eval=True, prompt_augmentation=True, paraphrase_weight=0.2, lr=2.5e-4, warmup_steps=60, ramp_regularizers=True),
        LatentExperiment("latent next lowerlr warmup rampreg charngram annot75 holdout confirm seed46", 46, 16, "char_ngram", 0.75, True, ood_weight=0.25, vicreg_weight=0.25, use_extended_families=True, family_holdout_eval=True, prompt_augmentation=True, paraphrase_weight=0.2, lr=2.5e-4, warmup_steps=60, ramp_regularizers=True),
        LatentExperiment("latent next lowerlr warmup rampreg charngram annot75 holdout confirm seed47", 47, 16, "char_ngram", 0.75, True, ood_weight=0.25, vicreg_weight=0.25, use_extended_families=True, family_holdout_eval=True, prompt_augmentation=True, paraphrase_weight=0.2, lr=2.5e-4, warmup_steps=60, ramp_regularizers=True),
        LatentExperiment("latent next lowerlr warmup80 rampreg charngram annot75 holdout confirm seed45", 45, 16, "char_ngram", 0.75, True, ood_weight=0.25, vicreg_weight=0.25, use_extended_families=True, family_holdout_eval=True, prompt_augmentation=True, paraphrase_weight=0.2, lr=2.5e-4, warmup_steps=80, ramp_regularizers=True),
        LatentExperiment("latent next lowerlr warmup80 rampreg charngram annot75 holdout confirm seed46", 46, 16, "char_ngram", 0.75, True, ood_weight=0.25, vicreg_weight=0.25, use_extended_families=True, family_holdout_eval=True, prompt_augmentation=True, paraphrase_weight=0.2, lr=2.5e-4, warmup_steps=80, ramp_regularizers=True),
        LatentExperiment("latent next lowerlr warmup80 rampreg charngram annot75 holdout confirm seed47", 47, 16, "char_ngram", 0.75, True, ood_weight=0.25, vicreg_weight=0.25, use_extended_families=True, family_holdout_eval=True, prompt_augmentation=True, paraphrase_weight=0.2, lr=2.5e-4, warmup_steps=80, ramp_regularizers=True),
        LatentExperiment("latent extended tuned balanced lowtemp", 42, 16, "byte_hash", 1.0, True, temperature=0.03, embedding_weight=0.9, mse_weight=0.05, ood_weight=0.3, vicreg_weight=0.2, use_extended_families=True),
        LatentExperiment("latent extended tuned balanced lowtemp confirm seed43", 43, 16, "byte_hash", 1.0, True, temperature=0.03, embedding_weight=0.9, mse_weight=0.05, ood_weight=0.3, vicreg_weight=0.2, use_extended_families=True),
        LatentExperiment("latent research multisource nulls annot67 confirm seed49", 49, 16, "char_ngram", 0.67, True, ood_weight=0.25, vicreg_weight=0.25, use_extended_families=True, family_holdout_eval=True, prompt_augmentation=True, paraphrase_weight=0.2, lr=2.5e-4, warmup_steps=80, ramp_regularizers=True, use_virtual_nulls=True, use_multisource_families=True),
        LatentExperiment("latent research multisource nulls annot67 confirm seed50", 50, 16, "char_ngram", 0.67, True, ood_weight=0.25, vicreg_weight=0.25, use_extended_families=True, family_holdout_eval=True, prompt_augmentation=True, paraphrase_weight=0.2, lr=2.5e-4, warmup_steps=80, ramp_regularizers=True, use_virtual_nulls=True, use_multisource_families=True),
        LatentExperiment("latent research minednegs nulls annot67 confirm seed49", 49, 16, "char_ngram", 0.67, True, ood_weight=0.25, vicreg_weight=0.25, use_extended_families=True, family_holdout_eval=True, prompt_augmentation=True, paraphrase_weight=0.2, lr=2.5e-4, warmup_steps=80, ramp_regularizers=True, use_mined_negatives=True, use_virtual_nulls=True),
        LatentExperiment("latent research minednegs nulls annot67 confirm seed50", 50, 16, "char_ngram", 0.67, True, ood_weight=0.25, vicreg_weight=0.25, use_extended_families=True, family_holdout_eval=True, prompt_augmentation=True, paraphrase_weight=0.2, lr=2.5e-4, warmup_steps=80, ramp_regularizers=True, use_mined_negatives=True, use_virtual_nulls=True),
        LatentExperiment("latent research fullmix annot67 confirm seed49", 49, 16, "char_ngram", 0.67, True, ood_weight=0.25, vicreg_weight=0.25, use_extended_families=True, family_holdout_eval=True, prompt_augmentation=True, paraphrase_weight=0.2, lr=2.5e-4, warmup_steps=80, ramp_regularizers=True, use_mined_negatives=True, use_virtual_nulls=True, use_multisource_families=True),
        LatentExperiment("latent research fullmix annot67 confirm seed50", 50, 16, "char_ngram", 0.67, True, ood_weight=0.25, vicreg_weight=0.25, use_extended_families=True, family_holdout_eval=True, prompt_augmentation=True, paraphrase_weight=0.2, lr=2.5e-4, warmup_steps=80, ramp_regularizers=True, use_mined_negatives=True, use_virtual_nulls=True, use_multisource_families=True),
        LatentExperiment("latent research multisource nulls corrected annot67 confirm seed49", 49, 16, "char_ngram", 0.67, True, ood_weight=0.25, vicreg_weight=0.25, use_extended_families=True, family_holdout_eval=True, prompt_augmentation=True, paraphrase_weight=0.2, lr=2.5e-4, warmup_steps=80, ramp_regularizers=True, use_virtual_nulls=True, use_multisource_families=True),
        LatentExperiment("latent research multisource nulls corrected annot67 confirm seed50", 50, 16, "char_ngram", 0.67, True, ood_weight=0.25, vicreg_weight=0.25, use_extended_families=True, family_holdout_eval=True, prompt_augmentation=True, paraphrase_weight=0.2, lr=2.5e-4, warmup_steps=80, ramp_regularizers=True, use_virtual_nulls=True, use_multisource_families=True),
        LatentExperiment("latent research fullmix corrected annot67 confirm seed49", 49, 16, "char_ngram", 0.67, True, ood_weight=0.25, vicreg_weight=0.25, use_extended_families=True, family_holdout_eval=True, prompt_augmentation=True, paraphrase_weight=0.2, lr=2.5e-4, warmup_steps=80, ramp_regularizers=True, use_mined_negatives=True, use_virtual_nulls=True, use_multisource_families=True),
        LatentExperiment("latent research fullmix corrected annot67 confirm seed50", 50, 16, "char_ngram", 0.67, True, ood_weight=0.25, vicreg_weight=0.25, use_extended_families=True, family_holdout_eval=True, prompt_augmentation=True, paraphrase_weight=0.2, lr=2.5e-4, warmup_steps=80, ramp_regularizers=True, use_mined_negatives=True, use_virtual_nulls=True, use_multisource_families=True),
    ]
    return recipes


def save_run_artifacts(run_id: str, exp: LatentExperiment, metrics: dict[str, float | bool]) -> None:
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": exp.name,
        "seed": exp.seed,
        "patch_size": exp.patch_size,
        "tokenizer_mode": exp.tokenizer_mode,
        "annotation_prob": exp.annotation_prob,
        "use_hard_negatives": exp.use_hard_negatives,
        "temperature": exp.temperature,
        "embedding_weight": exp.embedding_weight,
        "mse_weight": exp.mse_weight,
        "sigreg_weight": exp.sigreg_weight,
        "ood_weight": exp.ood_weight,
        "vicreg_weight": exp.vicreg_weight,
        "use_extended_families": exp.use_extended_families,
        "paraphrase_weight": exp.paraphrase_weight,
        "use_stronger_family_negatives": exp.use_stronger_family_negatives,
        "family_holdout_eval": exp.family_holdout_eval,
        "prompt_augmentation": exp.prompt_augmentation,
        "lr": exp.lr,
        "warmup_steps": exp.warmup_steps,
        "ramp_regularizers": exp.ramp_regularizers,
        "use_mined_negatives": exp.use_mined_negatives,
        "use_virtual_nulls": exp.use_virtual_nulls,
        "use_multisource_families": exp.use_multisource_families,
        "metrics": metrics,
    }
    (run_dir / "latent_space_ablation.json").write_text(json.dumps(payload, indent=2) + "\n")


def collect_family_negative_texts(
    batch_pairs: list[tuple[str, str, str, list[str]]],
    family_lookup: dict[str, TaskFamily],
    use_stronger_family_negatives: bool,
) -> list[str]:
    negative_texts: list[str] = []
    for _, _, family_name, negatives in batch_pairs:
        negative_texts.extend(negatives)
        if not use_stronger_family_negatives:
            continue
        family = family_lookup[family_name]
        negative_texts.extend(list(family.codes))
        for other_family_name, other_family in family_lookup.items():
            if other_family_name == family_name:
                continue
            if other_family_name.split("_")[0] == family_name.split("_")[0]:
                negative_texts.extend(list(other_family.codes))
                negative_texts.extend(list(other_family.hard_negatives))
    return negative_texts


def family_paraphrase_loss(
    batch_pairs: list[tuple[str, str, str, list[str]]],
    family_lookup: dict[str, TaskFamily],
    tokenizer: FlexibleTokenizer,
    config,
    device: str,
    model: JEPAModel,
    annotation_prob: float,
    z_text: torch.Tensor,
    z_pred: torch.Tensor,
    z_code: torch.Tensor,
) -> torch.Tensor:
    if not batch_pairs:
        return z_text.new_tensor(0.0)

    alt_prompts: list[str] = []
    alt_codes: list[str] = []
    for prompt, _, family_name, _ in batch_pairs:
        family = family_lookup[family_name]
        prompt_choices = [candidate for candidate in family.prompts if candidate != prompt]
        alt_prompts.append(random.choice(prompt_choices or list(family.prompts)))
        alt_code_raw = random.choice(family.codes)
        alt_codes.append(annotate_code(random.choice(family.prompts), alt_code_raw, annotation_prob))

    alt_text_ids = tokenizer.batch_encode(alt_prompts, config.max_seq_len, device)
    alt_code_ids = tokenizer.batch_encode(alt_codes, config.max_seq_len, device)
    z_alt_text = model.encode(alt_text_ids)
    z_alt_pred = model.predict(z_alt_text, action_id=1)
    z_alt_code = model.encode(alt_code_ids)

    return (
        normalized_mse_loss(z_text, z_alt_text)
        + normalized_mse_loss(z_pred, z_alt_pred)
        + normalized_mse_loss(z_code, z_alt_code)
    ) / 3.0


def encode_hard_negatives(
    tokenizer: FlexibleTokenizer,
    config,
    device: str,
    hard_negative_texts: list[str],
    model: JEPAModel,
) -> torch.Tensor:
    if not hard_negative_texts:
        return torch.empty(0, config.embed_dim, device=device)
    negative_ids = tokenizer.batch_encode(hard_negative_texts, config.max_seq_len, device)
    with torch.no_grad():
        negative_latents = model.encode(negative_ids)
    return negative_latents.detach()


def train_and_evaluate(exp: LatentExperiment, device: str) -> dict[str, float | bool]:
    set_seed(exp.seed)
    config = _proxy_config(15_000_000, "v5_distinct")
    config.patch_size = exp.patch_size
    model = JEPAModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=exp.lr)
    tokenizer = FlexibleTokenizer(vocab_size=config.vocab_size, mode=exp.tokenizer_mode)
    if exp.use_multisource_families:
        families = MULTISOURCE_TASK_FAMILIES
    elif exp.use_extended_families:
        families = EXTENDED_TASK_FAMILIES
    else:
        families = TASK_FAMILIES
    family_lookup = {family.name: family for family in families}
    pairs = build_training_pairs(
        repeats=128,
        annotation_prob=exp.annotation_prob,
        families=families,
        holdout_last_prompt=exp.family_holdout_eval,
        prompt_augmentation=exp.prompt_augmentation,
    )
    batch_size = 6
    steps = 360
    code_queue = QueueBuffer(512, config.embed_dim, device)
    stat_queue = QueueBuffer(512, config.embed_dim, device)

    model.train()
    for step in range(steps):
        if exp.warmup_steps > 0:
            lr_scale = min((step + 1) / exp.warmup_steps, 1.0)
            for param_group in optimizer.param_groups:
                param_group["lr"] = exp.lr * lr_scale

        batch_pairs = [pairs[(step * batch_size + offset) % len(pairs)] for offset in range(batch_size)]
        batch_prompts = [item[0] for item in batch_pairs]
        batch_codes = [item[1] for item in batch_pairs]
        hard_negative_texts: list[str] = collect_family_negative_texts(
            batch_pairs,
            family_lookup,
            exp.use_stronger_family_negatives,
        ) if exp.use_hard_negatives else []

        texts = tokenizer.batch_encode(batch_prompts, config.max_seq_len, device)
        codes = tokenizer.batch_encode(batch_codes, config.max_seq_len, device)
        ood = tokenizer.batch_encode(sample_ood_queries(batch_size), config.max_seq_len, device)

        z_text = model.encode(texts)
        z_code = model.encode(codes)
        z_pred = model.predict(z_text, action_id=1)
        z_ood = model.encode(ood)
        z_ood_pred = model.predict(z_ood, action_id=1)

        mined_negative_pool = torch.empty(0, config.embed_dim, device=device)
        if exp.use_mined_negatives or exp.use_virtual_nulls:
            candidate_negative_texts = build_negative_candidates(
                batch_pairs,
                family_lookup,
                exp.use_mined_negatives,
                exp.use_virtual_nulls,
            )
            mined_negative_pool = encode_mined_negative_pool(
                tokenizer,
                config,
                device,
                model,
                candidate_negative_texts,
                z_pred,
            )

        negative_pool = code_queue.get()
        hard_negative_pool = encode_hard_negatives(tokenizer, config, device, hard_negative_texts, model)
        pool_parts = [pool for pool in (negative_pool, hard_negative_pool, mined_negative_pool) if pool.numel()]
        if len(pool_parts) > 1:
            combined_pool = torch.cat(pool_parts, dim=0)
        elif pool_parts:
            combined_pool = pool_parts[0]
        else:
            combined_pool = negative_pool

        pred_loss, _ = paired_alignment_loss(
            z_text,
            z_code,
            z_pred,
            negative_pool=combined_pool,
            temperature=exp.temperature,
            prediction_weight=1.0,
            embedding_weight=exp.embedding_weight,
            mse_weight=exp.mse_weight,
        )
        code_candidates = torch.cat([z_code.detach(), combined_pool], dim=0) if combined_pool.numel() else z_code.detach()
        ignorance = ignorance_penalty(z_ood, code_candidates) + ignorance_penalty(z_ood_pred, code_candidates)
        stat_queue.push(z_text)
        stat_queue.push(z_code)
        reg_loss = sigreg_loss(stat_queue.get().unsqueeze(1), m=128, lambda_reg=0.05) if stat_queue.get().shape[0] >= 64 else z_text.new_tensor(0.0)
        joint_latents = torch.cat([z_text, z_code, z_pred], dim=0)
        vicreg_loss = variance_covariance_loss(joint_latents)
        paraphrase_loss = family_paraphrase_loss(
            batch_pairs,
            family_lookup,
            tokenizer,
            config,
            device,
            model,
            exp.annotation_prob,
            z_text,
            z_pred,
            z_code,
        )
        regularizer_scale = min((step + 1) / max(steps // 3, 1), 1.0) if exp.ramp_regularizers else 1.0

        loss = (
            pred_loss
            + regularizer_scale * exp.ood_weight * ignorance
            + regularizer_scale * exp.sigreg_weight * reg_loss
            + regularizer_scale * exp.vicreg_weight * vicreg_loss
            + regularizer_scale * exp.paraphrase_weight * paraphrase_loss
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        code_queue.push(z_code)

    snippets = build_index_snippets(families)
    holdout_queries = build_holdout_queries(families) if exp.family_holdout_eval else []
    with torch.no_grad():
        model.eval()
        code_ids = tokenizer.batch_encode(snippets, config.max_seq_len, device)
        z_code = model.encode(code_ids)
        index = VectorIndex(snippets, z_code.cpu())

        dense_scores: list[tuple[str, float]] = []
        for label, queries in (("known", KNOWN_QUERIES), ("ood", OOD_QUERIES)):
            for query in queries:
                query_ids = tokenizer.batch_encode([query], config.max_seq_len, device)
                z_query = model.encode(query_ids)
                z_pred = model.predict(z_query, action_id=1)
                score = float(index.search(z_pred.cpu(), k=1).scores[0].item())
                dense_scores.append((label, score))

        pair_sims = []
        for left, right in PARAPHRASE_PAIRS:
            left_ids = tokenizer.batch_encode([left], config.max_seq_len, device)
            right_ids = tokenizer.batch_encode([right], config.max_seq_len, device)
            left_latent = F.normalize(model.encode(left_ids).float(), dim=-1)
            right_latent = F.normalize(model.encode(right_ids).float(), dim=-1)
            pair_sims.append(float((left_latent @ right_latent.T).item()))

        z_norm = F.normalize(z_code.float(), dim=-1)
        sim_matrix = z_norm @ z_norm.T
        code_avg_similarity = float(((sim_matrix.sum() - len(snippets)) / (len(snippets) * (len(snippets) - 1))).item())
        code_isotropy = isotropic_score(z_code)
        collapsed = collapse_detected(z_code, threshold=0.98)

        holdout_scores: list[tuple[str, float]] = []
        if holdout_queries:
            for query in holdout_queries:
                query_ids = tokenizer.batch_encode([query], config.max_seq_len, device)
                z_query = model.encode(query_ids)
                z_pred = model.predict(z_query, action_id=1)
                score = float(index.search(z_pred.cpu(), k=1).scores[0].item())
                holdout_scores.append(("holdout", score))

    dense_gap = avg(dense_scores, "known") - avg(dense_scores, "ood")
    metrics = {
        "dense_gap": dense_gap,
        "ood_gap": avg(dense_scores, "known") - avg(dense_scores, "ood"),
        "known_paraphrase_similarity": sum(pair_sims) / max(len(pair_sims), 1),
        "code_isotropy": code_isotropy,
        "code_avg_similarity": code_avg_similarity,
        "collapse_detected": collapsed,
        "holdout_similarity": avg(holdout_scores, "holdout") if holdout_scores else 0.0,
        "dense_gap_pass": dense_gap > 0.08,
        "cohesion_pass": (sum(pair_sims) / max(len(pair_sims), 1)) > 0.65,
        "isotropy_pass": code_isotropy > 0.55,
    }
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-experiments", type=int, default=999)
    parser.add_argument("--match", type=str, default="")
    args = parser.parse_args()

    ensure_results_header()
    history_rows = parse_results_table()
    seen_descriptions = {row.get("description", "") for row in history_rows if row.get("status") == "ok"}
    queue = [exp for exp in candidate_queue() if f"{exp.name} seed{exp.seed}" not in seen_descriptions]
    if args.match:
        match = args.match.lower()
        queue = [exp for exp in queue if match in exp.name.lower()]
    if not queue:
        return 0

    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_index = next_run_index(history_rows)
    for exp in queue[: args.max_experiments]:
        description = f"{exp.name} seed{exp.seed}"
        run_id = f"{run_index:03d}-{slugify(description)}"
        metrics = train_and_evaluate(exp, device)
        save_run_artifacts(run_id, exp, metrics)
        append_result(run_id, device, metrics, description)
        print(f"{run_id}: dense_gap={metrics['dense_gap']:.3f} paraphrase={metrics['known_paraphrase_similarity']:.3f} isotropy={metrics['code_isotropy']:.3f} avg_sim={metrics['code_avg_similarity']:.3f} collapse={metrics['collapse_detected']}")
        run_index += 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())