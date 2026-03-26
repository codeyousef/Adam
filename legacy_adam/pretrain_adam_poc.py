"""
pretrain_adam_poc.py — From-Scratch Pretraining for Adam PoC (494M)

Trains a 494M-parameter Qwen2-architecture decoder-only model from random
initialization on a 100% synthetic knowledge-sparse corpus. Emits all data
required for a research paper (loss curves, scaling law fits, hardware
utilization, LaTeX tables) and monitors RTX 4090 temperature and power in
real time.

Usage:
    python pretrain_adam_poc.py --data hope/adam_training_data/pretrain_corpus \
        --val-data hope/adam_training_data/pretrain_val.jsonl

Smoke test (quick run on small data):
    python pretrain_adam_poc.py --total-tokens 1000000 --save-every 50 \
        --val-every 100 --log-every 5 \
        --data <small.jsonl> --val-data <small.jsonl>
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterator

import torch
import torch.nn as nn
import torch.utils.checkpoint as ckpt
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM


# ─────────────────────────────────────────────────────────────────────────────
# Section 1: Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PretrainConfig:
    # ── Architecture (PoC 494M) ───────────────────────────────────────────
    n_layers: int = 24
    d_model: int = 896
    n_heads: int = 14          # query heads
    n_kv_heads: int = 2        # GQA 7:1 ratio
    d_ff: int = 4864           # ~5.4× SwiGLU expansion
    rope_theta: float = 10000.0
    vocab_size: int = 151936   # Qwen2.5 tokenizer unchanged
    max_seq_len: int = 2048

    # ── Training ──────────────────────────────────────────────────────────
    batch_size: int = 4
    bf16: bool = True
    grad_clip: float = 1.0
    total_tokens: int = 6_000_000_000   # 6B tokens

    # ── Optimizer ─────────────────────────────────────────────────────────
    lr: float = 3e-4
    lr_min_ratio: float = 0.1   # cosine decays to 0.1 × peak
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    warmup_ratio: float = 0.05  # 5% of total steps

    # ── Data ──────────────────────────────────────────────────────────────
    data_path: str = "hope/adam_training_data/pretrain_corpus"
    val_path: str = "hope/adam_training_data/pretrain_val.jsonl"
    tokenizer_name: str = "Qwen/Qwen2.5-0.5B"  # same vocab

    # ── Output ────────────────────────────────────────────────────────────
    output_dir: str = "adam_poc_checkpoints"
    save_every: int = 500
    keep_checkpoints: int = 5
    log_every: int = 10
    val_every: int = 5000
    grad_flow_every: int = 200
    scaling_every: int = 1000

    # ── Resume ────────────────────────────────────────────────────────────
    resume_from: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Section 2: Model Construction
# ─────────────────────────────────────────────────────────────────────────────

def build_model(config: PretrainConfig) -> Qwen2ForCausalLM:
    """Build 494M Qwen2 model from random init with scaled weight initialization."""
    qwen_cfg = Qwen2Config(
        hidden_size=config.d_model,
        num_hidden_layers=config.n_layers,
        num_attention_heads=config.n_heads,
        num_key_value_heads=config.n_kv_heads,
        intermediate_size=config.d_ff,
        hidden_act="silu",           # SwiGLU uses SiLU gate
        rms_norm_eps=1e-6,
        rope_theta=config.rope_theta,
        vocab_size=config.vocab_size,
        max_position_embeddings=config.max_seq_len * 2,  # headroom
        tie_word_embeddings=True,    # tied embed → ~494M target (not 630M)
        use_sliding_window=False,
        attention_dropout=0.0,
    )
    model = Qwen2ForCausalLM(qwen_cfg)  # random init — no pretrained weights
    _apply_scaled_init(model, config)
    return model


def _apply_scaled_init(model: Qwen2ForCausalLM, config: PretrainConfig) -> None:
    """Stability-spec initialization from Section 1a of the research document."""
    # Embeddings: std = 1/sqrt(d_model)
    std_emb = 1.0 / math.sqrt(config.d_model)
    nn.init.normal_(model.model.embed_tokens.weight, mean=0.0, std=std_emb)
    # lm_head shares embed_tokens.weight when tie_word_embeddings=True,
    # so do NOT re-initialize it separately (would overwrite embed init).


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# ─────────────────────────────────────────────────────────────────────────────
# Section 3: Selective Gradient Checkpointing
# ─────────────────────────────────────────────────────────────────────────────

def apply_selective_gc(model: Qwen2ForCausalLM, every_n: int = 4) -> None:
    """
    Enable gradient checkpointing.

    Ideally we'd checkpoint every `every_n` layers (6 of 24) for a 15%
    throughput cost, but monkey-patching layer.forward with use_reentrant=False
    triggers a tensor-count mismatch under BF16 autocast. HF's built-in
    gradient_checkpointing_enable() handles autocast correctly and is the safe
    choice. On H200 (141 GB) with a 494M model, GC may not be necessary at all.
    """
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    n_layers = len(model.model.layers)
    print(f"  Gradient checkpointing: all {n_layers} layers (HF built-in, use_reentrant=False)")
    # NOTE: GC is conservative for H100 (80GB) — a 494M model fits without it.
    # Keeping GC enabled for checkpoint-compatibility when resuming.


# ─────────────────────────────────────────────────────────────────────────────
# Section 4: Streaming Dataset (no-padding token packing)
# ─────────────────────────────────────────────────────────────────────────────

class PretrainingDataset(IterableDataset):
    """
    Streams JSONL files with {"text": "..."} records.
    Tokenizes and packs tokens into exactly seq_len-token windows with no
    padding — pure packing for maximum training efficiency.

    data_path can be:
      - A single .jsonl file
      - A directory containing .jsonl files (streamed in sorted order)
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        seq_len: int,
        seed: int = 42,
    ):
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.seed = seed

    def _jsonl_files(self) -> list[Path]:
        if self.data_path.is_dir():
            files = sorted(self.data_path.glob("*.jsonl"))
            if not files:
                raise FileNotFoundError(
                    f"No .jsonl files found in {self.data_path}"
                )
            return files
        elif self.data_path.is_file():
            return [self.data_path]
        else:
            raise FileNotFoundError(f"Data path not found: {self.data_path}")

    def _stream_texts(self) -> Iterator[str]:
        files = self._jsonl_files()
        rng = random.Random(self.seed)
        order = list(range(len(files)))
        rng.shuffle(order)
        for idx in order:
            path = files[idx]
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        doc = json.loads(line)
                        text = doc.get("text", "")
                        if text:
                            yield text
                    except json.JSONDecodeError:
                        continue

    def __iter__(self):
        eos_id = self.tokenizer.eos_token_id
        token_buffer: list[int] = []

        for text in self._stream_texts():
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            ids.append(eos_id)  # document boundary
            token_buffer.extend(ids)

            while len(token_buffer) >= self.seq_len + 1:
                chunk = token_buffer[: self.seq_len + 1]
                token_buffer = token_buffer[self.seq_len :]
                input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
                labels = torch.tensor(chunk[1:], dtype=torch.long)
                yield {"input_ids": input_ids, "labels": labels}


# ─────────────────────────────────────────────────────────────────────────────
# Section 5: Hardware Monitor (RTX 4090 specific)
# ─────────────────────────────────────────────────────────────────────────────

class HardwareMonitor:
    """
    Polls nvidia-smi for GPU metrics every log_every steps.
    Issues warnings at 83°C and critical alerts at 87°C.
    Detects power throttling when draw > 95% of limit.
    """

    WARN_TEMP_C = 75       # H100 SXM runs cooler; warn earlier
    CRITICAL_TEMP_C = 80
    THROTTLE_RATIO = 0.95

    def __init__(self):
        self._available = self._check_nvidia_smi()

    def _check_nvidia_smi(self) -> bool:
        try:
            subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, timeout=5, check=True,
            )
            return True
        except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            print("  WARNING: nvidia-smi not available — hardware monitoring disabled")
            return False

    def get_metrics(self) -> dict:
        if not self._available:
            return {}
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=temperature.gpu,power.draw,power.limit,"
                    "memory.used,memory.total,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True, text=True, timeout=5,
            )
            parts = [p.strip() for p in result.stdout.strip().split(",")]
            temp_c = float(parts[0])
            power_draw_w = float(parts[1])
            power_limit_w = float(parts[2])
            vram_used_gb = float(parts[3]) / 1024.0
            vram_total_gb = float(parts[4]) / 1024.0
            gpu_util_pct = float(parts[5])
            return {
                "temp_c": temp_c,
                "power_draw_w": power_draw_w,
                "power_limit_w": power_limit_w,
                "vram_used_gb": vram_used_gb,
                "vram_total_gb": vram_total_gb,
                "gpu_util_pct": gpu_util_pct,
                "throttling": power_draw_w > self.THROTTLE_RATIO * power_limit_w,
            }
        except Exception as exc:
            return {"error": str(exc)}

    def check_and_warn(self, metrics: dict, step: int) -> None:
        if not metrics or "error" in metrics:
            return
        temp = metrics.get("temp_c", 0)
        if temp >= self.CRITICAL_TEMP_C:
            print(
                f"[STEP {step}] CRITICAL: GPU {temp}°C — consider pausing training",
                flush=True,
            )
        elif temp >= self.WARN_TEMP_C:
            print(f"[STEP {step}] WARNING: GPU {temp}°C — approaching thermal limit", flush=True)
        if metrics.get("throttling", False):
            draw = metrics.get("power_draw_w", 0)
            limit = metrics.get("power_limit_w", 0)
            print(
                f"[STEP {step}] WARNING: Power throttling "
                f"({draw:.0f}W / {limit:.0f}W = {100*draw/max(limit,1):.1f}%)",
                flush=True,
            )


# ─────────────────────────────────────────────────────────────────────────────
# Section 6: Research Paper Metrics
# ─────────────────────────────────────────────────────────────────────────────

class PretrainMetrics:
    """
    Manages all CSV/JSON output for the research paper.

    Files written:
      training_metrics.csv   — step loss, LR, grad norm, throughput (every log_every)
      hardware_metrics.csv   — GPU temp, power, VRAM, utilization (every log_every)
      val_metrics.csv        — held-out loss and perplexity (every val_every)
      scaling_law_data.csv   — tokens vs. loss for Chinchilla fit (every scaling_every)
      gradient_flow.csv      — per-layer grad norms (every grad_flow_every)
      checkpoint_log.csv     — checkpoint paths and stats
      experiment_metadata.json
      summary_statistics.json
      paper_tables.tex       — 3 LaTeX tables (written at end)
    """

    def __init__(self, output_dir: str, config: PretrainConfig):
        self.metrics_dir = Path(output_dir) / "paper_metrics"
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.config = config
        self.start_time = datetime.now()

        # CSV writers
        self._train_file = open(self.metrics_dir / "training_metrics.csv", "w", newline="")
        self._hw_file = open(self.metrics_dir / "hardware_metrics.csv", "w", newline="")
        self._val_file = open(self.metrics_dir / "val_metrics.csv", "w", newline="")
        self._scaling_file = open(self.metrics_dir / "scaling_law_data.csv", "w", newline="")
        self._grad_file = open(self.metrics_dir / "gradient_flow.csv", "w", newline="")
        self._ckpt_file = open(self.metrics_dir / "checkpoint_log.csv", "w", newline="")

        self._train_writer = csv.DictWriter(
            self._train_file,
            fieldnames=["step", "timestamp", "loss", "perplexity", "lr",
                        "grad_norm", "tokens_seen", "throughput_tok_s"],
        )
        self._hw_writer = csv.DictWriter(
            self._hw_file,
            fieldnames=["step", "timestamp", "temp_c", "power_draw_w",
                        "power_limit_w", "vram_used_gb", "vram_total_gb",
                        "gpu_util_pct", "throttling"],
        )
        self._val_writer = csv.DictWriter(
            self._val_file,
            fieldnames=["step", "tokens_seen", "val_loss", "val_perplexity"],
        )
        self._scaling_writer = csv.DictWriter(
            self._scaling_file,
            fieldnames=["tokens_seen", "train_loss"],
        )
        self._grad_writer = csv.DictWriter(
            self._grad_file,
            fieldnames=["step", "layer_name", "grad_norm"],
        )
        self._ckpt_writer = csv.DictWriter(
            self._ckpt_file,
            fieldnames=["step", "path", "tokens_seen", "loss"],
        )

        for writer in [
            self._train_writer, self._hw_writer, self._val_writer,
            self._scaling_writer, self._grad_writer, self._ckpt_writer,
        ]:
            writer.writeheader()

        for f in [self._train_file, self._hw_file, self._val_file,
                  self._scaling_file, self._grad_file, self._ckpt_file]:
            f.flush()

        # In-memory accumulators for summary
        self._train_losses: list[float] = []
        self._throughputs: list[float] = []
        self._hw_snapshots: list[dict] = []
        self._val_rows: list[dict] = []
        self._scaling_rows: list[dict] = []

        # Save experiment metadata
        self._save_metadata()

    def _save_metadata(self) -> None:
        meta = {
            "experiment_name": "adam_poc_pretrain",
            "start_time": self.start_time.isoformat(),
            "config": asdict(self.config),
        }
        with open(self.metrics_dir / "experiment_metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

    def log_step(
        self,
        step: int,
        loss: float,
        lr: float,
        grad_norm: float,
        tokens_seen: int,
        throughput: float,
        hw: dict,
    ) -> None:
        ts = datetime.now().isoformat()
        ppl = math.exp(min(loss, 20))  # cap to avoid overflow

        self._train_writer.writerow({
            "step": step,
            "timestamp": ts,
            "loss": f"{loss:.6f}",
            "perplexity": f"{ppl:.2f}",
            "lr": f"{lr:.2e}",
            "grad_norm": f"{grad_norm:.4f}",
            "tokens_seen": tokens_seen,
            "throughput_tok_s": f"{throughput:.0f}",
        })
        self._train_file.flush()
        self._train_losses.append(loss)
        self._throughputs.append(throughput)

        if hw and "temp_c" in hw:
            self._hw_writer.writerow({
                "step": step,
                "timestamp": ts,
                "temp_c": hw.get("temp_c", ""),
                "power_draw_w": hw.get("power_draw_w", ""),
                "power_limit_w": hw.get("power_limit_w", ""),
                "vram_used_gb": f"{hw.get('vram_used_gb', 0):.2f}",
                "vram_total_gb": f"{hw.get('vram_total_gb', 0):.2f}",
                "gpu_util_pct": hw.get("gpu_util_pct", ""),
                "throttling": int(hw.get("throttling", False)),
            })
            self._hw_file.flush()
            self._hw_snapshots.append(hw)

    def log_val(self, step: int, tokens_seen: int, val_loss: float, val_ppl: float) -> None:
        row = {
            "step": step,
            "tokens_seen": tokens_seen,
            "val_loss": f"{val_loss:.6f}",
            "val_perplexity": f"{val_ppl:.2f}",
        }
        self._val_writer.writerow(row)
        self._val_file.flush()
        self._val_rows.append(row)

    def log_scaling_point(self, tokens_seen: int, train_loss: float) -> None:
        row = {"tokens_seen": tokens_seen, "train_loss": f"{train_loss:.6f}"}
        self._scaling_writer.writerow(row)
        self._scaling_file.flush()
        self._scaling_rows.append(row)

    def log_grad_flow(self, step: int, norms: dict[str, float]) -> None:
        for layer_name, grad_norm in norms.items():
            self._grad_writer.writerow({
                "step": step,
                "layer_name": layer_name,
                "grad_norm": f"{grad_norm:.6f}",
            })
        self._grad_file.flush()

    def log_checkpoint(self, step: int, path: str, tokens_seen: int, loss: float) -> None:
        self._ckpt_writer.writerow({
            "step": step,
            "path": path,
            "tokens_seen": tokens_seen,
            "loss": f"{loss:.6f}",
        })
        self._ckpt_file.flush()

    def recent_throughput(self, window: int = 100) -> float:
        if not self._throughputs:
            return 0.0
        recent = self._throughputs[-window:]
        return sum(recent) / len(recent)

    def save_summary(self, final_step: int, final_loss: float, total_tokens: int) -> None:
        """Write summary_statistics.json and paper_tables.tex."""
        end_time = datetime.now()
        total_time_h = (end_time - self.start_time).total_seconds() / 3600.0

        hw = self._hw_snapshots
        summary = {
            "experiment": "adam_poc_pretrain",
            "final_step": final_step,
            "final_loss": final_loss,
            "final_perplexity": math.exp(min(final_loss, 20)),
            "total_tokens_trained": total_tokens,
            "total_time_hours": total_time_h,
            "avg_throughput_tok_s": self.recent_throughput(len(self._throughputs)),
            "hardware": {
                "avg_temp_c": _mean([s["temp_c"] for s in hw if "temp_c" in s]),
                "max_temp_c": max((s["temp_c"] for s in hw if "temp_c" in s), default=0),
                "avg_power_w": _mean([s["power_draw_w"] for s in hw if "power_draw_w" in s]),
                "avg_vram_gb": _mean([s["vram_used_gb"] for s in hw if "vram_used_gb" in s]),
                "avg_gpu_util_pct": _mean([s["gpu_util_pct"] for s in hw if "gpu_util_pct" in s]),
            },
        }
        with open(self.metrics_dir / "summary_statistics.json", "w") as f:
            json.dump(summary, f, indent=2)

        self._write_latex_tables(summary, total_tokens)
        self._close_files()
        print(f"\n  Paper metrics saved to {self.metrics_dir}")

    def _write_latex_tables(self, summary: dict, total_tokens: int) -> None:
        """Generate 3 LaTeX tables for the research paper."""
        lines = []
        lines.append("% Auto-generated by pretrain_adam_poc.py")
        lines.append(f"% Generated: {datetime.now().isoformat()}")
        lines.append("")

        # ── Table 1: Training Dynamics ────────────────────────────────────
        lines.append(r"\begin{table}[h]")
        lines.append(r"\centering")
        lines.append(r"\caption{Training Dynamics — Adam PoC 494M}")
        lines.append(r"\label{tab:training-dynamics}")
        lines.append(r"\begin{tabular}{lllllr}")
        lines.append(r"\hline")
        lines.append(r"Tokens (B) & Train Loss & Val Loss & Val PPL & LR & Throughput (tok/s) \\")
        lines.append(r"\hline")

        # Sample at 10%, 25%, 50%, 75%, 100%
        milestones = [0.10, 0.25, 0.50, 0.75, 1.00]
        for pct in milestones:
            target_tok = int(total_tokens * pct)
            # Find closest scaling row
            row_s = _closest_row(self._scaling_rows, "tokens_seen", target_tok)
            row_v = _closest_row(self._val_rows, "tokens_seen", target_tok)
            tok_b = target_tok / 1e9
            t_loss = row_s["train_loss"] if row_s else "—"
            v_loss = row_v["val_loss"] if row_v else "—"
            v_ppl = row_v["val_perplexity"] if row_v else "—"
            lines.append(
                f"{tok_b:.1f} & {t_loss} & {v_loss} & {v_ppl} & — & — \\\\"
            )

        lines.append(r"\hline")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table}")
        lines.append("")

        # ── Table 2: Hardware Utilization ─────────────────────────────────
        hw = self._hw_snapshots
        n = len(hw)
        if n >= 3:
            warmup_hw = hw[: n // 10]
            final_hw = hw[-n // 10 :]
            main_hw = hw[n // 10 : -n // 10]
        else:
            warmup_hw = main_hw = final_hw = hw

        def hw_row(label: str, snaps: list[dict]) -> str:
            if not snaps:
                return f"{label} & — & — & — & — & — \\\\"
            avg_t = _mean([s.get("temp_c", 0) for s in snaps])
            max_t = max(s.get("temp_c", 0) for s in snaps)
            avg_p = _mean([s.get("power_draw_w", 0) for s in snaps])
            avg_v = _mean([s.get("vram_used_gb", 0) for s in snaps])
            avg_u = _mean([s.get("gpu_util_pct", 0) for s in snaps])
            return (
                f"{label} & {avg_t:.1f} & {max_t:.1f} & "
                f"{avg_p:.0f} & {avg_v:.1f} & {avg_u:.1f} \\\\"
            )

        lines.append(r"\begin{table}[h]")
        lines.append(r"\centering")
        lines.append(r"\caption{Hardware Utilization — H100 SXM5 80GB}")
        lines.append(r"\label{tab:hardware}")
        lines.append(r"\begin{tabular}{llllll}")
        lines.append(r"\hline")
        lines.append(r"Stage & Avg Temp (°C) & Max Temp (°C) & Avg Power (W) & Avg VRAM (GB) & Avg GPU Util (\%) \\")
        lines.append(r"\hline")
        lines.append(hw_row("Warmup", warmup_hw))
        lines.append(hw_row("Main training", main_hw))
        lines.append(hw_row("Final 10\\%", final_hw))
        lines.append(r"\hline")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table}")
        lines.append("")

        # ── Table 3: Scaling Law Fit ──────────────────────────────────────
        # Fit L = a + b * tokens^c using log-space linear regression
        fit_info = "Insufficient data for fit"
        if len(self._scaling_rows) >= 5:
            try:
                import numpy as np
                xs = np.array([float(r["tokens_seen"]) for r in self._scaling_rows])
                ys = np.array([float(r["train_loss"]) for r in self._scaling_rows])
                log_x = np.log(xs)
                log_y = np.log(ys)
                # log(L) = log(A) + alpha * log(D)  →  linear fit
                coeffs = np.polyfit(log_x, log_y, 1)
                alpha = coeffs[0]
                log_A = coeffs[1]
                A = math.exp(log_A)
                r_sq = float(np.corrcoef(log_x, log_y)[0, 1] ** 2)
                fit_info = f"L = {A:.4f} × D^{{{alpha:.4f}}}, R²={r_sq:.4f}"
            except Exception:
                pass

        lines.append(r"\begin{table}[h]")
        lines.append(r"\centering")
        lines.append(r"\caption{Scaling Law Fit — Loss vs. Tokens Seen}")
        lines.append(r"\label{tab:scaling}")
        lines.append(r"\begin{tabular}{ll}")
        lines.append(r"\hline")
        lines.append(r"Fit & Parameters \\")
        lines.append(r"\hline")
        lines.append(f"Power law (log-log linear) & {fit_info} \\\\")
        lines.append(r"\hline")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table}")

        with open(self.metrics_dir / "paper_tables.tex", "w") as f:
            f.write("\n".join(lines) + "\n")

    def _close_files(self) -> None:
        for f in [
            self._train_file, self._hw_file, self._val_file,
            self._scaling_file, self._grad_file, self._ckpt_file,
        ]:
            try:
                f.close()
            except Exception:
                pass


def _mean(values: list) -> float:
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else 0.0


def _closest_row(rows: list[dict], key: str, target: int) -> dict | None:
    if not rows:
        return None
    return min(rows, key=lambda r: abs(int(r[key]) - target))


# ─────────────────────────────────────────────────────────────────────────────
# Section 7: Optimizer and LR Schedule
# ─────────────────────────────────────────────────────────────────────────────

def build_optimizer(model: nn.Module, config: PretrainConfig):
    """8-bit AdamW (bitsandbytes). Falls back to standard AdamW if bnb unavailable."""
    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
            betas=(config.beta1, config.beta2),
        )
        print("  Optimizer: 8-bit AdamW (bitsandbytes)")
    except ImportError:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
            betas=(config.beta1, config.beta2),
        )
        print("  Optimizer: standard AdamW (bitsandbytes not available)")
    return optimizer


def build_lr_scheduler(optimizer, total_steps: int, warmup_steps: int, config: PretrainConfig):
    """
    Linear warmup (5% of total_steps) → cosine decay to lr_min_ratio × peak.
    Does NOT decay to zero — floors at 0.1 × peak.
    """
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return config.lr_min_ratio + (1.0 - config.lr_min_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ─────────────────────────────────────────────────────────────────────────────
# Section 8: Gradient Flow Analysis
# ─────────────────────────────────────────────────────────────────────────────

def get_layer_grad_norms(model: nn.Module) -> dict[str, float]:
    """
    Returns per-layer gradient L2 norms. Called every grad_flow_every steps.
    Prints warnings for exploding (>10) or vanishing (<1e-6) gradients.
    """
    norms: dict[str, float] = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            norm = param.grad.norm().item()
            norms[name] = norm
            if norm > 10.0:
                print(f"  WARNING grad exploding: {name} = {norm:.2f}", flush=True)
            elif norm < 1e-6 and norm > 0:
                print(f"  WARNING grad vanishing: {name} = {norm:.2e}", flush=True)
    return norms


# ─────────────────────────────────────────────────────────────────────────────
# Section 9: Evaluation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model: nn.Module,
    val_dataset: PretrainingDataset,
    config: PretrainConfig,
    max_batches: int = 50,
) -> tuple[float, float]:
    """Returns (val_loss, val_perplexity) over up to max_batches validation batches."""
    model.eval()
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, num_workers=0)
    total_loss = 0.0
    n_batches = 0

    for batch in val_loader:
        if n_batches >= max_batches:
            break
        input_ids = batch["input_ids"].cuda()
        labels = batch["labels"].cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=config.bf16):
            outputs = model(input_ids=input_ids, labels=labels)
        total_loss += outputs.loss.item()
        n_batches += 1

    model.train()
    if n_batches == 0:
        return float("inf"), float("inf")
    avg_loss = total_loss / n_batches
    return avg_loss, math.exp(min(avg_loss, 20))


# ─────────────────────────────────────────────────────────────────────────────
# Section 10: Checkpoint I/O
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(
    model: nn.Module,
    optimizer,
    scheduler,
    step: int,
    tokens_seen: int,
    loss: float,
    config: PretrainConfig,
    metrics: PretrainMetrics,
    output_dir: str,
) -> None:
    ckpt_dir = Path(output_dir) / f"checkpoint-{step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Save model weights (full precision for future loading)
    model.save_pretrained(str(ckpt_dir))

    # Save training state
    torch.save(
        {
            "step": step,
            "tokens_seen": tokens_seen,
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "loss": loss,
            "config": asdict(config),
        },
        ckpt_dir / "trainer_state.pt",
    )

    metrics.log_checkpoint(step, str(ckpt_dir), tokens_seen, loss)
    print(f"  Checkpoint saved → {ckpt_dir}", flush=True)

    # Prune old checkpoints (keep last N)
    _prune_checkpoints(output_dir, config.keep_checkpoints)


def load_checkpoint(
    resume_from: str,
    model: nn.Module,
    optimizer,
    scheduler,
) -> tuple[int, int]:
    """Returns (step, tokens_seen) to resume from."""
    ckpt_dir = Path(resume_from)
    state_path = ckpt_dir / "trainer_state.pt"
    if not state_path.exists():
        raise FileNotFoundError(f"No trainer_state.pt in {ckpt_dir}")

    state = torch.load(state_path, map_location="cpu")
    optimizer.load_state_dict(state["optimizer_state"])
    scheduler.load_state_dict(state["scheduler_state"])
    step = state["step"]
    tokens_seen = state["tokens_seen"]

    # Load model weights
    from transformers import Qwen2ForCausalLM
    loaded = Qwen2ForCausalLM.from_pretrained(str(ckpt_dir))
    model.load_state_dict(loaded.state_dict())

    print(f"  Resumed from step {step}, tokens_seen={tokens_seen/1e9:.2f}B")
    return step, tokens_seen


def _prune_checkpoints(output_dir: str, keep: int) -> None:
    """Delete oldest checkpoints, keeping the most recent `keep` checkpoints."""
    ckpt_dirs = sorted(
        Path(output_dir).glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[1]),
    )
    to_delete = ckpt_dirs[:-keep] if len(ckpt_dirs) > keep else []
    for old in to_delete:
        try:
            shutil.rmtree(old)
            print(f"  Pruned old checkpoint: {old.name}")
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Section 11: Argument Parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> PretrainConfig:
    parser = argparse.ArgumentParser(
        description="Adam PoC 494M — from-scratch pretraining"
    )

    # Architecture
    parser.add_argument("--n-layers", type=int, default=24)
    parser.add_argument("--d-model", type=int, default=896)
    parser.add_argument("--n-heads", type=int, default=14)
    parser.add_argument("--n-kv-heads", type=int, default=2)
    parser.add_argument("--d-ff", type=int, default=4864)
    parser.add_argument("--rope-theta", type=float, default=10000.0)
    parser.add_argument("--vocab-size", type=int, default=151936)
    parser.add_argument("--seq-len", type=int, default=2048,
                        dest="max_seq_len")

    # Training
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--total-tokens", type=int, default=6_000_000_000)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    # Optimizer
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lr-min-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)

    # Data
    parser.add_argument("--data", type=str, default="hope/adam_training_data/pretrain_corpus",
                        dest="data_path")
    parser.add_argument("--val-data", type=str,
                        default="hope/adam_training_data/pretrain_val.jsonl",
                        dest="val_path")
    parser.add_argument("--tokenizer", type=str, default="Qwen/Qwen2.5-0.5B",
                        dest="tokenizer_name")

    # Output
    parser.add_argument("--output", type=str, default="adam_poc_checkpoints",
                        dest="output_dir")
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--keep-checkpoints", type=int, default=3)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--val-every", type=int, default=5000)
    parser.add_argument("--grad-flow-every", type=int, default=200)
    parser.add_argument("--scaling-every", type=int, default=1000)

    # Resume
    parser.add_argument("--resume", type=str, default="", dest="resume_from")

    args = parser.parse_args()

    return PretrainConfig(
        n_layers=args.n_layers,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        vocab_size=args.vocab_size,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        total_tokens=args.total_tokens,
        grad_clip=args.grad_clip,
        lr=args.lr,
        lr_min_ratio=args.lr_min_ratio,
        weight_decay=args.weight_decay,
        beta1=args.beta1,
        beta2=args.beta2,
        warmup_ratio=args.warmup_ratio,
        data_path=args.data_path,
        val_path=args.val_path,
        tokenizer_name=args.tokenizer_name,
        output_dir=args.output_dir,
        save_every=args.save_every,
        keep_checkpoints=args.keep_checkpoints,
        log_every=args.log_every,
        val_every=args.val_every,
        grad_flow_every=args.grad_flow_every,
        scaling_every=args.scaling_every,
        resume_from=args.resume_from,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Section 12: Main Training Loop
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    config = parse_args()

    # ── Setup ──────────────────────────────────────────────────────────────
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Adam PoC 494M — From-Scratch Pretraining")
    print("=" * 70)

    # Enable TF32 for faster matmul on Ampere+
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print("\nBuilding model...")
    model = build_model(config).cuda()
    apply_selective_gc(model, every_n=4)

    n_params = count_parameters(model)
    print(f"  Parameters: {n_params / 1e6:.1f}M")
    print(f"  Architecture: {config.n_layers}L / {config.d_model}d / "
          f"{config.n_heads}Q / {config.n_kv_heads}KV / {config.d_ff}FF")

    # ── Tokenizer & Datasets ───────────────────────────────────────────────
    print(f"\nLoading tokenizer: {config.tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)

    print(f"Loading training data: {config.data_path}")
    train_dataset = PretrainingDataset(
        config.data_path, tokenizer, config.max_seq_len, seed=42
    )
    print(f"Loading validation data: {config.val_path}")
    val_dataset = PretrainingDataset(
        config.val_path, tokenizer, config.max_seq_len, seed=0
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        num_workers=4,
        pin_memory=True,
    )

    # ── Training schedule ─────────────────────────────────────────────────
    tokens_per_step = config.batch_size * config.max_seq_len
    total_steps = config.total_tokens // tokens_per_step
    warmup_steps = int(total_steps * config.warmup_ratio)

    def _fmt_tok(n: int) -> str:
        return f"{n/1e9:.2f}B" if n >= 1_000_000_000 else f"{n/1e6:.1f}M"

    print(f"\nTraining schedule:")
    print(f"  Total tokens:  {_fmt_tok(config.total_tokens)}")
    print(f"  Tokens/step:   {tokens_per_step:,}")
    print(f"  Total steps:   {total_steps:,}")
    print(f"  Warmup steps:  {warmup_steps:,}")
    print(f"  Peak LR:       {config.lr:.1e}")
    print(f"  Min LR:        {config.lr * config.lr_min_ratio:.1e}")
    print(f"  Checkpoints:   every {config.save_every} steps (keep {config.keep_checkpoints})")

    # ── Optimizer & Scheduler ─────────────────────────────────────────────
    print("\nBuilding optimizer...")
    optimizer = build_optimizer(model, config)
    scheduler = build_lr_scheduler(optimizer, total_steps, warmup_steps, config)

    # ── Monitoring & Metrics ──────────────────────────────────────────────
    hw_monitor = HardwareMonitor()
    metrics = PretrainMetrics(config.output_dir, config)
    print(f"\nPaper metrics → {config.output_dir}/paper_metrics/")

    # ── Resume from checkpoint ────────────────────────────────────────────
    step = 0
    tokens_seen = 0
    if config.resume_from:
        print(f"\nResuming from: {config.resume_from}")
        step, tokens_seen = load_checkpoint(
            config.resume_from, model, optimizer, scheduler
        )

    # ── Initial VRAM check ────────────────────────────────────────────────
    hw_init = hw_monitor.get_metrics()
    if hw_init and "vram_used_gb" in hw_init:
        print(f"\nInitial VRAM: {hw_init['vram_used_gb']:.1f} GB / "
              f"{hw_init['vram_total_gb']:.1f} GB")
        print(f"GPU temp:     {hw_init['temp_c']}°C")

    # ── Training Loop ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Starting training")
    print("=" * 70)

    model.train()
    t_window_start = time.time()
    loss_window: list[float] = []

    try:
        for batch in train_loader:
            if tokens_seen >= config.total_tokens:
                break

            input_ids = batch["input_ids"].cuda()
            labels = batch["labels"].cuda()

            # Forward pass (BF16 — no GradScaler needed for BF16)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=config.bf16):
                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs.loss

            # Backward pass
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.grad_clip
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            step += 1
            tokens_seen += tokens_per_step
            loss_window.append(loss.item())

            # ── Log every log_every steps ──────────────────────────────
            if step % config.log_every == 0:
                elapsed = time.time() - t_window_start
                throughput = (config.log_every * tokens_per_step) / max(elapsed, 1e-6)
                avg_loss = sum(loss_window) / len(loss_window)
                loss_window.clear()

                current_lr = scheduler.get_last_lr()[0]
                hw = hw_monitor.get_metrics()
                hw_monitor.check_and_warn(hw, step)

                metrics.log_step(
                    step, avg_loss, current_lr, grad_norm.item(),
                    tokens_seen, throughput, hw,
                )

                tok_str = (
                    f"{tokens_seen/1e9:.3f}B" if tokens_seen >= 1e9
                    else f"{tokens_seen/1e6:.1f}M"
                )
                print(
                    f"[{step:>7}] loss={avg_loss:.4f}  ppl={math.exp(min(avg_loss,20)):.1f}"
                    f"  lr={current_lr:.2e}  gnorm={grad_norm:.3f}"
                    f"  tok={tok_str}  {throughput:.0f}tok/s"
                    + (f"  {hw['temp_c']:.0f}°C/{hw['power_draw_w']:.0f}W"
                       if hw and "temp_c" in hw else ""),
                    flush=True,
                )
                t_window_start = time.time()

            # ── Gradient flow every grad_flow_every steps ──────────────
            if step % config.grad_flow_every == 0:
                norms = get_layer_grad_norms(model)
                metrics.log_grad_flow(step, norms)

            # ── Scaling law point every scaling_every steps ────────────
            if step % config.scaling_every == 0:
                metrics.log_scaling_point(tokens_seen, loss.item())

            # ── Validation every val_every steps ───────────────────────
            if step % config.val_every == 0:
                print(f"\n[{step}] Running validation...", flush=True)
                val_loss, val_ppl = evaluate(model, val_dataset, config)
                metrics.log_val(step, tokens_seen, val_loss, val_ppl)
                print(
                    f"[{step:>7}] VAL loss={val_loss:.4f}  ppl={val_ppl:.1f}"
                    f"  tok={_fmt_tok(tokens_seen)}\n",
                    flush=True,
                )
                model.train()

            # ── Checkpoint every save_every steps ──────────────────────
            if step % config.save_every == 0:
                save_checkpoint(
                    model, optimizer, scheduler, step, tokens_seen,
                    loss.item(), config, metrics, config.output_dir,
                )

            # ── Throughput fallback warning every 1000 steps ───────────
            if step % 1000 == 0 and step > 0:
                recent = metrics.recent_throughput(100)
                if 0 < recent < 80_000:
                    print(
                        f"[{step}] WARNING: throughput {recent:.0f} tok/s < 80k target. "
                        f"Consider --seq-len 1536 fallback.",
                        flush=True,
                    )

    except KeyboardInterrupt:
        print("\nTraining interrupted by user — saving final checkpoint...")

    # ── Final checkpoint and summary ──────────────────────────────────────
    final_loss = loss.item() if "loss" in dir() else float("inf")
    save_checkpoint(
        model, optimizer, scheduler, step, tokens_seen,
        final_loss, config, metrics, config.output_dir,
    )
    metrics.save_summary(step, final_loss, tokens_seen)

    print("\n" + "=" * 70)
    print(f"Pretraining complete.")
    print(f"  Steps:          {step:,}")
    print(f"  Tokens trained: {_fmt_tok(tokens_seen)} / {_fmt_tok(config.total_tokens)}")
    print(f"  Final loss:     {final_loss:.4f}")
    print(f"  Checkpoints:    {config.output_dir}/")
    print(f"  Paper metrics:  {config.output_dir}/paper_metrics/")
    print("=" * 70)


if __name__ == "__main__":
    main()
