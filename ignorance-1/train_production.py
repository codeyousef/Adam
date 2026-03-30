from __future__ import annotations
import copy
import random
import torch
import torch.nn.functional as F
import time
import argparse
import yaml
import math
from pathlib import Path
from tqdm import tqdm
import sys

# Add project root to sys.path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.jepa import JEPAConfig, JEPAModel, approximate_model_params
from src.losses.alignment import (
    ignorance_penalty,
    momentum_queue_contrastive_loss,
    paired_alignment_loss,
    prototype_alignment_loss,
    retrieval_margin_loss,
    pairwise_similarity_penalty,
    retrieval_vicreg_loss,
)
from src.utils.data import (
    RetrievalTrainingExample,
    SimpleTokenizer,
    make_text_code_pairs,
    make_benchmark_text_code_pairs,
    make_retrieval_training_examples,
    sample_ood_queries,
)
from src.training.phase4 import _proxy_config, _scaled_training_hparams, _update_ema_model, _lr_multiplier
from src.losses.sigreg import sigreg_loss, isotropic_score, collapse_detected, covariance_logdet_loss

class LatentBuffer:
    def __init__(self, size: int, dim: int, device: str):
        self.buffer = torch.zeros(size, dim, device=device)
        self.ptr = 0
        self.size = size
        self.is_full = False

    def push(self, x: torch.Tensor):
        x = x.detach()
        batch = x.shape[0]
        if self.ptr + batch <= self.size:
            self.buffer[self.ptr:self.ptr + batch] = x
            self.ptr += batch
        else:
            remaining = self.size - self.ptr
            self.buffer[self.ptr:] = x[:remaining]
            self.buffer[:batch - remaining] = x[remaining:]
            self.ptr = batch - remaining
            self.is_full = True
        if self.ptr >= self.size:
            self.ptr = 0
            self.is_full = True

    def get(self) -> torch.Tensor:
        return self.buffer if self.is_full else self.buffer[:self.ptr]

    def sample(self, max_items: int) -> torch.Tensor:
        available = self.get()
        if available.shape[0] <= max_items:
            return available
        indices = torch.randperm(available.shape[0], device=available.device)[:max_items]
        return available.index_select(0, indices)

class AttrDict(dict):
    def __getattr__(self, name):
        if name in self:
            return self[name]
        raise AttributeError(f"'AttrDict' object has no attribute '{name}'")


def _build_training_pairs(batch_size: int) -> list[tuple[str, str]]:
    primary_pairs = make_text_code_pairs(repeats=max(batch_size * 32, 512))
    benchmark_pairs = make_benchmark_text_code_pairs(repeats=max(batch_size * 16, 256))
    unique_pairs = list(dict.fromkeys(primary_pairs + benchmark_pairs))
    random.shuffle(unique_pairs)
    return unique_pairs


def _build_training_examples(
    batch_size: int,
    max_hard_negatives_per_example: int,
    use_surface_code_variants: bool,
) -> list[RetrievalTrainingExample]:
    return make_retrieval_training_examples(
        repeats=max(batch_size * 32, 512),
        benchmark_repeats=max(batch_size * 16, 256),
        max_hard_negatives=max(max_hard_negatives_per_example, 1),
        use_surface_code_variants=use_surface_code_variants,
    )


def _build_paraphrase_families(pairs: list[tuple[str, str]]) -> list[tuple[str, list[str]]]:
    families: dict[str, list[str]] = {}
    for text, code in pairs:
        prompts = families.setdefault(code, [])
        if text not in prompts:
            prompts.append(text)
    return [(code, prompts) for code, prompts in families.items() if len(prompts) >= 2]


def _build_paraphrase_families_from_examples(examples: list[RetrievalTrainingExample]) -> list[tuple[str, list[str]]]:
    families: dict[str, list[str]] = {}
    for example in examples:
        prompts = families.setdefault(example.code, [])
        if example.query not in prompts:
            prompts.append(example.query)
    return [(code, prompts) for code, prompts in families.items() if len(prompts) >= 2]


def _build_prompt_family_map(paraphrase_families: list[tuple[str, list[str]]]) -> dict[str, list[str]]:
    return {code: prompts for code, prompts in paraphrase_families}


def _sample_alternate_queries_from_examples(
    examples: list[RetrievalTrainingExample],
    prompt_family_map: dict[str, list[str]],
) -> list[str]:
    alternate_queries: list[str] = []
    for example in examples:
        prompts = prompt_family_map.get(example.code, [])
        candidates = [prompt for prompt in prompts if prompt != example.query]
        alternate_queries.append(random.choice(candidates) if candidates else example.query)
    return alternate_queries


def _sample_batch_pairs(
    pairs: list[tuple[str, str]],
    paraphrase_families: list[tuple[str, list[str]]],
    *,
    batch_size: int,
    step: int,
    paraphrase_batch_probability: float,
) -> list[tuple[str, str]]:
    batch_pairs = [pairs[(step * batch_size + offset) % len(pairs)] for offset in range(batch_size)]
    if batch_size < 2 or not paraphrase_families or random.random() >= paraphrase_batch_probability:
        return batch_pairs

    code, prompts = random.choice(paraphrase_families)
    chosen_prompts = random.sample(prompts, k=2)
    paraphrase_pairs = [(chosen_prompts[0], code), (chosen_prompts[1], code)]
    if batch_size == 2:
        return paraphrase_pairs
    return paraphrase_pairs + batch_pairs[2:batch_size]


def _sample_batch_examples(
    examples: list[RetrievalTrainingExample],
    paraphrase_families: list[tuple[str, list[str]]],
    *,
    batch_size: int,
    step: int,
    paraphrase_batch_probability: float,
) -> list[RetrievalTrainingExample]:
    batch_examples = [examples[(step * batch_size + offset) % len(examples)] for offset in range(batch_size)]
    if batch_size < 2 or not paraphrase_families or random.random() >= paraphrase_batch_probability:
        return batch_examples

    code, prompts = random.choice(paraphrase_families)
    selected_prompts = random.sample(prompts, k=2)
    hard_negatives = batch_examples[0].hard_negatives if batch_examples else []
    paraphrase_examples = [
        RetrievalTrainingExample(query=selected_prompts[0], code=code, hard_negatives=hard_negatives, family="paraphrase"),
        RetrievalTrainingExample(query=selected_prompts[1], code=code, hard_negatives=hard_negatives, family="paraphrase"),
    ]
    if batch_size == 2:
        return paraphrase_examples
    return paraphrase_examples + batch_examples[2:batch_size]


def _ramp_scale(step: int, total_steps: int, ramp_fraction: float) -> float:
    if ramp_fraction <= 0.0:
        return 1.0
    ramp_steps = max(int(round(total_steps * ramp_fraction)), 1)
    return min((step + 1) / ramp_steps, 1.0)

def train_production(config_path: str, size: int, output_path: str, device: str):
    with open(config_path, "r") as f:
        full_config = yaml.safe_load(f)
    
    config = AttrDict(full_config.get("phase4", {}))
    proxy_recipe = config.proxy_recipe
    microbatch_size = max(1, min(getattr(config, "microbatch_size", 1), config.batch_size))
    ood_weight = float(getattr(config, "ood_weight", 0.2))
    clf_weight = float(getattr(config, "clf_weight", 0.25))
    ema_target_decay = float(getattr(config, "ema_target_decay", 0.0))
    alignment_temperature = float(getattr(config, "alignment_temperature", 0.05))
    retrieval_margin = float(getattr(config, "retrieval_margin", 0.2))
    retrieval_margin_weight = float(getattr(config, "retrieval_margin_weight", 0.75))
    spread_weight = float(getattr(config, "spread_weight", 0.1))
    query_spread_weight = float(getattr(config, "query_spread_weight", 0.0))
    pred_spread_weight = float(getattr(config, "pred_spread_weight", 0.0))
    use_vicreg_retrieval = bool(getattr(config, "use_vicreg_retrieval", False))
    vicreg_weight = float(getattr(config, "vicreg_weight", 0.0))
    vicreg_invariance_weight = float(getattr(config, "vicreg_invariance_weight", 1.0))
    vicreg_variance_weight = float(getattr(config, "vicreg_variance_weight", 1.0))
    vicreg_covariance_weight = float(getattr(config, "vicreg_covariance_weight", 1.0))
    vicreg_variance_target = float(getattr(config, "vicreg_variance_target", 1.0))
    vicreg_prediction_weight = float(getattr(config, "vicreg_prediction_weight", 0.0))
    use_query_multiview = bool(getattr(config, "use_query_multiview", False))
    query_multiview_weight = float(getattr(config, "query_multiview_weight", 0.0))
    query_multiview_prediction_weight = float(getattr(config, "query_multiview_prediction_weight", 0.0))
    use_momentum_queue = bool(getattr(config, "use_momentum_queue", False))
    momentum_queue_weight = float(getattr(config, "momentum_queue_weight", 0.0))
    momentum_queue_prediction_weight = float(getattr(config, "momentum_queue_prediction_weight", 0.0))
    momentum_queue_temperature = float(getattr(config, "momentum_queue_temperature", alignment_temperature))
    use_family_prototypes = bool(getattr(config, "use_family_prototypes", False))
    prototype_target = str(getattr(config, "prototype_target", "family")).strip().lower()
    prototype_weight = float(getattr(config, "prototype_weight", 0.0))
    prototype_code_weight = float(getattr(config, "prototype_code_weight", 0.0))
    prototype_prediction_weight = float(getattr(config, "prototype_prediction_weight", 0.0))
    prototype_repulsion_weight = float(getattr(config, "prototype_repulsion_weight", 0.0))
    prototype_temperature = float(getattr(config, "prototype_temperature", 0.1))
    vicreg_queue_samples = int(getattr(config, "vicreg_queue_samples", 0) or 0)
    ignorance_warmup_fraction = float(getattr(config, "ignorance_warmup_fraction", 0.0))
    ignorance_start_step = int(getattr(config, "ignorance_start_step", 0) or 0)
    ignorance_ramp_steps = int(getattr(config, "ignorance_ramp_steps", 0) or 0)
    rank_reg_weight = float(getattr(config, "rank_reg_weight", 0.0))
    rank_reg_eps = float(getattr(config, "rank_reg_eps", 1e-4))
    rank_reg_target = str(getattr(config, "rank_reg_target", "none")).strip().lower()
    rank_reg_targets = {t for t in rank_reg_target.replace("+", ",").split(",") if t}
    if "none" in rank_reg_targets:
        rank_reg_targets = set()
    query_margin_weight = float(getattr(config, "query_margin_weight", 0.0))
    query_margin = float(getattr(config, "query_margin", retrieval_margin))
    stat_buffer_size = int(getattr(config, "stat_buffer_size", 1024) or 1024)
    query_buffer_size = int(getattr(config, "query_buffer_size", 2048) or 2048)
    code_buffer_size = int(getattr(config, "code_buffer_size", 2048) or 2048)
    pred_buffer_size = int(getattr(config, "pred_buffer_size", 2048) or 2048)
    paged_optimizer = bool(getattr(config, "paged_optimizer", False))
    optimizer_name = str(getattr(config, "optimizer", "")).strip().lower()
    max_seq_len_override = int(getattr(config, "max_seq_len", 0) or 0)
    use_retrieval_data_strategy = bool(getattr(config, "use_retrieval_data_strategy", False))
    max_hard_negatives_per_example = int(getattr(config, "max_hard_negatives_per_example", 0) or 0)
    use_surface_code_variants = bool(getattr(config, "use_surface_code_variants", False))
    scheduler = str(getattr(config, "scheduler", "none"))
    warmup_fraction = float(getattr(config, "warmup_fraction", 0.0))
    min_lr_ratio = float(getattr(config, "min_lr_ratio", 0.0))
    ramp_regularizers = bool(getattr(config, "ramp_regularizers", False))
    regularizer_ramp_fraction = float(getattr(config, "regularizer_ramp_fraction", 0.2))
    paraphrase_batch_probability = float(getattr(config, "paraphrase_batch_probability", 0.0))
    num_microbatches_per_step = max((config.batch_size + microbatch_size - 1) // microbatch_size, 1)
    
    model_config = _proxy_config(size, proxy_recipe)
    if max_seq_len_override > 0:
        model_config.max_seq_len = max_seq_len_override
    scaled_steps, scaled_lr, step_mult, lr_div = _scaled_training_hparams(config, size)
    
    print(f"Training production model: {size:,} params (proxy: {approximate_model_params(model_config):,})")
    print(f"Recipe: {proxy_recipe}")
    print(f"Hyperparams: steps={scaled_steps}, lr={scaled_lr:.8f} (step_mult={step_mult:.2f}, lr_div={lr_div:.2f})")
    print(f"Batching: batch_size={config.batch_size}, microbatch_size={microbatch_size}")
    print(f"Sequence length: max_seq_len={model_config.max_seq_len}")
    if not optimizer_name:
        optimizer_name = "paged_adamw8bit" if paged_optimizer else "adamw8bit"
    print(f"Optimizer: {optimizer_name}")
    print(
        "Data strategy: "
        f"use_retrieval_data_strategy={use_retrieval_data_strategy}, "
        f"max_hard_negatives_per_example={max_hard_negatives_per_example}, "
        f"use_surface_code_variants={use_surface_code_variants}"
    )
    print(
        "Schedule: "
        f"scheduler={scheduler}, warmup_fraction={warmup_fraction:.2f}, min_lr_ratio={min_lr_ratio:.2f}, "
        f"ramp_regularizers={ramp_regularizers}, regularizer_ramp_fraction={regularizer_ramp_fraction:.2f}"
    )
    print(f"Batch mining: paraphrase_batch_probability={paraphrase_batch_probability:.2f}")
    print(f"Aux losses: ood_weight={ood_weight:.2f}, clf_weight={clf_weight:.2f}")
    print(
        "Retrieval shaping: "
        f"margin={retrieval_margin:.2f}, margin_weight={retrieval_margin_weight:.2f}, "
        f"spread_weight={spread_weight:.2f}, query_spread_weight={query_spread_weight:.2f}, "
        f"pred_spread_weight={pred_spread_weight:.2f}, ema_target_decay={ema_target_decay:.4f}, "
        f"alignment_temperature={alignment_temperature:.3f}"
    )
    print(
        "Anti-collapse: "
        f"use_vicreg_retrieval={use_vicreg_retrieval}, vicreg_weight={vicreg_weight:.2f}, "
        f"vicreg_prediction_weight={vicreg_prediction_weight:.2f}, queue_samples={vicreg_queue_samples}, "
        f"ignorance_warmup_fraction={ignorance_warmup_fraction:.2f}"
    )
    print(
        "Rank reg: "
        f"weight={rank_reg_weight:.3f}, eps={rank_reg_eps:.1e}, target={rank_reg_target}, "
        f"ignorance_start_step={ignorance_start_step}, ignorance_ramp_steps={ignorance_ramp_steps}"
    )
    print(
        "Query multiview: "
        f"use_query_multiview={use_query_multiview}, query_multiview_weight={query_multiview_weight:.2f}, "
        f"query_multiview_prediction_weight={query_multiview_prediction_weight:.2f}"
    )
    print(
        "Momentum queue: "
        f"use_momentum_queue={use_momentum_queue}, momentum_queue_weight={momentum_queue_weight:.2f}, "
        f"momentum_queue_prediction_weight={momentum_queue_prediction_weight:.2f}, "
        f"momentum_queue_temperature={momentum_queue_temperature:.3f}"
    )
    print(
        "Prototypes: "
        f"use_family_prototypes={use_family_prototypes}, prototype_weight={prototype_weight:.2f}, "
        f"prototype_target={prototype_target}, "
        f"prototype_code_weight={prototype_code_weight:.2f}, "
        f"prototype_prediction_weight={prototype_prediction_weight:.2f}, "
        f"prototype_repulsion_weight={prototype_repulsion_weight:.2f}, "
        f"prototype_temperature={prototype_temperature:.2f}"
    )
    print(f"Device: {device}")
    
    tokenizer = SimpleTokenizer(vocab_size=4096)
    import bitsandbytes as bnb
    model = JEPAModel(model_config).to(device).to(torch.bfloat16)
    target_model = None
    if ema_target_decay > 0.0:
        target_model = copy.deepcopy(model)
        target_model.eval()
        for parameter in target_model.parameters():
            parameter.requires_grad_(False)
    optimizer_map = {
        "adamw8bit": bnb.optim.AdamW8bit,
        "adamw32bit": bnb.optim.AdamW32bit,
        "paged_adamw8bit": bnb.optim.PagedAdamW8bit,
        "paged_adamw32bit": bnb.optim.PagedAdamW32bit,
        "paged_adamw": bnb.optim.PagedAdamW,
        "adamw": bnb.optim.AdamW,
    }
    if optimizer_name not in optimizer_map:
        raise ValueError(
            "Unsupported optimizer "
            f"{optimizer_name!r}. Expected one of: {', '.join(sorted(optimizer_map))}"
        )
    optimizer_cls = optimizer_map[optimizer_name]
    
    # Latent Buffer for stable regularization
    buffer = LatentBuffer(size=stat_buffer_size, dim=model_config.embed_dim, device=device)
    query_buffer = LatentBuffer(size=query_buffer_size, dim=model_config.embed_dim, device=device)
    code_buffer = LatentBuffer(size=code_buffer_size, dim=model_config.embed_dim, device=device)
    pred_buffer = LatentBuffer(size=pred_buffer_size, dim=model_config.embed_dim, device=device)
    
    if use_retrieval_data_strategy:
        training_examples = _build_training_examples(
            config.batch_size,
            max_hard_negatives_per_example,
            use_surface_code_variants,
        )
        paraphrase_families = _build_paraphrase_families_from_examples(training_examples)
        prompt_family_map = _build_prompt_family_map(paraphrase_families)
        if prototype_target == "family":
            prototype_names = sorted({example.family for example in training_examples})
            prototype_key_fn = lambda example: example.family
        elif prototype_target == "code":
            prototype_names = sorted({example.code for example in training_examples})
            prototype_key_fn = lambda example: example.code
        else:
            raise ValueError(f"Unsupported prototype_target {prototype_target!r}. Expected 'family' or 'code'.")
        prototype_to_idx = {name: idx for idx, name in enumerate(prototype_names)}
        pairs = None
    else:
        pairs = _build_training_pairs(config.batch_size)
        training_examples = None
        paraphrase_families = _build_paraphrase_families(pairs)
        prompt_family_map = {}
        prototype_to_idx = {}
        prototype_key_fn = None

    if use_family_prototypes and not use_retrieval_data_strategy:
        raise ValueError("Family prototypes require use_retrieval_data_strategy=True")

    prototype_table = None
    if use_family_prototypes:
        prototype_table = torch.nn.Embedding(len(prototype_to_idx), model_config.embed_dim, device=device)
        torch.nn.init.normal_(prototype_table.weight, mean=0.0, std=0.02)

    optimization_params = list(model.parameters())
    if prototype_table is not None:
        optimization_params.extend(prototype_table.parameters())
    optimizer = optimizer_cls(optimization_params, lr=scaled_lr)
    
    model.train()
    pbar = tqdm(total=scaled_steps, desc="Training")
    
    torch.cuda.reset_peak_memory_stats() if device.startswith("cuda") else None
    start_time = time.time()
    
    for step in range(scaled_steps):
        lr_scale = _lr_multiplier(
            step,
            scaled_steps,
            scheduler=scheduler,
            warmup_fraction=warmup_fraction,
            min_lr_ratio=min_lr_ratio,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = scaled_lr * lr_scale

        regularizer_scale = _ramp_scale(step, scaled_steps, regularizer_ramp_fraction) if ramp_regularizers else 1.0
        current_margin_weight = retrieval_margin_weight * regularizer_scale
        current_spread_weight = spread_weight * regularizer_scale
        current_query_spread_weight = query_spread_weight * regularizer_scale
        current_pred_spread_weight = pred_spread_weight * regularizer_scale
        current_sigreg_weight = 0.5 * regularizer_scale
        current_rank_reg_weight = rank_reg_weight * regularizer_scale
        current_vicreg_weight = vicreg_weight * regularizer_scale
        current_query_margin_weight = query_margin_weight * regularizer_scale
        current_momentum_queue_weight = momentum_queue_weight * regularizer_scale
        current_momentum_queue_prediction_weight = momentum_queue_prediction_weight * regularizer_scale
        current_prototype_weight = prototype_weight * regularizer_scale
        current_prototype_code_weight = prototype_code_weight * regularizer_scale
        current_prototype_prediction_weight = prototype_prediction_weight * regularizer_scale
        current_prototype_repulsion_weight = prototype_repulsion_weight * regularizer_scale
        if ignorance_start_step > 0 and step < ignorance_start_step:
            ignorance_scale = 0.0
        elif ignorance_ramp_steps > 0:
            ramp_step = max(step - ignorance_start_step, 0)
            ignorance_scale = min((ramp_step + 1) / max(ignorance_ramp_steps, 1), 1.0)
        elif ignorance_warmup_fraction > 0.0:
            ignorance_scale = _ramp_scale(step, scaled_steps, ignorance_warmup_fraction)
        else:
            ignorance_scale = 1.0
        current_ood_weight = ood_weight * ignorance_scale
        current_clf_weight = clf_weight * ignorance_scale

        if use_retrieval_data_strategy:
            batch_examples = _sample_batch_examples(
                training_examples,
                paraphrase_families,
                batch_size=config.batch_size,
                step=step,
                paraphrase_batch_probability=paraphrase_batch_probability,
            )
        else:
            batch_pairs = _sample_batch_pairs(
                pairs,
                paraphrase_families,
                batch_size=config.batch_size,
                step=step,
                paraphrase_batch_probability=paraphrase_batch_probability,
            )
        optimizer.zero_grad(set_to_none=True)
        loss_value = 0.0
        num_microbatches = 0

        active_batch = batch_examples if use_retrieval_data_strategy else batch_pairs

        for start in range(0, len(active_batch), microbatch_size):
            if use_retrieval_data_strategy:
                micro_examples = batch_examples[start : start + microbatch_size]
                texts = tokenizer.batch_encode([example.query for example in micro_examples], model_config.max_seq_len, device)
                codes = tokenizer.batch_encode([example.code for example in micro_examples], model_config.max_seq_len, device)
                if use_query_multiview:
                    alt_queries = _sample_alternate_queries_from_examples(micro_examples, prompt_family_map)
                    alt_texts = tokenizer.batch_encode(alt_queries, model_config.max_seq_len, device)
                else:
                    alt_texts = None
                if use_family_prototypes:
                    family_labels = torch.tensor(
                        [prototype_to_idx[prototype_key_fn(example)] for example in micro_examples],
                        device=device,
                        dtype=torch.long,
                    )
                else:
                    family_labels = None
                hard_negative_codes = []
                for example in micro_examples:
                    hard_negative_codes.extend(example.hard_negatives[:max_hard_negatives_per_example])
                hard_negative_codes = list(dict.fromkeys(hard_negative_codes))
            else:
                micro_pairs = batch_pairs[start : start + microbatch_size]
                texts = tokenizer.batch_encode([p[0] for p in micro_pairs], model_config.max_seq_len, device)
                codes = tokenizer.batch_encode([p[1] for p in micro_pairs], model_config.max_seq_len, device)
                alt_texts = None
                family_labels = None
                hard_negative_codes = []

            if use_retrieval_data_strategy:
                ood_count = len(micro_examples)
            else:
                ood_count = len(micro_pairs)
            ood = tokenizer.batch_encode(sample_ood_queries(ood_count), model_config.max_seq_len, device)

            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                z_text = model.encode(texts)
                if alt_texts is not None:
                    z_text_alt = model.encode(alt_texts)
                else:
                    z_text_alt = None
                with torch.no_grad():
                    z_code = target_model.encode(codes) if target_model is not None else model.encode(codes)
                    if hard_negative_codes:
                        hard_negative_tokens = tokenizer.batch_encode(hard_negative_codes, model_config.max_seq_len, device)
                        z_hard_negatives = target_model.encode(hard_negative_tokens) if target_model is not None else model.encode(hard_negative_tokens)
                    else:
                        z_hard_negatives = z_code.new_zeros((0, z_code.shape[-1]))
                z_ood = model.encode(ood)
                z_pred = model.predict(z_text, action_id=1)
                z_ood_pred = model.predict(z_ood, action_id=1)
                coding_logits = model.query_logits(z_text)
                ood_logits = model.query_logits(z_ood)

                if code_buffer.get().numel() and z_hard_negatives.numel():
                    negative_pool = torch.cat([z_hard_negatives, code_buffer.get()], dim=0)
                elif z_hard_negatives.numel():
                    negative_pool = z_hard_negatives
                else:
                    negative_pool = code_buffer.get()

                queue_negatives = code_buffer.sample(vicreg_queue_samples) if vicreg_queue_samples > 0 else code_buffer.get()

                pred_loss, _ = paired_alignment_loss(
                    z_text,
                    z_code,
                    z_pred,
                    negative_pool=negative_pool,
                    temperature=alignment_temperature,
                    prediction_weight=1.25,
                    embedding_weight=0.75,
                    mse_weight=0.15,
                )
                code_candidates = torch.cat([z_code.detach(), negative_pool], dim=0) if negative_pool.numel() else z_code.detach()
                margin_loss = retrieval_margin_loss(z_pred, z_code, negative_pool=negative_pool, margin=retrieval_margin)
                margin_loss = margin_loss + 0.5 * retrieval_margin_loss(z_text, z_code, negative_pool=negative_pool, margin=retrieval_margin)
                if use_vicreg_retrieval and current_vicreg_weight > 0.0:
                    query_queue = query_buffer.sample(vicreg_queue_samples) if vicreg_queue_samples > 0 else query_buffer.get()
                    code_queue = code_buffer.sample(vicreg_queue_samples) if vicreg_queue_samples > 0 else code_buffer.get()
                    vicreg_loss, _ = retrieval_vicreg_loss(
                        z_text,
                        z_code,
                        query_queue=query_queue,
                        positive_queue=code_queue,
                        invariance_weight=vicreg_invariance_weight,
                        variance_weight=vicreg_variance_weight,
                        covariance_weight=vicreg_covariance_weight,
                        variance_target=vicreg_variance_target,
                    )
                    if vicreg_prediction_weight > 0.0:
                        pred_queue = pred_buffer.sample(vicreg_queue_samples) if vicreg_queue_samples > 0 else pred_buffer.get()
                        pred_vicreg_loss, _ = retrieval_vicreg_loss(
                            z_pred,
                            z_code,
                            query_queue=pred_queue,
                            positive_queue=code_queue,
                            invariance_weight=vicreg_invariance_weight,
                            variance_weight=vicreg_variance_weight,
                            covariance_weight=vicreg_covariance_weight,
                            variance_target=vicreg_variance_target,
                        )
                        vicreg_loss = vicreg_loss + vicreg_prediction_weight * pred_vicreg_loss
                    if z_text_alt is not None and query_multiview_weight > 0.0:
                        query_queue = query_buffer.sample(vicreg_queue_samples) if vicreg_queue_samples > 0 else query_buffer.get()
                        query_multiview_loss, _ = retrieval_vicreg_loss(
                            z_text,
                            z_text_alt,
                            query_queue=query_queue,
                            positive_queue=query_queue,
                            invariance_weight=vicreg_invariance_weight,
                            variance_weight=vicreg_variance_weight,
                            covariance_weight=vicreg_covariance_weight,
                            variance_target=vicreg_variance_target,
                        )
                        vicreg_loss = vicreg_loss + query_multiview_weight * query_multiview_loss
                    if z_text_alt is not None and query_multiview_prediction_weight > 0.0:
                        pred_queue = pred_buffer.sample(vicreg_queue_samples) if vicreg_queue_samples > 0 else pred_buffer.get()
                        query_queue = query_buffer.sample(vicreg_queue_samples) if vicreg_queue_samples > 0 else query_buffer.get()
                        pred_query_multiview_loss, _ = retrieval_vicreg_loss(
                            z_pred,
                            z_text_alt,
                            query_queue=pred_queue,
                            positive_queue=query_queue,
                            invariance_weight=vicreg_invariance_weight,
                            variance_weight=vicreg_variance_weight,
                            covariance_weight=vicreg_covariance_weight,
                            variance_target=vicreg_variance_target,
                        )
                        vicreg_loss = vicreg_loss + query_multiview_prediction_weight * pred_query_multiview_loss
                else:
                    vicreg_loss = z_text.new_tensor(0.0)
                if use_momentum_queue and (current_momentum_queue_weight > 0.0 or current_momentum_queue_prediction_weight > 0.0):
                    queue_text_loss, _ = momentum_queue_contrastive_loss(
                        z_text,
                        z_code,
                        negative_queue=queue_negatives,
                        temperature=momentum_queue_temperature,
                    )
                    queue_pred_loss, _ = momentum_queue_contrastive_loss(
                        z_pred,
                        z_code,
                        negative_queue=queue_negatives,
                        temperature=momentum_queue_temperature,
                    )
                    momentum_queue_loss = (
                        current_momentum_queue_weight * queue_text_loss
                        + current_momentum_queue_prediction_weight * queue_pred_loss
                    )
                else:
                    momentum_queue_loss = z_text.new_tensor(0.0)
                if prototype_table is not None and family_labels is not None:
                    prototype_text_loss, _ = prototype_alignment_loss(
                        z_text,
                        prototype_table.weight,
                        family_labels,
                        temperature=prototype_temperature,
                    )
                    prototype_code_loss, _ = prototype_alignment_loss(
                        z_code,
                        prototype_table.weight,
                        family_labels,
                        temperature=prototype_temperature,
                    )
                    prototype_pred_loss, _ = prototype_alignment_loss(
                        z_pred,
                        prototype_table.weight,
                        family_labels,
                        temperature=prototype_temperature,
                    )
                    prototype_repulsion_loss = pairwise_similarity_penalty(prototype_table.weight, max_samples=256)
                    prototype_loss = (
                        current_prototype_weight * prototype_text_loss
                        + current_prototype_code_weight * prototype_code_loss
                        + current_prototype_prediction_weight * prototype_pred_loss
                        + current_prototype_repulsion_weight * prototype_repulsion_loss
                    )
                else:
                    prototype_loss = z_text.new_tensor(0.0)
                ignorance_loss = ignorance_penalty(z_ood, code_candidates) + ignorance_penalty(z_ood_pred, code_candidates)
                clf_loss = F.binary_cross_entropy_with_logits(coding_logits, torch.ones_like(coding_logits))
                clf_loss = clf_loss + F.binary_cross_entropy_with_logits(ood_logits, torch.zeros_like(ood_logits))

                # Use buffer to get a broader estimate of the distribution.
                buffer.push(z_text)
                buffer.push(z_code)
                query_buffer.push(z_text)
                if z_text_alt is not None:
                    buffer.push(z_text_alt)
                    query_buffer.push(z_text_alt)
                code_buffer.push(z_code)
                pred_buffer.push(z_pred)

                z_pool = buffer.get()
                if z_pool.shape[0] >= 128:
                    reg_loss = sigreg_loss(z_pool.unsqueeze(1), m=1024, lambda_reg=max(current_sigreg_weight, 1e-4))
                    spread_loss = pairwise_similarity_penalty(code_buffer.get(), max_samples=128)
                    query_spread_loss = pairwise_similarity_penalty(query_buffer.get(), max_samples=128)
                    pred_spread_loss = pairwise_similarity_penalty(pred_buffer.get(), max_samples=128)
                    rank_reg_loss = z_text.new_tensor(0.0)
                    if current_rank_reg_weight > 0.0 and rank_reg_targets:
                        rank_targets = rank_reg_targets
                        if "all" in rank_targets:
                            rank_targets = {"code", "query", "pred"}
                        rank_components = []
                        if "both" in rank_targets:
                            rank_targets = (rank_targets - {"both"}) | {"code", "query"}
                        if "code" in rank_targets:
                            rank_components.append(covariance_logdet_loss(code_buffer.get(), eps=rank_reg_eps))
                        if "query" in rank_targets:
                            rank_components.append(covariance_logdet_loss(query_buffer.get(), eps=rank_reg_eps))
                        if "pred" in rank_targets:
                            rank_components.append(covariance_logdet_loss(pred_buffer.get(), eps=rank_reg_eps))
                        if rank_components:
                            rank_reg_loss = torch.stack(rank_components).mean()
                    query_margin_loss = z_text.new_tensor(0.0)
                    if z_text_alt is not None and current_query_margin_weight > 0.0:
                        query_margin_loss = retrieval_margin_loss(
                            z_text,
                            z_text_alt,
                            negative_pool=query_buffer.get(),
                            margin=query_margin,
                        )
                    micro_loss = (
                        pred_loss
                        + current_margin_weight * margin_loss
                        + current_vicreg_weight * vicreg_loss
                        + momentum_queue_loss
                        + prototype_loss
                        + current_ood_weight * ignorance_loss
                        + current_clf_weight * clf_loss
                        + current_sigreg_weight * reg_loss
                        + current_rank_reg_weight * rank_reg_loss
                        + current_query_margin_weight * query_margin_loss
                        + current_spread_weight * spread_loss
                        + current_query_spread_weight * query_spread_loss
                        + current_pred_spread_weight * pred_spread_loss
                    )
                else:
                    rank_reg_loss = z_text.new_tensor(0.0)
                    query_margin_loss = z_text.new_tensor(0.0)
                    micro_loss = (
                        pred_loss
                        + current_margin_weight * margin_loss
                        + current_vicreg_weight * vicreg_loss
                        + momentum_queue_loss
                        + prototype_loss
                        + current_ood_weight * ignorance_loss
                        + current_clf_weight * clf_loss
                        + current_rank_reg_weight * rank_reg_loss
                        + current_query_margin_weight * query_margin_loss
                    )

            num_microbatches += 1
            loss_value += float(micro_loss.detach().cpu().item())
            (micro_loss / num_microbatches_per_step).backward()

        optimizer.step()
        if target_model is not None:
            _update_ema_model(target_model, model, ema_target_decay)
        loss = torch.tensor(loss_value / max(num_microbatches, 1), device=device)
        
        pbar.update(1)
        if step % 10 == 0:
            z_stat = buffer.get()
            iso = isotropic_score(z_stat) if z_stat.shape[0] > 4 else 0.0
            pbar.set_postfix({"loss": f"{loss.item():.6f}", "iso": f"{iso:.2f}"})
        
        if (step + 1) % 500 == 0:
            # Save intermediate checkpoint
            torch.save(model.state_dict(), output_path + ".tmp")
            
    pbar.close()
    elapsed = time.time() - start_time
    peak_vram = torch.cuda.max_memory_allocated() / 1e9 if device.startswith("cuda") else 0
    
    print(f"Training complete in {elapsed/60:.2f} minutes.")
    print(f"Peak VRAM: {peak_vram:.2f} GB")
    
    print(f"Saving final model to {output_path}...")
    torch.save(model.state_dict(), output_path)
    if Path(output_path + ".tmp").exists():
        Path(output_path + ".tmp").unlink()
    
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/overnight.yaml")
    parser.add_argument("--size", type=int, default=2700000000)
    parser.add_argument("--output", type=str, default="artifacts/ignorance_1_2.7b.pt")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    
    train_production(args.config, args.size, args.output, args.device)
