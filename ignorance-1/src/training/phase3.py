from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from src.models.jepa import JEPAConfig, JEPAModel
from src.utils.data import SimpleTokenizer, multi_step_tasks
from src.utils.retrieval import VectorIndex


class CEMPlanner:
    def __init__(self, horizon: int, num_samples: int, num_elites: int, num_iterations: int, embed_dim: int):
        self.horizon = horizon
        self.num_samples = num_samples
        self.num_elites = num_elites
        self.num_iterations = num_iterations
        self.embed_dim = embed_dim

    def optimize(self, cost_fn):
        mean = torch.zeros(self.horizon, self.embed_dim)
        std = torch.ones(self.horizon, self.embed_dim)
        best_actions = mean.clone()
        best_cost = float("inf")
        energy_trace: list[float] = []
        for _ in range(self.num_iterations):
            samples = mean.unsqueeze(0) + std.unsqueeze(0) * torch.randn(self.num_samples, self.horizon, self.embed_dim)
            costs = torch.tensor([cost_fn(sample) for sample in samples], dtype=torch.float32)
            elite_idx = torch.topk(-costs, self.num_elites).indices
            elites = samples[elite_idx]
            mean = elites.mean(dim=0)
            std = elites.std(dim=0).clamp_min(1e-3)
            current_best = int(torch.argmin(costs).item())
            if float(costs[current_best].item()) < best_cost:
                best_cost = float(costs[current_best].item())
                best_actions = samples[current_best]
            energy_trace.append(best_cost)
        return best_actions, best_cost, energy_trace


def run_phase3(config, device: str) -> dict:
    seq_len = 128
    tokenizer = SimpleTokenizer(vocab_size=4096)
    model = JEPAModel(JEPAConfig(max_seq_len=seq_len, predictor_layers=3, predictor_heads=6)).to(device)
    tasks = multi_step_tasks()[: config.tasks]

    doc_map: dict[str, str] = {}
    for task in tasks:
        for doc_id in task.solution_docs:
            doc_map[doc_id] = doc_id.replace("_", " ")

    with torch.no_grad():
        doc_ids = list(doc_map.keys())
        doc_embeddings = []
        for doc_id in doc_ids:
            tensor = tokenizer.batch_encode([doc_map[doc_id]], seq_len, device)
            doc_embeddings.append(model.encode(tensor).squeeze(0).cpu())
    index = VectorIndex(doc_ids, torch.stack(doc_embeddings, dim=0))
    planner = CEMPlanner(
        horizon=config.horizon,
        num_samples=config.num_samples,
        num_elites=config.num_elites,
        num_iterations=config.num_iterations,
        embed_dim=model.config.embed_dim,
    )

    successes = 0
    monotonic = 0
    traces = []

    for task in tasks:
        current = tokenizer.batch_encode([task.question], seq_len, device)
        target = tokenizer.batch_encode([task.answer], seq_len, device)
        with torch.no_grad():
            z0 = model.encode(current).squeeze(0).cpu()
            zg = model.encode(target).squeeze(0).cpu()

        def cost_fn(action_sequence: torch.Tensor) -> float:
            z = z0.clone()
            for step in range(action_sequence.shape[0]):
                query = F.normalize(action_sequence[step].unsqueeze(0), dim=-1)
                retrieved = index.search(query, k=1).embeddings.squeeze(0)
                z = 0.5 * z + 0.5 * retrieved
            return float(F.mse_loss(z, zg).item())

        best_actions, _, energy_trace = planner.optimize(cost_fn)
        traces.append(energy_trace)
        if all(energy_trace[idx] <= energy_trace[idx - 1] + 1e-6 for idx in range(1, len(energy_trace))):
            monotonic += 1

        retrieved_ids = []
        for step in range(best_actions.shape[0]):
            result = index.search(F.normalize(best_actions[step].unsqueeze(0), dim=-1), k=1)
            retrieved_ids.append(result.ids[0])
        if any(doc_id in retrieved_ids for doc_id in task.solution_docs):
            successes += 1

    success_rate = successes / max(len(tasks), 1)
    monotonic_fraction = monotonic / max(len(tasks), 1)
    return {
        "planning_success_rate": success_rate,
        "passes": success_rate >= (2.0 / 3.0) and monotonic_fraction >= 0.9,
        "monotonic_energy_fraction": monotonic_fraction,
        "energy_traces": traces,
    }