#!/usr/bin/env python3
"""
Adam Paper Metrics - Comprehensive data collection for research publication

Captures:
1. Training dynamics (loss curves, gradient norms, learning rates)
2. Probe trajectories (L1-L4 accuracy over steps)
3. CPI analysis (A_CF vs A_P divergence)
4. Sample outputs (model responses at key checkpoints)
5. Ablation data (per-category performance)
6. Statistical summaries (mean, std, confidence intervals)
7. Comparison baselines (before/after, vs base model)
"""

import json
import csv
import time
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, Any
from datetime import datetime
import statistics


@dataclass
class TrainingMetrics:
    """Single training step metrics."""
    step: int
    timestamp: str
    loss: float
    learning_rate: float
    gradient_norm: Optional[float] = None
    epoch: Optional[float] = None
    samples_seen: Optional[int] = None
    tokens_seen: Optional[int] = None
    throughput_tokens_per_sec: Optional[float] = None
    gpu_memory_used_gb: Optional[float] = None
    gpu_utilization_pct: Optional[float] = None


@dataclass
class ProbeMetrics:
    """Validation probe metrics at a checkpoint."""
    step: int
    timestamp: str

    # Level accuracies
    level1_accuracy: float
    level2_accuracy: float
    level3_accuracy: float
    level4_accuracy: float

    # Aggregate metrics
    counterfactual_accuracy: float  # A_CF (L1 + L2)
    overall_accuracy: float

    # Per-probe breakdown
    probe_results: dict = field(default_factory=dict)

    # Sample outputs (for qualitative analysis)
    sample_outputs: list = field(default_factory=list)


@dataclass
class CPIMetrics:
    """Context-Parametric Inversion tracking."""
    step: int
    a_cf: float  # Counterfactual accuracy
    a_p: float   # Parametric/overall accuracy
    a_cf_delta: float  # Change from previous
    a_p_delta: float   # Change from previous
    inversion_detected: bool
    inversion_score: float  # How strong the inversion signal is


@dataclass
class SampleOutput:
    """Captured model output for qualitative analysis."""
    step: int
    probe_name: str
    probe_level: int
    prompt: str
    expected: str
    actual: str
    passed: bool
    score: float
    category: str  # counterfactual_physics, syllogism, constraint, etc.


@dataclass
class AblationMetrics:
    """Per-category performance breakdown for ablation studies."""
    step: int
    category: str
    accuracy: float
    num_samples: int
    avg_score: float
    pass_rate: float


class PaperMetricsCollector:
    """
    Central collector for all paper metrics.

    Outputs:
    - CSV files for easy plotting (loss curves, probe trajectories)
    - JSON files for detailed analysis
    - LaTeX-ready tables
    - Sample outputs for appendix
    """

    def __init__(self, output_dir: str, experiment_name: str = "adam"):
        self.output_dir = Path(output_dir) / "paper_metrics"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name

        # Storage
        self.training_metrics: list[TrainingMetrics] = []
        self.probe_metrics: list[ProbeMetrics] = []
        self.cpi_metrics: list[CPIMetrics] = []
        self.sample_outputs: list[SampleOutput] = []
        self.ablation_metrics: list[AblationMetrics] = []

        # Timing
        self.start_time = time.time()
        self.phase_times: dict[str, float] = {}

        # Initialize CSV files with headers
        self._init_csv_files()

        # Metadata
        self.metadata = {
            "experiment_name": experiment_name,
            "start_time": datetime.now().isoformat(),
            "config": {},
        }

    def _init_csv_files(self):
        """Initialize CSV files with headers."""

        # Training metrics CSV
        training_csv = self.output_dir / "training_metrics.csv"
        with open(training_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "step", "timestamp", "loss", "learning_rate", "gradient_norm",
                "epoch", "samples_seen", "tokens_seen", "throughput_tokens_per_sec",
                "gpu_memory_used_gb", "gpu_utilization_pct"
            ])

        # Probe metrics CSV
        probe_csv = self.output_dir / "probe_metrics.csv"
        with open(probe_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "step", "timestamp", "level1_accuracy", "level2_accuracy",
                "level3_accuracy", "level4_accuracy", "counterfactual_accuracy",
                "overall_accuracy"
            ])

        # CPI metrics CSV
        cpi_csv = self.output_dir / "cpi_metrics.csv"
        with open(cpi_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "step", "a_cf", "a_p", "a_cf_delta", "a_p_delta",
                "inversion_detected", "inversion_score"
            ])

        # Ablation metrics CSV
        ablation_csv = self.output_dir / "ablation_metrics.csv"
        with open(ablation_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "step", "category", "accuracy", "num_samples", "avg_score", "pass_rate"
            ])

    def set_config(self, config: dict):
        """Store experiment configuration."""
        self.metadata["config"] = config
        config_path = self.output_dir / "experiment_config.json"
        with open(config_path, "w") as f:
            json.dump(self.metadata, f, indent=2)

    def log_training_step(
        self,
        step: int,
        loss: float,
        learning_rate: float,
        gradient_norm: Optional[float] = None,
        epoch: Optional[float] = None,
        samples_seen: Optional[int] = None,
        tokens_seen: Optional[int] = None,
        throughput: Optional[float] = None,
        gpu_memory_gb: Optional[float] = None,
        gpu_util_pct: Optional[float] = None,
    ):
        """Log a training step."""

        metrics = TrainingMetrics(
            step=step,
            timestamp=datetime.now().isoformat(),
            loss=loss,
            learning_rate=learning_rate,
            gradient_norm=gradient_norm,
            epoch=epoch,
            samples_seen=samples_seen,
            tokens_seen=tokens_seen,
            throughput_tokens_per_sec=throughput,
            gpu_memory_used_gb=gpu_memory_gb,
            gpu_utilization_pct=gpu_util_pct,
        )

        self.training_metrics.append(metrics)

        # Append to CSV
        csv_path = self.output_dir / "training_metrics.csv"
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                metrics.step, metrics.timestamp, metrics.loss, metrics.learning_rate,
                metrics.gradient_norm, metrics.epoch, metrics.samples_seen,
                metrics.tokens_seen, metrics.throughput_tokens_per_sec,
                metrics.gpu_memory_used_gb, metrics.gpu_utilization_pct
            ])

    def log_probe_results(
        self,
        step: int,
        level1_acc: float,
        level2_acc: float,
        level3_acc: float,
        level4_acc: float,
        probe_results: dict,
        sample_outputs: list[dict] = None,
    ):
        """Log validation probe results."""

        cf_acc = (level1_acc + level2_acc) / 2 if level2_acc > 0 else level1_acc
        overall = (level1_acc + level2_acc + level3_acc + level4_acc) / 4

        metrics = ProbeMetrics(
            step=step,
            timestamp=datetime.now().isoformat(),
            level1_accuracy=level1_acc,
            level2_accuracy=level2_acc,
            level3_accuracy=level3_acc,
            level4_accuracy=level4_acc,
            counterfactual_accuracy=cf_acc,
            overall_accuracy=overall,
            probe_results=probe_results,
            sample_outputs=sample_outputs or [],
        )

        self.probe_metrics.append(metrics)

        # Append to CSV
        csv_path = self.output_dir / "probe_metrics.csv"
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                metrics.step, metrics.timestamp, metrics.level1_accuracy,
                metrics.level2_accuracy, metrics.level3_accuracy,
                metrics.level4_accuracy, metrics.counterfactual_accuracy,
                metrics.overall_accuracy
            ])

        # Update CPI tracking
        self._update_cpi_metrics(step, cf_acc, overall)

        # Store sample outputs
        if sample_outputs:
            for sample in sample_outputs:
                self.sample_outputs.append(SampleOutput(**sample))

    def _update_cpi_metrics(self, step: int, a_cf: float, a_p: float):
        """Track Context-Parametric Inversion metrics."""

        # Calculate deltas
        if self.cpi_metrics:
            prev = self.cpi_metrics[-1]
            a_cf_delta = a_cf - prev.a_cf
            a_p_delta = a_p - prev.a_p
        else:
            a_cf_delta = 0.0
            a_p_delta = 0.0

        # Detect inversion: A_CF declining while A_P stable/rising
        inversion_detected = a_cf_delta < -0.05 and a_p_delta >= -0.02
        inversion_score = max(0, -a_cf_delta) + max(0, a_p_delta) if inversion_detected else 0.0

        metrics = CPIMetrics(
            step=step,
            a_cf=a_cf,
            a_p=a_p,
            a_cf_delta=a_cf_delta,
            a_p_delta=a_p_delta,
            inversion_detected=inversion_detected,
            inversion_score=inversion_score,
        )

        self.cpi_metrics.append(metrics)

        # Append to CSV
        csv_path = self.output_dir / "cpi_metrics.csv"
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                metrics.step, metrics.a_cf, metrics.a_p, metrics.a_cf_delta,
                metrics.a_p_delta, metrics.inversion_detected, metrics.inversion_score
            ])

    def log_ablation(
        self,
        step: int,
        category: str,
        accuracy: float,
        num_samples: int,
        avg_score: float,
        pass_rate: float,
    ):
        """Log per-category ablation metrics."""

        metrics = AblationMetrics(
            step=step,
            category=category,
            accuracy=accuracy,
            num_samples=num_samples,
            avg_score=avg_score,
            pass_rate=pass_rate,
        )

        self.ablation_metrics.append(metrics)

        # Append to CSV
        csv_path = self.output_dir / "ablation_metrics.csv"
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                metrics.step, metrics.category, metrics.accuracy,
                metrics.num_samples, metrics.avg_score, metrics.pass_rate
            ])

    def log_sample_output(
        self,
        step: int,
        probe_name: str,
        probe_level: int,
        prompt: str,
        expected: str,
        actual: str,
        passed: bool,
        score: float,
        category: str,
    ):
        """Log a sample model output for qualitative analysis."""

        sample = SampleOutput(
            step=step,
            probe_name=probe_name,
            probe_level=probe_level,
            prompt=prompt,
            expected=expected,
            actual=actual,
            passed=passed,
            score=score,
            category=category,
        )

        self.sample_outputs.append(sample)

    def log_phase_time(self, phase_name: str, duration_seconds: float):
        """Log time spent in a training phase."""
        self.phase_times[phase_name] = duration_seconds

    def generate_summary_statistics(self) -> dict:
        """Generate summary statistics for the paper."""

        stats = {
            "experiment": self.experiment_name,
            "total_training_time_hours": (time.time() - self.start_time) / 3600,
            "phase_times": self.phase_times,
        }

        # Training stats
        if self.training_metrics:
            losses = [m.loss for m in self.training_metrics]
            stats["training"] = {
                "total_steps": len(self.training_metrics),
                "final_loss": losses[-1],
                "min_loss": min(losses),
                "loss_reduction": losses[0] - losses[-1] if len(losses) > 1 else 0,
            }

        # Probe stats
        if self.probe_metrics:
            final = self.probe_metrics[-1]
            initial = self.probe_metrics[0]

            stats["probes"] = {
                "final_level1": final.level1_accuracy,
                "final_level2": final.level2_accuracy,
                "final_level3": final.level3_accuracy,
                "final_level4": final.level4_accuracy,
                "final_a_cf": final.counterfactual_accuracy,
                "final_overall": final.overall_accuracy,
                "level1_improvement": final.level1_accuracy - initial.level1_accuracy,
                "level2_improvement": final.level2_accuracy - initial.level2_accuracy,
                "a_cf_improvement": final.counterfactual_accuracy - initial.counterfactual_accuracy,
            }

            # Find peak A_CF
            peak_cf = max(self.probe_metrics, key=lambda m: m.counterfactual_accuracy)
            stats["probes"]["peak_a_cf"] = peak_cf.counterfactual_accuracy
            stats["probes"]["peak_a_cf_step"] = peak_cf.step

        # CPI stats
        if self.cpi_metrics:
            inversions = [m for m in self.cpi_metrics if m.inversion_detected]
            stats["cpi"] = {
                "inversion_detected": len(inversions) > 0,
                "num_inversion_events": len(inversions),
                "first_inversion_step": inversions[0].step if inversions else None,
                "max_inversion_score": max(m.inversion_score for m in self.cpi_metrics),
            }

        return stats

    def generate_latex_tables(self) -> str:
        """Generate LaTeX-formatted tables for the paper."""

        latex = []

        # Main results table
        stats = self.generate_summary_statistics()

        if "probes" in stats:
            p = stats["probes"]
            latex.append(r"""
\begin{table}[h]
\centering
\caption{Adam Validation Probe Results}
\label{tab:probe_results}
\begin{tabular}{lcc}
\toprule
\textbf{Probe Level} & \textbf{Final Accuracy} & \textbf{Improvement} \\
\midrule
Level 1 (Basic Override) & %.1f\%% & +%.1f\%% \\
Level 2 (Numerical Override) & %.1f\%% & +%.1f\%% \\
Level 3 (Underdetermined) & %.1f\%% & -- \\
Level 4 (Constraint) & %.1f\%% & -- \\
\midrule
\textbf{$A_{CF}$ (Counterfactual)} & \textbf{%.1f\%%} & \textbf{+%.1f\%%} \\
\bottomrule
\end{tabular}
\end{table}
""" % (
                p["final_level1"] * 100, p["level1_improvement"] * 100,
                p["final_level2"] * 100, p["level2_improvement"] * 100,
                p["final_level3"] * 100,
                p["final_level4"] * 100,
                p["final_a_cf"] * 100, p["a_cf_improvement"] * 100,
            ))

        # Training dynamics table
        if "training" in stats:
            t = stats["training"]
            latex.append(r"""
\begin{table}[h]
\centering
\caption{Training Dynamics}
\label{tab:training_dynamics}
\begin{tabular}{lr}
\toprule
\textbf{Metric} & \textbf{Value} \\
\midrule
Total Steps & %d \\
Final Loss & %.4f \\
Minimum Loss & %.4f \\
Loss Reduction & %.4f \\
Training Time (hours) & %.2f \\
\bottomrule
\end{tabular}
\end{table}
""" % (
                t["total_steps"], t["final_loss"], t["min_loss"],
                t["loss_reduction"], stats["total_training_time_hours"],
            ))

        return "\n".join(latex)

    def save_all(self):
        """Save all collected metrics to files."""

        # Summary statistics JSON
        stats = self.generate_summary_statistics()
        stats_path = self.output_dir / "summary_statistics.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)

        # Full probe metrics JSON
        probe_path = self.output_dir / "probe_metrics_full.json"
        with open(probe_path, "w") as f:
            json.dump([asdict(m) for m in self.probe_metrics], f, indent=2)

        # Sample outputs JSON
        samples_path = self.output_dir / "sample_outputs.json"
        with open(samples_path, "w") as f:
            json.dump([asdict(s) for s in self.sample_outputs], f, indent=2)

        # LaTeX tables
        latex = self.generate_latex_tables()
        latex_path = self.output_dir / "paper_tables.tex"
        with open(latex_path, "w") as f:
            f.write(latex)

        # Metadata with final stats
        self.metadata["end_time"] = datetime.now().isoformat()
        self.metadata["summary_statistics"] = stats
        meta_path = self.output_dir / "experiment_metadata.json"
        with open(meta_path, "w") as f:
            json.dump(self.metadata, f, indent=2)

        print(f"\nPaper metrics saved to {self.output_dir}/")
        print(f"  - training_metrics.csv (loss curves)")
        print(f"  - probe_metrics.csv (accuracy trajectories)")
        print(f"  - cpi_metrics.csv (inversion tracking)")
        print(f"  - ablation_metrics.csv (per-category breakdown)")
        print(f"  - summary_statistics.json")
        print(f"  - sample_outputs.json (qualitative examples)")
        print(f"  - paper_tables.tex (LaTeX tables)")

    def print_summary(self):
        """Print a summary to console."""

        stats = self.generate_summary_statistics()

        print("\n" + "="*60)
        print("EXPERIMENT SUMMARY")
        print("="*60)

        if "training" in stats:
            t = stats["training"]
            print(f"\nTraining:")
            print(f"  Steps: {t['total_steps']}")
            print(f"  Final Loss: {t['final_loss']:.4f}")
            print(f"  Loss Reduction: {t['loss_reduction']:.4f}")

        if "probes" in stats:
            p = stats["probes"]
            print(f"\nProbe Accuracy (Final):")
            print(f"  Level 1 (Basic Override):    {p['final_level1']:.1%}")
            print(f"  Level 2 (Numerical):         {p['final_level2']:.1%}")
            print(f"  Level 3 (Underdetermined):   {p['final_level3']:.1%}")
            print(f"  Level 4 (Constraint):        {p['final_level4']:.1%}")
            print(f"  A_CF (Counterfactual):       {p['final_a_cf']:.1%}")
            print(f"\n  Peak A_CF: {p['peak_a_cf']:.1%} at step {p['peak_a_cf_step']}")

        if "cpi" in stats:
            c = stats["cpi"]
            if c["inversion_detected"]:
                print(f"\n[!] Context-Parametric Inversion detected at step {c['first_inversion_step']}")
            else:
                print(f"\nNo Context-Parametric Inversion detected")

        print(f"\nTotal Time: {stats['total_training_time_hours']:.2f} hours")
        print("="*60)


# =============================================================================
# HELPER FUNCTIONS FOR INTEGRATION
# =============================================================================

def get_gpu_metrics() -> tuple[Optional[float], Optional[float]]:
    """Get current GPU memory and utilization."""
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(", ")
            mem_mb = float(parts[0])
            util_pct = float(parts[1])
            return mem_mb / 1024, util_pct  # Convert to GB
    except:
        pass
    return None, None


def extract_sample_outputs_from_report(report, probes: list, num_samples: int = 3) -> list[dict]:
    """Extract sample outputs from a validation report for paper appendix."""

    samples = []

    # Get a mix of passed and failed samples
    passed = [r for r in report.results if r.passed][:num_samples]
    failed = [r for r in report.results if not r.passed][:num_samples]

    for result in passed + failed:
        # Find the corresponding probe
        probe = next((p for p in probes if p["name"] == result.probe_name), None)

        samples.append({
            "step": report.step,
            "probe_name": result.probe_name,
            "probe_level": result.level,
            "prompt": probe["prompt"] if probe else "",
            "expected": str(probe.get("expected_patterns", []) if probe else ""),
            "actual": result.actual,
            "passed": result.passed,
            "score": result.score,
            "category": f"level_{result.level}",
        })

    return samples


# =============================================================================
# MAIN - For standalone testing
# =============================================================================

if __name__ == "__main__":
    # Demo usage
    collector = PaperMetricsCollector("./test_output", "demo_experiment")

    # Simulate some training steps
    for step in range(0, 1000, 10):
        loss = 2.0 - (step / 1000) * 1.5 + 0.1 * (step % 100) / 100
        collector.log_training_step(
            step=step,
            loss=loss,
            learning_rate=2e-5 * (1 - step/1000),
        )

    # Simulate probe results
    for step in [0, 250, 500, 750, 1000]:
        collector.log_probe_results(
            step=step,
            level1_acc=0.2 + 0.6 * (step / 1000),
            level2_acc=0.1 + 0.5 * (step / 1000),
            level3_acc=0.1 + 0.4 * (step / 1000),
            level4_acc=0.1 + 0.3 * (step / 1000),
            probe_results={},
        )

    collector.save_all()
    collector.print_summary()
