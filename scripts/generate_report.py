from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def generate_plots_and_summary(
    results_dir: Path,
    training_metrics_path: Path | None = None,
    evaluation_report_path: Path | None = None,
    robustness_report_path: Path | None = None,
) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    summary_lines = ["# Thesis Research Experiment Summary\n"]

    # 1. Training & Validation Curves
    if training_metrics_path and training_metrics_path.is_file():
        with training_metrics_path.open("r", encoding="utf-8") as f:
            train_data = json.load(f)

        loss_history = train_data.get("loss_history", [])
        val_history = train_data.get("validation_history", [])
        epochs = len(loss_history)

        if loss_history:
            fig, ax1 = plt.subplots(figsize=(8, 5))
            color = "tab:blue"
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Training Loss", color=color)
            ax1.plot(range(1, epochs + 1), loss_history, marker="o", color=color, label="Train Loss")
            ax1.tick_params(axis="y", labelcolor=color)
            ax1.grid(True, linestyle="--", alpha=0.5)

            if val_history:
                val_f1 = [h.get("macro_f1", 0.0) for h in val_history]
                ax2 = ax1.twinx()
                color = "tab:orange"
                ax2.set_ylabel("Val Macro-F1", color=color)
                ax2.plot(range(1, len(val_f1) + 1), val_f1, marker="s", color=color, label="Val Macro-F1")
                ax2.tick_params(axis="y", labelcolor=color)

            plt.title(f"Training Loss and Validation Performance ({epochs} Epochs)")
            fig.tight_layout()
            curve_path = figures_dir / "training_curves.png"
            plt.savefig(curve_path, dpi=200)
            plt.close()
            logger.info("Saved training curves to %s", curve_path)

        summary_lines.append("## Training Summary")
        summary_lines.append(f"- Epochs Trained: {epochs}")
        if loss_history:
            summary_lines.append(f"- Final Train Loss: {loss_history[-1]:.4f}")
        if val_history:
            best_f1 = max(h.get("macro_f1", 0.0) for h in val_history)
            summary_lines.append(f"- Best Validation Macro-F1: {best_f1:.4f}\n")

    # 2. Held-Out Evaluation & Confusion Matrix
    if evaluation_report_path and evaluation_report_path.is_file():
        with evaluation_report_path.open("r", encoding="utf-8") as f:
            eval_data = json.load(f)

        metrics = eval_data.get("metrics", {})
        conf_mat = np.array(eval_data.get("confusion_matrix", []))
        class_names = eval_data.get("class_names", [])

        summary_lines.append("## Held-Out Test Evaluation")
        summary_lines.append(f"- Total Test Samples: {eval_data.get('total_samples', 0)}")
        summary_lines.append(f"- Top-1 Accuracy: {metrics.get('top1_accuracy', 0.0):.4f}")
        summary_lines.append(f"- Macro-F1: {metrics.get('macro_f1', 0.0):.4f}")
        summary_lines.append(f"- Weighted-F1: {metrics.get('weighted_f1', 0.0):.4f}\n")

        if conf_mat.size > 0 and len(class_names) <= 30:
            fig, ax = plt.subplots(figsize=(10, 8))
            cax = ax.matshow(conf_mat, cmap="Blues")
            fig.colorbar(cax)
            ax.set_xticks(range(len(class_names)))
            ax.set_yticks(range(len(class_names)))
            ax.set_xticklabels(class_names, rotation=90)
            ax.set_yticklabels(class_names)
            plt.title("Held-Out Test Confusion Matrix", pad=20)
            plt.xlabel("Predicted")
            plt.ylabel("True")
            conf_path = figures_dir / "confusion_matrix.png"
            fig.tight_layout()
            plt.savefig(conf_path, dpi=200)
            plt.close()
            logger.info("Saved confusion matrix to %s", conf_path)

    # 3. Robustness Evaluation & Degradation Curves
    if robustness_report_path and robustness_report_path.is_file():
        with robustness_report_path.open("r", encoding="utf-8") as f:
            rob_data = json.load(f)

        records = rob_data.get("results", [])
        clean_acc = rob_data.get("clean_accuracy", 0.0)

        summary_lines.append("## Robustness Evaluation (Controlled Disturbances)")
        summary_lines.append(f"- Clean Accuracy: {clean_acc:.4f}")

        # Group by noise type
        noise_groups: dict[str, list[tuple[float, float, float]]] = {}
        for r in records:
            ntype = r.get("noise_type", "")
            sev = r.get("severity", 0.0)
            n_acc = r.get("noisy", {}).get("accuracy", 0.0)
            deg = r.get("accuracy_degradation", 0.0)
            if ntype not in noise_groups:
                noise_groups[ntype] = []
            noise_groups[ntype].append((sev, n_acc, deg))

        summary_lines.append("| Disturbance Type | Severity | Noisy Accuracy | Accuracy Degradation |")
        summary_lines.append("|---|---|---|---|")
        for ntype, rows in sorted(noise_groups.items()):
            for sev, n_acc, deg in sorted(rows):
                summary_lines.append(f"| {ntype} | {sev:.1f} | {n_acc:.4f} | {deg:.4f} |")

        # Plot robustness degradation
        if noise_groups:
            plt.figure(figsize=(9, 6))
            for ntype, rows in sorted(noise_groups.items()):
                rows_sorted = sorted(rows)
                sevs = [r[0] for r in rows_sorted]
                accs = [r[1] for r in rows_sorted]
                plt.plot(sevs, accs, marker="o", label=ntype)
            plt.title("Model Robustness Under Controlled Disturbances")
            plt.xlabel("Perturbation Severity")
            plt.ylabel("Accuracy")
            plt.ylim(0.0, 1.05)
            plt.grid(True, linestyle="--", alpha=0.6)
            plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            plt.tight_layout()
            rob_plot_path = figures_dir / "robustness_curves.png"
            plt.savefig(rob_plot_path, dpi=200)
            plt.close()
            logger.info("Saved robustness curves to %s", rob_plot_path)

    summary_file = results_dir / "research_summary.md"
    summary_file.write_text("\n".join(summary_lines), encoding="utf-8")
    logger.info("Generated comprehensive research report at %s", summary_file)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate thesis figures and research report from experiment outputs.")
    parser.add_argument("--results-dir", default="results", help="Directory containing experiment results")
    parser.add_argument("--training-metrics", default="results/training_metrics.json", help="Path to training metrics JSON")
    parser.add_argument("--evaluation-report", default="results/evaluation_report.json", help="Path to evaluation report JSON")
    parser.add_argument("--robustness-report", default="results/robustness/report.json", help="Path to robustness report JSON")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    train_p = Path(args.training_metrics) if Path(args.training_metrics).is_file() else None
    eval_p = Path(args.evaluation_report) if Path(args.evaluation_report).is_file() else None
    rob_p = Path(args.robustness_report) if Path(args.robustness_report).is_file() else None

    generate_plots_and_summary(
        results_dir=results_dir,
        training_metrics_path=train_p,
        evaluation_report_path=eval_p,
        robustness_report_path=rob_p,
    )


if __name__ == "__main__":
    main()
