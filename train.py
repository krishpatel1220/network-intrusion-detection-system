from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from config import (
    FEATURE_COLUMNS,
    FULL_DATASET_PATH,
    MODELS_DIR,
    PRODUCTION_MODEL_NAME,
    REPORTS_DIR,
    TARGET_COLUMN,
)
from src.artifacts import ModelBundle, file_fingerprint
from src.evaluation import EvaluationResult, comparison_table, confusion_pairs
from src.preprocessing import clean_flows, split_dataset
from src.training import TrainedModel, experiment_config, model_specs, train_all_models

SEED_DEFAULT = 42


def _as_markdown(table: pd.DataFrame) -> str:
    try:
        return table.to_markdown(index=False)
    except ImportError:
        headers = [str(c) for c in table.columns]
        rows = [[("" if pd.isna(v) else str(v)) for v in row] for row in table.to_numpy()]
        lines = [
            "| " + " | ".join(headers) + " |",
            "|" + "|".join("---" for _ in headers) + "|",
        ]
        lines += ["| " + " | ".join(row) + " |" for row in rows]
        return "\n".join(lines)


def _write_model_report(out_dir: Path, trained: TrainedModel) -> None:
    payload = {}
    for split_name, result in trained.evaluations.items():
        payload[split_name] = {
            "accuracy": result.accuracy,
            "precision_macro": result.precision_macro,
            "recall_macro": result.recall_macro,
            "f1_macro": result.f1_macro,
            "f1_weighted": result.f1_weighted,
            "attack_fpr": result.attack_fpr,
        }
        result.per_class.to_csv(out_dir / f"{trained.name}_{split_name}_per_class.csv", index=False)
        result.confusion.to_csv(out_dir / f"{trained.name}_{split_name}_confusion.csv")
        (out_dir / f"{trained.name}_{split_name}_report.txt").write_text(
            result.report_text, encoding="utf-8"
        )
    payload["fit_seconds"] = round(trained.fit_seconds, 2)
    payload["best_iteration"] = trained.best_iteration
    (out_dir / f"{trained.name}_metrics.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


def _markdown_report(
    config: dict,
    trained_models: list[TrainedModel],
    test_results: list[EvaluationResult],
) -> str:
    lines = [
        "# Experiment report",
        "",
        f"- Generated: {config['generated_at']}",
        f"- Dataset: {config['dataset']}",
        f"- Rows: train {config['split_sizes']['train']:,} / "
        f"validation {config['split_sizes']['validation']:,} / "
        f"test {config['split_sizes']['test']:,}",
        f"- Train/test duplicate-overlap rows: "
        f"{config['train_test_overlap_rows']}",
        f"- Random seed: {config['random_state']}",
        "",
        "## Test-set comparison",
        "",
    ]
    table = comparison_table(test_results).round(6)
    lines.append(_as_markdown(table))
    lines.append("")
    for trained in trained_models:
        lines.append(f"## {trained.label}")
        lines.append("")
        test = trained.evaluations.get("test")
        if test is None:
            continue
        lines.append(f"Fit time: {trained.fit_seconds:.1f}s")
        if trained.best_iteration:
            lines.append(f"Best boosting iteration (early stopping): {trained.best_iteration}")
        lines.append("")
        lines.append("Per-class metrics (test):")
        lines.append("")
        lines.append(_as_markdown(test.per_class.round(4)))
        lines.append("")
        lines.append("Most frequent confusions (test):")
        lines.append("")
        pairs = confusion_pairs(test, top_n=5)
        lines.append(_as_markdown(pairs) if not pairs.empty else "_none_")
        lines.append("")
    return "\n".join(lines)
def run_training(
    data_path: Path,
    output_dir: Path,
    models_dir: Path,
    production_model: str = PRODUCTION_MODEL_NAME,
    model_names: list[str] | None = None,
    seed: int = SEED_DEFAULT,
    save_bundles: bool = True,
    save_all_bundles: bool = False,
) -> dict:
    """Execute the full experiment; returns a summary dict."""
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}. Run prepare_data.py first."
        )

    print(f"Loading {data_path} ...")
    df = pd.read_csv(data_path)
    df, cleaning_report = clean_flows(df, feature_columns=FEATURE_COLUMNS)
    print(
        f"  {len(df):,} usable rows "
        f"({cleaning_report.duplicate_rows_dropped:,} duplicates removed on this pass)"
    )
    if cleaning_report.unmapped_label_counts:
        print(f"  WARNING: unmapped labels dropped: {cleaning_report.unmapped_label_counts}")

    split = split_dataset(df, FEATURE_COLUMNS, target_column=TARGET_COLUMN, random_state=seed)
    sizes = split.sizes()
    print(
        f"  split: train {sizes['train']:,} / validation {sizes['validation']:,} / "
        f"test {sizes['test']:,}"
    )
    if split.train_test_overlap_rows:
        print(
            f"  WARNING: {split.train_test_overlap_rows:,} test rows are "
            "feature-identical to train rows; test scores may be inflated."
        )

    specs = model_specs()
    if model_names:
        unknown = sorted(set(model_names) - {s.name for s in specs})
        if unknown:
            raise ValueError(
                f"Unknown model name(s): {', '.join(unknown)}. "
                f"Available: {', '.join(s.name for s in specs)}"
            )
        specs = [s for s in specs if s.name in model_names]

    started = datetime.now(timezone.utc)
    run_dir = output_dir / f"experiment_{started:%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)

    dataset_info = {
        "path": str(data_path),
        **file_fingerprint(data_path),
        "rows_after_cleaning": len(df),
        "class_counts": cleaning_report.label_counts,
    }
    config = experiment_config(split, seed, dataset_info)
    config["generated_at"] = started.isoformat()
    config["production_model"] = production_model

    print("Training models ...")
    trained_models = train_all_models(split, specs=specs, random_state=seed)

    for trained in trained_models:
        _write_model_report(run_dir, trained)
        test = trained.evaluations.get("test")
        if test:
            print(
                f"  {trained.label:20s} accuracy={test.accuracy:.4%} "
                f"macro-F1={test.f1_macro:.4%} attack-FPR={test.attack_fpr:.4%}"
            )

    test_results = [t.evaluations["test"] for t in trained_models if "test" in t.evaluations]
    val_results = [t.evaluations["validation"] for t in trained_models if "validation" in t.evaluations]
    comparison = comparison_table(test_results)
    comparison.to_csv(run_dir / "comparison_test.csv", index=False)
    comparison_table(val_results).to_csv(run_dir / "comparison_validation.csv", index=False)
    # Backwards-compatible file consumed by the dashboard's Model page.
    models_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(models_dir / "model_comparison.csv", index=False)

    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (run_dir / "REPORT.md").write_text(
        _markdown_report(config, trained_models, test_results), encoding="utf-8"
    )

    saved_bundles: list[str] = []
    if save_bundles:
        wanted = (
            trained_models
            if save_all_bundles
            else [t for t in trained_models if t.name == production_model]
        )
        if not wanted:
            print(
                f"  NOTE: production model '{production_model}' was not among "
                "the trained models; no bundle saved."
            )
        for trained in wanted:
            bundle = ModelBundle(
                model_name=trained.name,
                model_label=trained.label,
                estimator=trained.estimator,
                label_encoder=trained.label_encoder,
                feature_names=list(FEATURE_COLUMNS),
                training_config=config,
                metrics={
                    name: {
                        "accuracy": r.accuracy,
                        "f1_macro": r.f1_macro,
                        "f1_weighted": r.f1_weighted,
                        "attack_fpr": r.attack_fpr,
                    }
                    for name, r in trained.evaluations.items()
                },
                library_versions=config["library_versions"],
                created_at=started.isoformat(),
            )
            path = models_dir / f"{trained.name}.bundle.joblib"
            bundle.save(path)
            saved_bundles.append(str(path))
            print(f"  saved bundle: {path}")

    summary = {
        "run_dir": str(run_dir),
        "bundles": saved_bundles,
        "test_comparison": comparison.to_dict(orient="records"),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Reports written to {run_dir}")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the NIDS classifiers.")
    parser.add_argument(
        "dataset",
        nargs="?",
        type=Path,
        default=FULL_DATASET_PATH,
        help=f"Cleaned labeled CSV (default: {FULL_DATASET_PATH})",
    )
    parser.add_argument("--seed", type=int, default=SEED_DEFAULT)
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Subset of models to train (default: all). "
        "Choices: logistic_regression decision_tree random_forest xgboost",
    )
    parser.add_argument(
        "--production-model",
        default=PRODUCTION_MODEL_NAME,
        help="Which trained model to deploy as the dashboard bundle.",
    )
    parser.add_argument(
        "--all-bundles",
        action="store_true",
        help="Save a bundle for every trained model, not just the production one.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Run the experiment without writing model bundles.",
    )
    parser.add_argument("--reports-dir", type=Path, default=REPORTS_DIR)
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    args = parser.parse_args(argv)

    run_training(
        data_path=args.dataset,
        output_dir=args.reports_dir,
        models_dir=args.models_dir,
        production_model=args.production_model,
        model_names=args.models,
        seed=args.seed,
        save_bundles=not args.no_save,
        save_all_bundles=args.all_bundles,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
