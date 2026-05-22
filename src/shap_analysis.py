from __future__ import annotations

import logging
import os
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
MPLCONFIG_DIR = PROJECT_ROOT / ".cache" / "matplotlib"
MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))

import json
import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from train import (
    CATEGORY_DATASETS,
    REPORTS_DIR,
    TemporalSplitConfig,
    get_feature_columns,
    load_category_feature_frame,
    load_temporal_frame,
)
from tune_selected_models import build_preprocessor

SHAP_DIR = REPORTS_DIR / "shap_analysis"
SHAP_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


def load_tuned_params(category_id: int) -> tuple[dict, int]:
    params_file = REPORTS_DIR / "selected_model_tuning_best_params.csv"
    if not params_file.exists():
        raise FileNotFoundError(f"Best params file not found: {params_file}")

    params_df = pd.read_csv(params_file)
    row = params_df.loc[params_df["CategoryID"] == category_id].iloc[0]
    params = json.loads(row["BestParamsJSON"])
    num_boost_round = int(params.pop("num_boost_round"))
    return params, num_boost_round


def train_final_model(category_id: int, config: TemporalSplitConfig):
    category_frame = load_category_feature_frame(category_id, config)
    numeric_columns, categorical_columns = get_feature_columns(category_id, category_frame)

    preprocessor = build_preprocessor(numeric_columns, categorical_columns)
    x_all = preprocessor.fit_transform(
        category_frame[numeric_columns + categorical_columns]
    )
    y_all = category_frame["Price"].astype("float64").to_numpy()

    params, num_boost_round = load_tuned_params(category_id)
    dataset = lgb.Dataset(x_all, label=y_all, free_raw_data=False)
    model = lgb.train(
        params=params,
        train_set=dataset,
        num_boost_round=num_boost_round,
    )

    feature_names = preprocessor.get_feature_names_out().tolist()
    return model, x_all, y_all, feature_names, category_frame


def compute_shap_values(model, x_data, feature_names):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_data)
    return shap_values


def generate_shap_summary_plot(shap_values, x_data, feature_names, category_name, output_path):
    shap.summary_plot(
        shap_values,
        x_data,
        feature_names=feature_names,
        show=False,
        plot_size=(14, 8),
    )
    plt.title(f"SHAP Summary - {category_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Summary plot saved: %s", output_path)


def generate_shap_bar_plot(shap_values, x_data, feature_names, category_name, output_path):
    shap.summary_plot(
        shap_values,
        x_data,
        feature_names=feature_names,
        plot_type="bar",
        show=False,
        plot_size=(12, 6),
    )
    plt.title(f"SHAP Feature Importance (Mean |SHAP|) - {category_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Bar plot saved: %s", output_path)


def generate_shap_dependence_plots(shap_values, x_data, feature_names, category_name, top_n=5):
    shap_df = pd.DataFrame(shap_values, columns=feature_names)
    mean_abs_shap = shap_df.abs().mean().sort_values(ascending=False)
    top_features = mean_abs_shap.head(top_n).index.tolist()

    x_df = pd.DataFrame(x_data, columns=feature_names)

    for feat in top_features:
        output_path = SHAP_DIR / f"shap_dependence_{category_name.replace(' ', '_').lower()}_{feat}.png"
        shap.dependence_plot(
            feat,
            shap_values,
            x_data,
            feature_names=feature_names,
            show=False,
        )
        plt.title(f"SHAP Dependence: {feat} - {category_name}", fontsize=12)
        plt.tight_layout()
        plt.savefig(output_path, dpi=120, bbox_inches="tight")
        plt.close()
        log.info("Dependence plot saved: %s", output_path)


def build_feature_importance_table(shap_values, feature_names, category_id, category_name):
    shap_df = pd.DataFrame(shap_values, columns=feature_names)
    mean_abs = shap_df.abs().mean()
    std_abs = shap_df.abs().std()

    importance_df = pd.DataFrame({
        "CategoryID": category_id,
        "CategoryName": category_name,
        "Feature": feature_names,
        "MeanAbsSHAP": mean_abs.values,
        "StdAbsSHAP": std_abs.values,
        "Rank": mean_abs.rank(ascending=False).astype(int),
    })
    return importance_df.sort_values("Rank").reset_index(drop=True)


def write_shap_report(all_importance_dfs: list[pd.DataFrame]):
    report_lines = [
        "# SHAP Feature Importance Analysis",
        "",
        f"- Generated: `{datetime.now().isoformat()}`",
        "- Method: `TreeExplainer` (LightGBM native)",
        "- Models: tuned LightGBM for UberX, Uber Comfort, Uber Black",
        "",
    ]

    for imp_df in all_importance_dfs:
        cat_name = imp_df["CategoryName"].iloc[0]
        report_lines.extend([
            f"## {cat_name}",
            "",
            "Top 15 features by mean |SHAP value|:",
            "",
            imp_df.head(15).to_markdown(index=False),
            "",
        ])

    report_lines.extend([
        "## Interpretation",
        "",
        "- SHAP values measure each feature's contribution to the prediction relative to the base value.",
        "- Positive SHAP = pushes price up; Negative SHAP = pushes price down.",
        "- Mean |SHAP| ranks features by overall impact magnitude.",
        "- Cross-category price features (e.g., Price_Comfort in UberX model) capture relative pricing signals.",
        "- User history features (UserPrior*) capture customer behavior patterns.",
        "- Temporal features (Create*) capture demand seasonality and surge pricing windows.",
    ])

    report_file = SHAP_DIR / "shap_analysis_report.md"
    report_file.write_text("\n".join(report_lines), encoding="utf-8")
    log.info("Report saved: %s", report_file)


def main():
    log.info("=" * 60)
    log.info("SHAP Feature Importance Analysis")
    log.info("=" * 60)

    config = TemporalSplitConfig()
    _ = load_temporal_frame(config)

    all_importance_dfs = []

    for category_id in sorted(CATEGORY_DATASETS):
        spec = CATEGORY_DATASETS[category_id]
        cat_name = spec["dataset_name"]
        log.info("Processing %s (CategoryID=%d)...", cat_name, category_id)

        model, x_all, y_all, feature_names, category_frame = train_final_model(category_id, config)

        log.info("Computing SHAP values for %s (samples=%d, features=%d)...",
                 cat_name, x_all.shape[0], x_all.shape[1])
        shap_values = compute_shap_values(model, x_all, feature_names)

        generate_shap_summary_plot(
            shap_values, x_all, feature_names, cat_name,
            SHAP_DIR / f"shap_summary_{cat_name.replace(' ', '_').lower()}.png",
        )
        generate_shap_bar_plot(
            shap_values, x_all, feature_names, cat_name,
            SHAP_DIR / f"shap_bar_{cat_name.replace(' ', '_').lower()}.png",
        )
        generate_shap_dependence_plots(
            shap_values, x_all, feature_names, cat_name, top_n=5,
        )

        importance_df = build_feature_importance_table(
            shap_values, feature_names, category_id, cat_name,
        )
        importance_df.to_csv(
            SHAP_DIR / f"shap_importance_{cat_name.replace(' ', '_').lower()}.csv",
            index=False,
        )
        all_importance_dfs.append(importance_df)

        log.info("SHAP analysis complete for %s", cat_name)

    write_shap_report(all_importance_dfs)
    log.info("=" * 60)
    log.info("SHAP analysis complete for all categories")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
