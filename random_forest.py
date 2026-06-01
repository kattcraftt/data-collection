# Random Forest Predictive Modeling Pipeline
#
# Reads   : NLP feature dataset from PostgreSQL
# Trains  : Random Forest classifier
# Outputs : Predictions, metrics, feature importance
# Exports : Results CSV + model artifacts

import os
import warnings
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")    # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime
from tqdm import tqdm

warnings.filterwarnings("ignore")

# Scikit-learn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection  import train_test_split, cross_val_score, StratifiedKFold, learning_curve
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.inspection import permutation_importance

# database and config
from database import get_engine, get_connection
from rf_config import RF_CONFIG, RF_OUTPUT_DIR
from config import CSV_OUTPUT_DIR

# database helpers
def load_nlp_features() -> pd.DataFrame:
    """
    Load NLP-extracted records from PostgreSQL.
    Only loads records that have been processed
    by both RoBERTa and BERTopic.
    """
    engine = get_engine()
    df = pd.read_sql("""
        SELECT
            id,
            original_id,
            source,
            platform_weight,
            text,
            score,
            engagement,
            created_date,
            keyword_count,
            sentiment_negative,
            sentiment_neutral,
            sentiment_positive,
            sentiment_label,
            sentiment_confidence,
            topic_id,
            topic_probability,
            topic_label,
            predicted_behavior
        FROM raw_combined_dataset
        WHERE prisma_included    = TRUE
          AND sentiment_label   IS NOT NULL
          AND topic_id          IS NOT NULL
          AND predicted_behavior IS NOT NULL
        ORDER BY id ASC;
    """, engine)
    return df


def save_predictions(df: pd.DataFrame,
                     batch_size: int = 500) -> int:
    """Save RF predictions back to PostgreSQL."""
    if "rf_prediction" not in df.columns:
        return 0

    cols = ["id", "rf_prediction",
               "rf_confidence", "rf_split"]
    avail = [c for c in cols if c in df.columns]
    records = df[avail].copy()
    records = records.where(pd.notnull(records), None)
    recs = records.to_dict("records")
    total = len(recs)
    updated = 0

    print(f"\n  Saving {total:,} predictions to PostgreSQL...")

    conn = get_connection()
    cur  = conn.cursor()

    for i in tqdm(range(0, total, batch_size),
                  desc="  DB Update"):
        batch = recs[i:i + batch_size]
        for rec in batch:
            try:
                rec_id = rec.pop("id")
                set_str = ", ".join([
                    f"{k} = %({k})s"
                    for k in rec.keys()
                ])
                rec["id"] = rec_id

                # add columns if not exist
                cur.execute(f"""
                    DO $$
                    BEGIN
                        BEGIN
                            ALTER TABLE raw_combined_dataset
                            ADD COLUMN rf_prediction VARCHAR(50);
                        EXCEPTION
                            WHEN duplicate_column THEN NULL;
                        END;
                        BEGIN
                            ALTER TABLE raw_combined_dataset
                            ADD COLUMN rf_confidence FLOAT;
                        EXCEPTION
                            WHEN duplicate_column THEN NULL;
                        END;
                        BEGIN
                            ALTER TABLE raw_combined_dataset
                            ADD COLUMN rf_split VARCHAR(10);
                        EXCEPTION
                            WHEN duplicate_column THEN NULL;
                        END;
                    END $$;
                """)

                cur.execute(f"""
                    UPDATE raw_combined_dataset
                    SET {set_str}
                    WHERE id = %(id)s;
                """, rec)

                updated += cur.rowcount

            except Exception as e:
                conn.rollback()
                continue

        conn.commit()

    cur.close()
    conn.close()
    print(f"  ✓ Saved {updated:,} predictions")
    return updated

# data preparation
def prepare_features(df: pd.DataFrame,
                     config: dict) -> tuple:
    """
    Prepare feature matrix X and target vector y
    for Random Forest training.
    """
    print("\n" + "=" * 55)
    print("  DATA PREPARATION")
    print("=" * 55)

    feature_cols = config["feature_columns"]
    target_col = config["target_column"]

    # validate columns exist
    available_features = [
        c for c in feature_cols
        if c in df.columns
    ]
    missing = set(feature_cols) - set(available_features)

    if missing:
        print(f"\n  Missing features: {missing}")
        print(f"  Using available : {available_features}")

    print(f"\n  Features selected  : {len(available_features)}")
    for f in available_features:
        print(f"    - {f}")

    # handle missing values
    print(f"\n  Handling missing values...")

    X = df[available_features].copy()

    # fill numeric NaN with median
    for col in X.columns:
        if X[col].dtype in ["float64", "int64"]:
            median = X[col].median()
            n_null = X[col].isna().sum()
            if n_null > 0:
                print(f"    {col}: filled {n_null} nulls "
                      f"with median={median:.3f}")
            X[col] = X[col].fillna(median)

    # encode target variable
    print(f"\n  Encoding target: {target_col}")

    le = LabelEncoder()
    le.classes_ = np.array(config["target_classes"])

    # only keep rows with valid target classes
    valid_mask = df[target_col].isin(
        config["target_classes"]
    )
    n_invalid = (~valid_mask).sum()

    if n_invalid > 0:
        print(f"    Removed {n_invalid} invalid "
              f"target records")

    X = X[valid_mask].reset_index(drop=True)
    y_raw = df.loc[valid_mask, target_col].reset_index(
        drop=True
    )
    df_valid = df[valid_mask].reset_index(drop=True)

    y = le.transform(y_raw)

    # class distribution
    print(f"\n  Target distribution:")
    print(f"  {'Class':<20} {'Count':>8} {'Percent':>9}")
    print(f"  {'-'*20} {'-'*8} {'-'*9}")

    for cls in config["target_classes"]:
        count = int((y_raw == cls).sum())
        pct = count / len(y_raw) * 100
        print(f"  {cls:<20} {count:>8,} {pct:>8.1f}%")

    print(f"\n  Total records      : {len(X):,}")
    print(f"  Features           : {X.shape[1]}")

    return X, y, le, df_valid, available_features

# train/test split
def split_data(X: pd.DataFrame,
               y: np.ndarray,
               config: dict) -> tuple:
    """Stratified train/test split."""
    print("\n" + "=" * 55)
    print("  TRAIN / TEST SPLIT")
    print("=" * 55)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size = config["test_size"],
        random_state = config["random_state_split"],
        stratify = y
    )

    print(f"\n  Split ratio        : "
          f"{int((1-config['test_size'])*100)}/"
          f"{int(config['test_size']*100)}")
    print(f"  Training samples   : {len(X_train):,}")
    print(f"  Testing samples    : {len(X_test):,}")

    return X_train, X_test, y_train, y_test

# train random forest
def train_model(X_train: pd.DataFrame,
                y_train: np.ndarray,
                config: dict) -> RandomForestClassifier:
    """Train Random Forest classifier."""
    print("\n" + "=" * 55)
    print("  TRAINING RANDOM FOREST")
    print("=" * 55)

    print(f"\n  Hyperparameters:")
    print(f"    n_estimators     : {config['n_estimators']}")
    print(f"    max_depth        : {config['max_depth']}")
    print(f"    min_samples_split: "
          f"{config['min_samples_split']}")
    print(f"    min_samples_leaf : "
          f"{config['min_samples_leaf']}")
    print(f"    max_features     : {config['max_features']}")
    print(f"    class_weight     : {config['class_weight']}")
    print(f"    random_state     : {config['random_state']}")

    rf_model = RandomForestClassifier(
        n_estimators = config["n_estimators"],
        max_depth = config["max_depth"],
        min_samples_split = config["min_samples_split"],
        min_samples_leaf = config["min_samples_leaf"],
        max_features = config["max_features"],
        class_weight = config["class_weight"],
        random_state = config["random_state"],
        n_jobs = config["n_jobs"],
        verbose = config["verbose"]
    )

    print(f"\n  Training on {len(X_train):,} samples...")
    start = datetime.now()
    rf_model.fit(X_train, y_train)
    duration = (datetime.now() - start).total_seconds()

    print(f"  ✓ Training complete in {duration:.1f}s")
    print(f"  ✓ Trees grown: {len(rf_model.estimators_)}")

    return rf_model

# evaluate model
def evaluate_model(rf_model: RandomForestClassifier,
                   X_train: pd.DataFrame,
                   X_test: pd.DataFrame,
                   y_train: np.ndarray,
                   y_test: np.ndarray,
                   le: LabelEncoder,
                   config: dict) -> dict:
    """
    Comprehensive model evaluation including
    accuracy, precision, recall, F1, AUC, and CV.
    """
    print("\n" + "=" * 55)
    print("  MODEL EVALUATION")
    print("=" * 55)

    class_names = list(le.classes_)
    metrics = {}

    # test set predictions
    y_pred = rf_model.predict(X_test)
    y_pred_proba= rf_model.predict_proba(X_test)
    y_train_pred= rf_model.predict(X_train)

    # core metrics
    test_acc = accuracy_score(y_test, y_pred)
    train_acc = accuracy_score(y_train, y_train_pred)
    precision = precision_score(
        y_test, y_pred,
        average="weighted", zero_division=0
    )
    recall = recall_score(
        y_test, y_pred,
        average="weighted", zero_division=0
    )
    f1         = f1_score(
        y_test, y_pred,
        average="weighted", zero_division=0
    )

    # AUC-ROC (one-vs-rest)
    try:
        auc = roc_auc_score(
            y_test,
            y_pred_proba,
            multi_class = "ovr",
            average = "weighted"
        )
    except Exception:
        auc = 0.0

    # cross-validation
    print(f"\n  Running {config['cv_folds']}-fold "
          f"cross-validation...")

    skf = StratifiedKFold(
        n_splits = config["cv_folds"],
        shuffle = True,
        random_state= config["random_state"]
    )

    # combine train + test for cv
    X_all = pd.concat(
        [X_train, X_test],
        ignore_index=True
    )
    y_all = np.concatenate([y_train, y_test])

    cv_scores = cross_val_score(
        rf_model, X_all, y_all,
        cv = skf,
        scoring = "accuracy",
        n_jobs = -1
    )

    cv_f1_scores = cross_val_score(
        rf_model, X_all, y_all,
        cv = skf,
        scoring = "f1_weighted",
        n_jobs = -1
    )

    # overfitting check
    overfit_gap = train_acc - test_acc

    # store all metrics
    metrics = {
        "train_accuracy": round(train_acc, 4),
        "test_accuracy": round(test_acc, 4),
        "precision_weighted": round(precision, 4),
        "recall_weighted": round(recall, 4),
        "f1_weighted": round(f1, 4),
        "auc_roc_weighted": round(auc, 4),
        "cv_accuracy_mean": round(float(cv_scores.mean()), 4),
        "cv_accuracy_std" : round(float(cv_scores.std()), 4),
        "cv_f1_mean": round(float(cv_f1_scores.mean()), 4),
        "cv_f1_std": round(float(cv_f1_scores.std()), 4),
        "overfit_gap": round(overfit_gap, 4),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_classes": len(class_names),
        "n_features": X_train.shape[1]
    }

    # print results
    print(f"\n  Core Metrics (Test Set):")
    print(f"  {'Metric':<25} {'Value':>8}")
    print(f"  {'-'*25} {'-'*8}")
    print(f"  {'Train Accuracy':<25} "
          f"{train_acc:>8.4f}")
    print(f"  {'Test Accuracy':<25} "
          f"{test_acc:>8.4f}")
    print(f"  {'Precision (weighted)':<25} "
          f"{precision:>8.4f}")
    print(f"  {'Recall (weighted)':<25} "
          f"{recall:>8.4f}")
    print(f"  {'F1 Score (weighted)':<25} "
          f"{f1:>8.4f}")
    print(f"  {'AUC-ROC (weighted)':<25} "
          f"{auc:>8.4f}")

    print(f"\n  Cross-Validation "
          f"({config['cv_folds']}-Fold):")
    print(f"  {'CV Accuracy':<25} "
          f"{cv_scores.mean():>8.4f} "
          f"± {cv_scores.std():.4f}")
    print(f"  {'CV F1 Score':<25} "
          f"{cv_f1_scores.mean():>8.4f} "
          f"± {cv_f1_scores.std():.4f}")

    print(f"\n  Overfitting Check:")
    print(f"  {'Train-Test Gap':<25} "
          f"{overfit_gap:>8.4f}")
    if overfit_gap > 0.1:
        print(f"  ⚠ Warning: Possible overfitting "
              f"(gap > 0.1)")
    else:
        print(f"  ✓ No significant overfitting detected")

    # per-class classification report
    print(f"\n  Per-Class Report:")
    report = classification_report(
        y_test, y_pred,
        target_names = class_names,
        zero_division= 0
    )
    print(report)

    # store predictions for export
    metrics["y_pred"] = y_pred
    metrics["y_pred_proba"] = y_pred_proba
    metrics["y_test"] = y_test
    metrics["class_names"] = class_names
    metrics["cv_scores"] = cv_scores

    return metrics

# feature importance
def analyze_feature_importance(
        rf_model: RandomForestClassifier,
        X_test: pd.DataFrame,
        y_test: np.ndarray,
        feature_cols: list,
        output_dir: str,
        timestamp: str) -> pd.DataFrame:
    """
    Compute and visualize feature importance using
    both Gini impurity and permutation importance.
    """
    print("\n" + "=" * 55)
    print("  FEATURE IMPORTANCE ANALYSIS")
    print("=" * 55)

    # gini importance
    gini_importance = pd.Series(
        rf_model.feature_importances_,
        index = feature_cols
    ).sort_values(ascending=False)

    print(f"\n  Gini Importance (Mean Decrease Impurity):")
    print(f"  {'Feature':<25} {'Importance':>12}")
    print(f"  {'-'*25} {'-'*12}")

    for feat, imp in gini_importance.items():
        bar = "█" * int(imp * 50)
        print(f"  {feat:<25} {imp:>12.4f}  {bar}")

    # permutation importance
    print(f"\n  Computing permutation importance...")
    try:
        perm_imp = permutation_importance(
            rf_model, X_test, y_test,
            n_repeats = 10,
            random_state = 42,
            n_jobs = -1
        )

        perm_importance = pd.Series(
            perm_imp.importances_mean,
            index = feature_cols
        ).sort_values(ascending=False)

        print(f"\n  Permutation Importance:")
        print(f"  {'Feature':<25} {'Importance':>12}")
        print(f"  {'-'*25} {'-'*12}")

        for feat, imp in perm_importance.items():
            bar = "█" * max(0, int(imp * 100))
            print(f"  {feat:<25} {imp:>12.4f}  {bar}")

    except Exception as e:
        print(f"  ✗ Permutation importance failed: {e}")
        perm_importance = gini_importance.copy()

    # combined importance dataframe
    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "gini_importance": rf_model.feature_importances_,
        "perm_importance": perm_importance.reindex(
                                feature_cols
                              ).values
    }).sort_values("gini_importance", ascending=False)

    # plot feature importance
    try:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Gini importance plot
        axes[0].barh(
            importance_df["feature"],
            importance_df["gini_importance"],
            color = "steelblue",
            alpha = 0.8
        )
        axes[0].set_xlabel("Gini Importance")
        axes[0].set_title(
            "Feature Importance (Gini Impurity)"
        )
        axes[0].invert_yaxis()

        # permutation importance plot
        axes[1].barh(
            importance_df["feature"],
            importance_df["perm_importance"],
            color = "darkorange",
            alpha = 0.8
        )
        axes[1].set_xlabel("Permutation Importance")
        axes[1].set_title(
            "Feature Importance (Permutation)"
        )
        axes[1].invert_yaxis()

        plt.suptitle(
            "Random Forest Feature Importance",
            fontsize = 14,
            fontweight = "bold"
        )
        plt.tight_layout()

        path_fi = (f"{output_dir}/"
                   f"feature_importance_{timestamp}.png")
        plt.savefig(path_fi, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\n  ✓ Feature importance plot saved")
        print(f"    → {path_fi}")

    except Exception as e:
        print(f"  ✗ Plot error: {e}")

    return importance_df

# confusion matrix plot
def plot_confusion_matrix(y_test: np.ndarray,
                          y_pred: np.ndarray,
                          class_names: list,
                          output_dir: str,
                          timestamp: str):
    """Plot and save confusion matrix."""
    try:
        cm = confusion_matrix(y_test, y_pred)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # raw counts
        sns.heatmap(
            cm,
            annot = True,
            fmt = "d",
            cmap = "Blues",
            xticklabels = class_names,
            yticklabels = class_names,
            ax = axes[0]
        )
        axes[0].set_title("Confusion Matrix (Counts)")
        axes[0].set_ylabel("Actual")
        axes[0].set_xlabel("Predicted")

        # normalized
        cm_norm = cm.astype("float") / (
            cm.sum(axis=1)[:, np.newaxis] + 1e-9
        )
        sns.heatmap(
            cm_norm,
            annot = True,
            fmt = ".2f",
            cmap = "Blues",
            xticklabels = class_names,
            yticklabels = class_names,
            ax = axes[1]
        )
        axes[1].set_title(
            "Confusion Matrix (Normalized)"
        )
        axes[1].set_ylabel("Actual")
        axes[1].set_xlabel("Predicted")

        plt.suptitle(
            "Random Forest — Confusion Matrix",
            fontsize   = 14,
            fontweight = "bold"
        )
        plt.tight_layout()

        path_cm = (f"{output_dir}/"
                   f"confusion_matrix_{timestamp}.png")
        plt.savefig(path_cm, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  ✓ Confusion matrix saved → {path_cm}")

    except Exception as e:
        print(f"  ✗ Confusion matrix plot error: {e}")

# learning curve plot
def plot_learning_curve(rf_model:  RandomForestClassifier,
                        X:         pd.DataFrame,
                        y:         np.ndarray,
                        output_dir:str,
                        timestamp: str):
    """Plot learning curve to diagnose bias/variance."""
    print(f"\n  Generating learning curve...")

    try:
        train_sizes, train_scores, test_scores = (
            learning_curve(
                rf_model, X, y,
                cv = 5,
                n_jobs = -1,
                train_sizes = np.linspace(0.1, 1.0, 10),
                scoring = "f1_weighted"
            )
        )

        train_mean = train_scores.mean(axis=1)
        train_std = train_scores.std(axis=1)
        test_mean = test_scores.mean(axis=1)
        test_std = test_scores.std(axis=1)

        plt.figure(figsize=(9, 6))
        plt.plot(
            train_sizes, train_mean,
            "o-", color="steelblue",
            label="Training Score"
        )
        plt.fill_between(
            train_sizes,
            train_mean - train_std,
            train_mean + train_std,
            alpha=0.15, color="steelblue"
        )
        plt.plot(
            train_sizes, test_mean,
            "o-", color="darkorange",
            label="Cross-Validation Score"
        )
        plt.fill_between(
            train_sizes,
            test_mean - test_std,
            test_mean + test_std,
            alpha=0.15, color="darkorange"
        )
        plt.xlabel("Training Samples")
        plt.ylabel("F1 Score (Weighted)")
        plt.title(
            "Random Forest — Learning Curve",
            fontsize   = 13,
            fontweight = "bold"
        )
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()

        path_lc = (f"{output_dir}/"
                   f"learning_curve_{timestamp}.png")
        plt.savefig(path_lc, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  ✓ Learning curve saved → {path_lc}")

    except Exception as e:
        print(f"  ✗ Learning curve error: {e}")

# export results
def export_results(df_valid:       pd.DataFrame,
                   X_train:        pd.DataFrame,
                   X_test:         pd.DataFrame,
                   y_train:        np.ndarray,
                   y_test:         np.ndarray,
                   metrics:        dict,
                   importance_df:  pd.DataFrame,
                   le:             LabelEncoder,
                   rf_model:       RandomForestClassifier,
                   timestamp:      str):
    """Export all results and model artifacts."""
    print("\n" + "=" * 55)
    print("  EXPORTING RESULTS")
    print("=" * 55)

    os.makedirs(RF_OUTPUT_DIR, exist_ok=True)

    # file 1: predictions dataset
    n_train = len(X_train)
    n_test = len(X_test)

    # build prediction records
    pred_rows = []

    # training predictions
    train_pred  = rf_model.predict(X_train)
    train_proba = rf_model.predict_proba(X_train)

    train_indices = X_train.index.tolist()
    for i, idx in enumerate(train_indices):
        pred_rows.append({
            "dataset_index": idx,
            "rf_prediction": le.inverse_transform([train_pred[i]])[0],
            "rf_confidence": float(max(train_proba[i])),
            "rf_split": "train"
        })

    # test predictions
    test_indices = X_test.index.tolist()
    y_pred = metrics["y_pred"]
    y_pred_proba = metrics["y_pred_proba"]

    for i, idx in enumerate(test_indices):
        pred_rows.append({
            "dataset_index": idx,
            "rf_prediction": le.inverse_transform([y_pred[i]])[0],
            "rf_confidence": float(max(y_pred_proba[i])),
            "rf_split": "test"
        })

    pred_df = pd.DataFrame(pred_rows)
    pred_df = pred_df.set_index("dataset_index")

    # merge with original data
    result_df = df_valid.copy()
    result_df["rf_prediction"] = pred_df["rf_prediction"]
    result_df["rf_confidence"] = pred_df["rf_confidence"]
    result_df["rf_split"] = pred_df["rf_split"]

    path_preds = (f"{RF_OUTPUT_DIR}/"
                  f"rf_predictions_{timestamp}.csv")
    result_df.to_csv(path_preds, index=False)
    print(f"\n  ✓ Predictions dataset")
    print(f"    Records  : {len(result_df):,}")
    print(f"    File     : {path_preds}")

    # file 2: metrics summary
    metrics_export = {
        k: v for k, v in metrics.items()
        if k not in ["y_pred", "y_pred_proba",
                     "y_test", "class_names",
                     "cv_scores"]
    }

    metrics_df = pd.DataFrame([{
        "timestamp"          : timestamp,
        **metrics_export
    }])

    path_metrics = (f"{RF_OUTPUT_DIR}/"
                    f"rf_metrics_{timestamp}.csv")
    metrics_df.to_csv(path_metrics, index=False)
    print(f"\n  ✓ Metrics summary")
    print(f"    File     : {path_metrics}")

    # file 3: feature importance
    path_fi = (f"{RF_OUTPUT_DIR}/"
               f"feature_importance_{timestamp}.csv")
    importance_df.to_csv(path_fi, index=False)
    print(f"\n  ✓ Feature importance")
    print(f"    File     : {path_fi}")

    # file 4: per-class report
    class_names = metrics["class_names"]
    report_dict = {}

    for i, cls in enumerate(class_names):
        cls_mask = metrics["y_test"] == i
        if cls_mask.sum() == 0:
            continue
        cls_pred = metrics["y_pred"][cls_mask]
        correct = (cls_pred == i).sum()

        report_dict[cls] = {
            "class": cls,
            "support": int(cls_mask.sum()),
            "correct": int(correct),
            "precision": round(float(
                precision_score(
                    metrics["y_test"] == i,
                    metrics["y_pred"] == i,
                    zero_division=0
                )
            ), 4),
            "recall": round(float(
                recall_score(
                    metrics["y_test"] == i,
                    metrics["y_pred"] == i,
                    zero_division=0
                )
            ), 4),
            "f1": round(float(
                f1_score(
                    metrics["y_test"] == i,
                    metrics["y_pred"] == i,
                    zero_division=0
                )
            ), 4)
        }

    report_df = pd.DataFrame(report_dict.values())
    path_report = (f"{RF_OUTPUT_DIR}/"
                   f"class_report_{timestamp}.csv")
    report_df.to_csv(path_report, index=False)
    print(f"\n  ✓ Per-class report")
    print(f"    File     : {path_report}")

    # file 5: save model artifact
    model_artifacts = {
        "model": rf_model,
        "label_encoder": le,
        "feature_columns": list(X_train.columns),
        "metrics": metrics_export,
        "timestamp": timestamp
    }

    path_model = (f"{RF_OUTPUT_DIR}/"
                  f"rf_model_{timestamp}.joblib")
    joblib.dump(model_artifacts, path_model)
    print(f"\n  ✓ Model artifact saved")
    print(f"    File     : {path_model}")

    return result_df

# print results summary
def print_results_summary(metrics:   dict,
                           timestamp: str):
    """Print final results for paper documentation."""

    print("\n" + "=" * 55)
    print("  RANDOM FOREST RESULTS SUMMARY")
    print("  (Use these values in your paper)")
    print("=" * 55)

    print(f"""
  ┌─────────────────────────────────────────────────┐
  │  PERFORMANCE METRICS                            │
  │                                                 │
  │  Train Accuracy       : {metrics['train_accuracy']:.4f}              │
  │  Test Accuracy        : {metrics['test_accuracy']:.4f}              │
  │  Precision (weighted) : {metrics['precision_weighted']:.4f}              │
  │  Recall (weighted)    : {metrics['recall_weighted']:.4f}              │
  │  F1 Score (weighted)  : {metrics['f1_weighted']:.4f}              │
  │  AUC-ROC (weighted)   : {metrics['auc_roc_weighted']:.4f}              │
  ├─────────────────────────────────────────────────┤
  │  CROSS-VALIDATION ({metrics['n_classes']}-CLASS)                │
  │                                                 │
  │  CV Accuracy          : {metrics['cv_accuracy_mean']:.4f}              │
  │    ± std              : {metrics['cv_accuracy_std']:.4f}              │
  │  CV F1 Score          : {metrics['cv_f1_mean']:.4f}              │
  │    ± std              : {metrics['cv_f1_std']:.4f}              │
  ├─────────────────────────────────────────────────┤
  │  DATASET SPLIT                                  │
  │                                                 │
  │  Training samples     : {metrics['n_train']:>8,}              │
  │  Testing samples      : {metrics['n_test']:>8,}              │
  │  Total features       : {metrics['n_features']:>8}              │
  │  Target classes       : {metrics['n_classes']:>8}              │
  ├─────────────────────────────────────────────────┤
  │  OVERFITTING CHECK                              │
  │                                                 │
  │  Train-Test Gap       : {metrics['overfit_gap']:.4f}              │
  │  Status               : {"✓ OK" if metrics['overfit_gap'] <= 0.1 else "⚠ Check model"}          │
  └─────────────────────────────────────────────────┘
    """)
# main
def main():
    print("\n" + "=" * 55)
    print("  RANDOM FOREST MODELING PIPELINE")
    print("  Predictive Customer Behavior Analytics")
    print("  Digital Markets 2027-2030")
    print("=" * 55)
    print(f"  Started  : "
          f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    os.makedirs(RF_OUTPUT_DIR, exist_ok=True)

    # step 1: load NLP features
    print("\n[load] Loading NLP feature records...")
    df = load_nlp_features()

    if df is None or df.empty:
        print("\n  ✗ No NLP records found.")
        print("  Run python nlp_extraction.py first.")
        return

    print(f"  ✓ Loaded {len(df):,} records")

    # step 2: prepare features
    X, y, le, df_valid, feature_cols = prepare_features(
        df, RF_CONFIG
    )

    if len(X) < 50:
        print(f"\n  ✗ Not enough records ({len(X)}) "
              f"to train. Need at least 50.")
        return

    # step 3: split data
    X_train, X_test, y_train, y_test = split_data(
        X, y, RF_CONFIG
    )

    # step 4: train model
    rf_model = train_model(X_train, y_train, RF_CONFIG)

    # step 5: evaluate model
    metrics = evaluate_model(
        rf_model,
        X_train, X_test,
        y_train, y_test,
        le, RF_CONFIG
    )

    # step 6: feature importance
    print("\n[analysis] Analyzing feature importance...")
    importance_df = analyze_feature_importance(
        rf_model = rf_model,
        X_test = X_test,
        y_test = y_test,
        feature_cols = feature_cols,
        output_dir = RF_OUTPUT_DIR,
        timestamp = timestamp
    )

    # step 7: Plots
    print("\n[plots] Generating visualizations...")
    plot_confusion_matrix(
        y_test = metrics["y_test"],
        y_pred = metrics["y_pred"],
        class_names = metrics["class_names"],
        output_dir = RF_OUTPUT_DIR,
        timestamp = timestamp
    )

    # combine X for learning curve
    X_all = pd.concat(
        [X_train, X_test],
        ignore_index=True
    )
    y_all = np.concatenate([y_train, y_test])

    plot_learning_curve(
        rf_model = rf_model,
        X = X_all,
        y = y_all,
        output_dir = RF_OUTPUT_DIR,
        timestamp = timestamp
    )

    # step 8: export results
    print("\n[export] Exporting all results...")
    result_df = export_results(
        df_valid = df_valid,
        X_train = X_train,
        X_test = X_test,
        y_train = y_train,
        y_test = y_test,
        metrics = metrics,
        importance_df = importance_df,
        le = le,
        rf_model = rf_model,
        timestamp = timestamp
    )

    # step 9: save predictions to PostgreSQL
    print("\n[save] Writing predictions to PostgreSQL...")
    result_df["id"] = df_valid["id"].values
    save_predictions(result_df)

    # step 10: print summary
    print_results_summary(metrics, timestamp)

    # final summary
    print("=" * 55)
    print("  RANDOM FOREST PIPELINE COMPLETE")
    print("=" * 55)
    print(f"  Records processed  : {len(df_valid):,}")
    print(f"  Test Accuracy      : "
          f"{metrics['test_accuracy']:.4f}")
    print(f"  F1 Score           : "
          f"{metrics['f1_weighted']:.4f}")
    print(f"  CV Accuracy        : "
          f"{metrics['cv_accuracy_mean']:.4f} "
          f"± {metrics['cv_accuracy_std']:.4f}")
    print(f"  Output directory   : {RF_OUTPUT_DIR}/")
    print(f"  Timestamp          : {timestamp}")
    print("=" * 55)
    print("\n  Pipeline complete.")
    print("  All stages finished:")
    print("    ✓ Data Collection  (main.py)")
    print("    ✓ PRISMA Filtering (prisma_filter.py)")
    print("    ✓ NLP Extraction   (nlp_extraction.py)")
    print("    ✓ Random Forest    (random_forest.py)")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()