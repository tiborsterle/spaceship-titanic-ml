"""
Spaceship Titanic - Binary Classification
Model: Random Forest (scikit-learn)

Outputs:
    - ../results/submission.csv     (Kaggle submission)
    - ../results/figures/           (all plots)
    - ../results/metrics.txt        (accuracy, F1, ROC-AUC)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve
)
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings("ignore")

# Paths
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "RandomForest")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
SUBMISSION_PATH = os.path.join(RESULTS_DIR, "submission.csv")
METRICS_PATH = os.path.join(RESULTS_DIR, "metrics.txt")

SEED = 42
np.random.seed(SEED)


# 1. DATA LOADING & FEATURE ENGINEERING
def load_data():
    """Load train and test CSVs."""
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    print(f"Train shape: {train.shape}  |  Test shape: {test.shape}")
    return train, test


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering:
      - Split Cabin into Deck, CabinNum, Side
      - Compute TotalSpending from amenity columns
      - Drop columns not useful for modelling
    """
    df = df.copy()

    # Split Cabin (format: deck/number/side) into three sub-features
    cabin_split = df["Cabin"].str.split("/", expand=True)
    df["Deck"] = cabin_split[0]
    df["CabinNum"] = pd.to_numeric(cabin_split[1], errors="coerce")
    df["Side"] = cabin_split[2]

    # Placeholder sum for EDA only; recomputed on imputed values inside impute_and_encode
    amenity_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    df["TotalSpending"] = df[amenity_cols].fillna(0).sum(axis=1)

    # Drop columns that won't help the model
    df.drop(columns=["Cabin", "Name", "PassengerId"], inplace=True, errors="ignore")

    return df


def impute_and_encode(train: pd.DataFrame, test: pd.DataFrame):
    """
    Missing value imputation (train statistics applied to both sets):
      - Numerical  -> median
      - Categorical -> mode
    Label-encode all categorical / boolean columns.
    Returns X_train, y_train, X_test, and the list of feature names.
    """
    y_train = train["Transported"].astype(int).values
    train = train.drop(columns=["Transported"])

    # Concatenate so the same transformation is applied consistently
    combined = pd.concat([train, test], keys=["train", "test"])

    num_cols = combined.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = combined.select_dtypes(exclude=[np.number]).columns.tolist()

    # Impute with train-set statistics only (no data leakage)
    for col in num_cols:
        median_val = combined.loc["train", col].median()
        combined[col] = combined[col].fillna(median_val)

    for col in cat_cols:
        mode_val = combined.loc["train", col].mode()
        if len(mode_val) > 0:
            combined[col] = combined[col].fillna(mode_val[0])

    # Recompute TotalSpending now that amenity columns are fully imputed
    amenity_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    existing_amenity = [c for c in amenity_cols if c in combined.columns]
    if existing_amenity:
        combined["TotalSpending"] = combined[existing_amenity].sum(axis=1)

    # Label-encode categorical columns
    le = LabelEncoder()
    for col in cat_cols:
        combined[col] = le.fit_transform(combined[col].astype(str))

    feature_names = combined.columns.tolist()
    X_train = combined.loc["train"].values.astype(np.float32)
    X_test = combined.loc["test"].values.astype(np.float32)

    return X_train, y_train, X_test, feature_names


# 2. EDA VISUALISATIONS
def plot_eda(train_raw: pd.DataFrame):
    """
    EDA plots:
      - Histograms of numerical features split by Transported
      - Count plots for categorical features vs. target
    """
    df = engineer_features(train_raw)

    num_cols = ["Age", "RoomService", "FoodCourt", "ShoppingMall",
                "Spa", "VRDeck", "TotalSpending", "CabinNum"]
    cat_cols = ["HomePlanet", "CryoSleep", "Destination", "VIP", "Deck", "Side"]

    # Histograms
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    for ax, col in zip(axes.flatten(), num_cols):
        if col not in df.columns:
            continue
        for label, grp in df.groupby("Transported"):
            ax.hist(grp[col].dropna(), bins=40, alpha=0.6,
                    label=f"Transported={label}", density=True)
        ax.set_title(col)
        ax.legend(fontsize=7)
    fig.suptitle("Numerical Feature Distributions by Transported", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "eda_numerical_histograms.png"), dpi=120)
    plt.close()

    # Count plots
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, col in zip(axes.flatten(), cat_cols):
        if col not in df.columns:
            continue
        counts = df.groupby([col, "Transported"]).size().unstack(fill_value=0)
        counts.plot(kind="bar", ax=ax, rot=45, legend=(col == cat_cols[0]))
        ax.set_title(col)
        ax.set_xlabel("")
    fig.suptitle("Categorical Features vs. Transported", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "eda_categorical_counts.png"), dpi=120)
    plt.close()

    print("EDA plots saved.")


# 3. RESULT VISUALISATIONS
def plot_confusion_matrix(y_true, y_pred, accuracy: float):
    """Confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Not Transported", "Transported"],
                yticklabels=["Not Transported", "Transported"])
    plt.title(f"Random Forest — Confusion Matrix\nAcc={accuracy:.3f}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "confusion_matrix.png"), dpi=120)
    plt.close()
    print("Confusion matrix saved.")


def plot_roc_curve(y_true, y_prob, auc: float):
    """ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"Random Forest (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — Random Forest")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "roc_curve.png"), dpi=120)
    plt.close()
    print("ROC curve saved.")


def plot_feature_importance(model, feature_names: list):
    """Top-15 feature importances."""
    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1][:15]
    plt.figure(figsize=(9, 5))
    plt.bar(range(len(idx)), importances[idx], color="seagreen")
    plt.xticks(range(len(idx)),
               [feature_names[i] for i in idx], rotation=45, ha="right")
    plt.title("Random Forest — Top-15 Feature Importances")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "feature_importance.png"), dpi=120)
    plt.close()
    print("Feature importance chart saved.")


# 4. MAIN PIPELINE
def main():
    # Load
    train_raw, test_raw = load_data()

    # EDA
    print("\n--- EDA ---")
    plot_eda(train_raw)

    # Feature engineering + preprocessing
    train_fe = engineer_features(train_raw)
    test_fe = engineer_features(test_raw)
    X, y, X_test, feature_names = impute_and_encode(train_fe, test_fe)

    print(f"\nFeature matrix: {X.shape}  |  Test matrix: {X_test.shape}")

    # 3-way 60/20/20 split: train -> selection val -> held-out eval
    X_tv, X_holdout, y_tv, y_holdout = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_tv, y_tv, test_size=0.25, stratify=y_tv, random_state=SEED
    )

    # Hyperparameter grid search (selection on X_val only)
    print("\n--- Hyperparameter search ---")
    hp_grid = [
        {"n_estimators": 200, "max_depth": 15},
        {"n_estimators": 300, "max_depth": 20},
        {"n_estimators": 300, "max_depth": None},
        {"n_estimators": 500, "max_depth": 20},
    ]

    best_acc, best_cfg, best_model = -1, None, None
    for cfg in hp_grid:
        m = RandomForestClassifier(
            n_estimators=cfg["n_estimators"],
            max_depth=cfg["max_depth"],
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=SEED,
        )
        m.fit(X_tr, y_tr)
        acc = accuracy_score(y_val, m.predict(X_val))
        print(f"  {cfg}  -> val_acc={acc:.4f}")
        if acc > best_acc:
            best_acc, best_cfg, best_model = acc, cfg, m

    print(f"\nBest config: {best_cfg}  (val_acc={best_acc:.4f})")

    # 5-Fold Cross Validation
    print("\n--- 5-Fold Cross Validation ---")
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_scores = cross_val_score(best_model, X, y, cv=kfold,
                                scoring="accuracy", n_jobs=1)
    print(f"  CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    model = best_model

    # Final evaluation on held-out set (never seen during selection)
    y_prob = model.predict_proba(X_holdout)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    acc = accuracy_score(y_holdout, y_pred)
    f1  = f1_score(y_holdout, y_pred)
    auc = roc_auc_score(y_holdout, y_prob)
    print(f"  Accuracy={acc:.4f}  F1={f1:.4f}  ROC-AUC={auc:.4f}")

    # Plots
    print("\n--- Saving plots ---")
    plot_confusion_matrix(y_holdout, y_pred, acc)
    plot_roc_curve(y_holdout, y_prob, auc)
    plot_feature_importance(model, feature_names)

    # Save metrics
    with open(METRICS_PATH, "w") as f:
        f.write("Random Forest — Evaluation Results\n")
        f.write("=" * 40 + "\n")
        f.write(f"Accuracy : {acc:.4f}\n")
        f.write(f"F1-Score : {f1:.4f}\n")
        f.write(f"ROC-AUC  : {auc:.4f}\n")
        f.write(f"CV Acc   : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}\n")
        f.write(f"Best cfg : {best_cfg}\n")
    print(f"Metrics written to {METRICS_PATH}")

    # Submission
    test_preds = model.predict(X_test).astype(bool)
    submission = pd.DataFrame({
        "PassengerId": test_raw["PassengerId"],
        "Transported": test_preds,
    })
    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission saved to {SUBMISSION_PATH}")
    print(f"\nTransported distribution:\n{submission['Transported'].value_counts()}")


if __name__ == "__main__":
    main()