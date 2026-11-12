"""
Spaceship Titanic - Binary Classification
Model: Neural Network (MLP) — PyTorch
 
Outputs:
    - results/MLP/submission_mlp.csv     (Kaggle submission)
    - results/MLP/figures/               (all plots)
    - results/MLP/metrics_mlp.txt        (accuracy, F1, ROC-AUC)
    - results/MLP/pt/                    (model checkpoints)
"""
 
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
 
warnings.filterwarnings("ignore")
 
 
# Paths
DATA_DIR    = os.path.join(os.path.dirname(__file__), "..", "dataset")  # fixed: added ".."
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "MLP")
FIG_DIR     = os.path.join(RESULTS_DIR, "figures")
PT_DIR      = os.path.join(RESULTS_DIR, "pt")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(PT_DIR, exist_ok=True)
 
TRAIN_PATH      = os.path.join(DATA_DIR, "train.csv")
TEST_PATH       = os.path.join(DATA_DIR, "test.csv")
SUBMISSION_PATH = os.path.join(RESULTS_DIR, "submission_mlp.csv")
METRICS_PATH    = os.path.join(RESULTS_DIR, "metrics_mlp.txt")
 
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
 
EARLY_STOP_PATIENCE = 15
 
 
# 1. DATA LOADING & FEATURE ENGINEERING
def load_data():
    """Load train and test CSVs."""
    train = pd.read_csv(TRAIN_PATH)
    test  = pd.read_csv(TEST_PATH)
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
 
    cabin_split    = df["Cabin"].str.split("/", expand=True)
    df["Deck"]     = cabin_split[0]
    df["CabinNum"] = pd.to_numeric(cabin_split[1], errors="coerce")
    df["Side"]     = cabin_split[2]
 
    amenity_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    df["TotalSpending"] = df[amenity_cols].sum(axis=1)
 
    df.drop(columns=["Cabin", "Name", "PassengerId"], inplace=True, errors="ignore")
    return df
 
 
def impute_and_encode(train: pd.DataFrame, test: pd.DataFrame):
    """
    Missing value imputation + label encoding + StandardScaler normalisation.
    Train-set statistics applied to both sets (no data leakage).
    Returns X_train, y_train, X_test, feature_names.
    """
    y_train = train["Transported"].astype(int).values
    train   = train.drop(columns=["Transported"])
 
    combined = pd.concat([train, test], keys=["train", "test"])
 
    num_cols = combined.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = combined.select_dtypes(exclude=[np.number]).columns.tolist()
 
    for col in num_cols:
        median_val = combined.loc["train", col].median()
        combined[col] = combined[col].fillna(median_val)
 
    for col in cat_cols:
        mode_val = combined.loc["train", col].mode()
        if len(mode_val) > 0:
            combined[col] = combined[col].fillna(mode_val[0])
 
    le = LabelEncoder()
    for col in cat_cols:
        combined[col] = le.fit_transform(combined[col].astype(str))
 
    feature_names = combined.columns.tolist()
    X_train_raw = combined.loc["train"].values.astype(np.float32)
    X_test_raw  = combined.loc["test"].values.astype(np.float32)
 
    # Normalise numerical features — important for MLP convergence
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test  = scaler.transform(X_test_raw)
 
    return X_train.astype(np.float32), y_train, X_test.astype(np.float32), feature_names
 
 
# 2. MLP ARCHITECTURE
class MLP(nn.Module):
    """
    Fully-connected feedforward network:
      Input → BatchNorm → [Linear → ReLU → Dropout] x3 → Linear → Sigmoid
    """
    def __init__(self, input_dim: int, hidden_sizes=(256, 128, 64),
                 dropout: float = 0.3):
        super().__init__()
        layers = [nn.BatchNorm1d(input_dim)]
        in_dim = input_dim
        for h in hidden_sizes:
            layers += [
                nn.Linear(in_dim, h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_dim = h
        layers += [nn.Linear(in_dim, 1), nn.Sigmoid()]
        self.net = nn.Sequential(*layers)
 
    def forward(self, x):
        return self.net(x).squeeze(1)
 
 
# 3. TRAINING UTILITIES
def train_model(model, X_tr, y_tr, X_val, y_val,
                epochs=100, batch_size=256, lr=1e-3,
                checkpoint_path="best_mlp.pt"):
    """
    Train MLP with Adam + BCE loss.
    Returns lists of train and validation losses per epoch.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5
    )
 
    # DataLoaders
    tr_ds  = TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr, dtype=torch.float32))
    tr_dl  = DataLoader(tr_ds, batch_size=batch_size, shuffle=True)
    val_ds = TensorDataset(torch.tensor(X_val), torch.tensor(y_val, dtype=torch.float32))
    val_dl = DataLoader(val_ds, batch_size=512)
 
    train_losses, val_losses = [], []
    best_val_loss    = float("inf")
    patience_counter = 0
 
    for epoch in range(1, epochs + 1):
        # train
        model.train()
        epoch_loss = 0.0
        for xb, yb in tr_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(xb)
        train_losses.append(epoch_loss / len(tr_ds))
 
        # validate
        model.eval()
        with torch.no_grad():
            val_loss = sum(
                criterion(model(xb.to(DEVICE)), yb.to(DEVICE)).item() * len(xb)
                for xb, yb in val_dl
            ) / len(val_ds)
        val_losses.append(val_loss)
        scheduler.step(val_loss)
 
        if epoch % 20 == 0:
            print(f"  Epoch {epoch:3d}/{epochs}  "
                  f"train_loss={train_losses[-1]:.4f}  "
                  f"val_loss={val_losses[-1]:.4f}")
 
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"  Early stop at epoch {epoch}  "
                      f"(best val_loss={best_val_loss:.4f})")
                break
 
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    return train_losses, val_losses
 
 
def predict_proba(model, X: np.ndarray, batch_size=512) -> np.ndarray:
    """Return predicted probabilities for class 1."""
    model.eval()
    dl = DataLoader(TensorDataset(torch.tensor(X)), batch_size=batch_size)
    probs = []
    with torch.no_grad():
        for (xb,) in dl:
            probs.append(model(xb.to(DEVICE)).cpu().numpy())
    return np.concatenate(probs)
 
 
# 4. EDA VISUALISATIONS
def plot_eda(train_raw: pd.DataFrame):
    """Histograms of numerical features and count plots for categorical ones."""
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
    plt.savefig(os.path.join(FIG_DIR, "mlp_eda_numerical_histograms.png"), dpi=120)
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
    plt.savefig(os.path.join(FIG_DIR, "mlp_eda_categorical_counts.png"), dpi=120)
    plt.close()
 
    print("EDA plots saved.")
 
 
# 5. RESULT VISUALISATIONS
def plot_loss_curves(train_losses, val_losses):
    """Training and validation loss over epochs."""
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(7, 4))
    plt.plot(epochs, train_losses, label="Train loss")
    plt.plot(epochs, val_losses,   label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("BCE Loss")
    plt.title("MLP — Training / Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "mlp_loss_curve.png"), dpi=120)
    plt.close()
    print("Loss curve saved.")
 
 
def plot_confusion_matrix(y_true, y_pred, accuracy: float):
    """Confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Purples",
                xticklabels=["Not Transported", "Transported"],
                yticklabels=["Not Transported", "Transported"])
    plt.title(f"MLP — Confusion Matrix\nAcc={accuracy:.3f}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "mlp_confusion_matrix.png"), dpi=120)
    plt.close()
    print("Confusion matrix saved.")
 
 
def plot_roc_curve(y_true, y_prob, auc: float):
    """ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"MLP (AUC={auc:.3f})", color="purple")
    plt.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — MLP")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "mlp_roc_curve.png"), dpi=120)
    plt.close()
    print("ROC curve saved.")
 
 
# 6. MAIN PIPELINE
def main():
    # Load
    train_raw, test_raw = load_data()
 
    # EDA
    print("\n--- EDA ---")
    plot_eda(train_raw)
 
    # Feature engineering + preprocessing
    train_fe = engineer_features(train_raw)
    test_fe  = engineer_features(test_raw)
    X, y, X_test, feature_names = impute_and_encode(train_fe, test_fe)
 
    print(f"\nFeature matrix: {X.shape}  |  Test matrix: {X_test.shape}")
 
    # 80/20 stratified split
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )
 
    # Hyperparameter grid
    hp_grid = [
        {"hidden_sizes": (256, 128, 64), "lr": 1e-3, "dropout": 0.3, "epochs": 100},
        {"hidden_sizes": (128, 64),      "lr": 5e-4, "dropout": 0.2, "epochs": 100},
        {"hidden_sizes": (256, 128),     "lr": 1e-3, "dropout": 0.4, "epochs": 100},
    ]
 
    print("\n--- Hyperparameter search ---")
    best_acc, best_cfg, best_model = -1, None, None
    for cfg in hp_grid:
        m = MLP(
            input_dim=X_tr.shape[1],
            hidden_sizes=cfg["hidden_sizes"],
            dropout=cfg["dropout"],
        ).to(DEVICE)
        train_model(m, X_tr, y_tr, X_val, y_val,
                    epochs=cfg["epochs"], lr=cfg["lr"],
                    checkpoint_path=os.path.join(PT_DIR, "hp_search_best.pt"))
        prob = predict_proba(m, X_val)
        acc  = accuracy_score(y_val, (prob >= 0.5).astype(int))
        print(f"  {cfg}  -> val_acc={acc:.4f}")
        if acc > best_acc:
            best_acc, best_cfg, best_model = acc, cfg, m
 
    print(f"\nBest config: {best_cfg}  (val_acc={best_acc:.4f})")
 
    # Re-train best config and record loss curves
    print("\n--- Re-training best model ---")
    model = MLP(
        input_dim=X_tr.shape[1],
        hidden_sizes=best_cfg["hidden_sizes"],
        dropout=best_cfg["dropout"],
    ).to(DEVICE)
    train_losses, val_losses = train_model(
        model, X_tr, y_tr, X_val, y_val,
        epochs=best_cfg["epochs"], lr=best_cfg["lr"],
        checkpoint_path=os.path.join(PT_DIR, "final_best.pt")
    )
 
    # K-Fold cross validation (5 folds) using best hyperparams
    print("\n--- 5-Fold Cross Validation ---")
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_accs = []
    for fold, (tr_idx, val_idx) in enumerate(kfold.split(X, y), 1):
        m = MLP(
            input_dim=X.shape[1],
            hidden_sizes=best_cfg["hidden_sizes"],
            dropout=best_cfg["dropout"],
        ).to(DEVICE)
        train_model(m, X[tr_idx], y[tr_idx], X[val_idx], y[val_idx],
                    epochs=best_cfg["epochs"], lr=best_cfg["lr"],
                    checkpoint_path=os.path.join(PT_DIR, f"cv_fold{fold}.pt"))
        prob = predict_proba(m, X[val_idx])
        acc  = accuracy_score(y[val_idx], (prob >= 0.5).astype(int))
        cv_accs.append(acc)
        print(f"  Fold {fold}: acc={acc:.4f}")
    cv_mean, cv_std = np.mean(cv_accs), np.std(cv_accs)
    print(f"  CV Accuracy: {cv_mean:.4f} ± {cv_std:.4f}")
 
    # Evaluate best model on validation set
    y_prob = predict_proba(model, X_val)
    y_pred = (y_prob >= 0.5).astype(int)
    acc    = accuracy_score(y_val, y_pred)
    f1     = f1_score(y_val, y_pred)
    auc    = roc_auc_score(y_val, y_prob)
    print(f"\n  Accuracy={acc:.4f}  F1={f1:.4f}  ROC-AUC={auc:.4f}")
 
    # Plots
    print("\n--- Saving plots ---")
    plot_loss_curves(train_losses, val_losses)
    plot_confusion_matrix(y_val, y_pred, acc)
    plot_roc_curve(y_val, y_prob, auc)
 
    # Save metrics
    with open(METRICS_PATH, "w") as f:
        f.write("MLP (Neural Network) — Evaluation Results\n")
        f.write("=" * 40 + "\n")
        f.write(f"Accuracy  : {acc:.4f}\n")
        f.write(f"F1-Score  : {f1:.4f}\n")
        f.write(f"ROC-AUC   : {auc:.4f}\n")
        f.write(f"CV Acc    : {cv_mean:.4f} ± {cv_std:.4f}\n")
        f.write(f"Best cfg  : {best_cfg}\n")
    print(f"Metrics written to {METRICS_PATH}")
 
    # Submission
    test_probs = predict_proba(model, X_test)
    test_preds = (test_probs >= 0.5).astype(bool)
    submission = pd.DataFrame({
        "PassengerId": test_raw["PassengerId"],
        "Transported": test_preds,
    })
    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission saved to {SUBMISSION_PATH}")
    print(f"\nTransported distribution:\n{submission['Transported'].value_counts()}")
 
 
if __name__ == "__main__":
    main()