"""
Model architectures for the IDS seed-variance study.

Six architectures spanning a spectrum from convex to strongly
non-convex optimization, and from single models to ensembles, directly
testing whether seed-driven instability is a deep-learning-specific
problem, general across ML approaches, or scales with the degree of
non-convexity in training:
  - LogisticRegressionModel: convex, multinomial logistic regression
    (scikit-learn), the "no non-convexity" baseline.
  - ShallowMLPModel: single-hidden-layer neural network (PyTorch), an
    intermediate point on the non-convexity spectrum.
  - IDSClassifierDNN: 3-hidden-layer feedforward neural network
    (PyTorch), strongly non-convex.
  - RandomForestModel: ensemble of decision trees (scikit-learn).
  - XGBoostModel: gradient boosting (xgboost).
  - LightGBMModel: gradient boosting (lightgbm), a second boosting
    method to broaden ensemble coverage beyond XGBoost alone.

All six expose the same interface: fit(X_train, y_train_idx, seed) and
predict(X_test) -> predicted class indices.

DELIBERATE SCOPE DECISION: full model checkpoints are NOT saved per seed.
The paper's claims rest on the per-seed metrics and per-instance
predictions (saved by train.py), not on the ability to reload an exact
trained model later. Saving checkpoints for 40 seeds x 6 architectures x
multiple datasets adds real storage and complexity for no rigor payoff.
This is a documented scope decision, not an oversight.
"""

import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb

import config

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_full_determinism(seed):
    """
    Reseeds every RNG the training pipeline touches. Must be called fresh
    before EVERY seed's run, not just once at the start of a script.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# DNN
# ---------------------------------------------------------------------------

class _DNNModule(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_sizes, dropout):
        super().__init__()
        layers = []
        prev_size = input_dim
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class IDSClassifierDNN:
    """
    Wraps the PyTorch feedforward network with a fit/predict interface,
    including a genuine held-out validation split for early stopping.
    """

    name = "dnn"

    def __init__(self, input_dim, num_classes):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.model = None
        self.epochs_run = None
        self.best_val_loss = None

    def fit(self, X_train, y_train_idx, seed):
        set_full_determinism(seed)

        n = len(X_train)
        n_val = int(n * config.DNN_VALIDATION_FRACTION)
        split_rng = np.random.default_rng(config.RANDOM_STATE_FOR_SPLIT)
        perm = split_rng.permutation(n)
        val_idx = perm[:n_val]
        train_idx = perm[n_val:]

        X_tr, y_tr = X_train[train_idx], y_train_idx[train_idx]
        X_val, y_val = X_train[val_idx], y_train_idx[val_idx]

        X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
        y_tr_t = torch.tensor(y_tr, dtype=torch.long)
        X_val_t = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
        y_val_t = torch.tensor(y_val, dtype=torch.long).to(DEVICE)

        train_dataset = TensorDataset(X_tr_t, y_tr_t)
        generator = torch.Generator()
        generator.manual_seed(seed)
        train_loader = DataLoader(
            train_dataset, batch_size=config.DNN_BATCH_SIZE, shuffle=True,
            generator=generator,
        )

        self.model = _DNNModule(
            self.input_dim, self.num_classes,
            config.DNN_HIDDEN_SIZES, config.DNN_DROPOUT,
        ).to(DEVICE)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=config.DNN_LEARNING_RATE)
        criterion = nn.CrossEntropyLoss()

        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        for epoch in range(config.DNN_MAX_EPOCHS):
            self.model.train()
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
                optimizer.zero_grad()
                loss = criterion(self.model(batch_X), batch_y)
                loss.backward()
                optimizer.step()

            self.model.eval()
            with torch.no_grad():
                val_logits = self.model(X_val_t)
                val_loss = criterion(val_logits, y_val_t).item()

            if val_loss < best_val_loss - 1e-5:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= config.DNN_EARLY_STOP_PATIENCE:
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        self.epochs_run = epoch + 1
        self.best_val_loss = best_val_loss
        return self

    def predict(self, X_test):
        self.model.eval()
        X_test_t = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            logits = self.model(X_test_t)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
        return preds


# ---------------------------------------------------------------------------
# Random Forest
# ---------------------------------------------------------------------------

class RandomForestModel:
    name = "random_forest"

    def __init__(self, input_dim=None, num_classes=None):
        self.model = None
        self.epochs_run = None

    def fit(self, X_train, y_train_idx, seed):
        self.model = RandomForestClassifier(
            n_estimators=config.RF_N_ESTIMATORS,
            max_depth=config.RF_MAX_DEPTH,
            random_state=seed,
            n_jobs=-1,
        )
        self.model.fit(X_train, y_train_idx)
        self.epochs_run = None
        return self

    def predict(self, X_test):
        return self.model.predict(X_test)


# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------

class XGBoostModel:
    name = "xgboost"

    def __init__(self, input_dim=None, num_classes=None):
        self.model = None
        self.num_classes = num_classes
        self.epochs_run = None

    def fit(self, X_train, y_train_idx, seed):
        self.model = xgb.XGBClassifier(
            n_estimators=config.XGB_N_ESTIMATORS,
            max_depth=config.XGB_MAX_DEPTH,
            learning_rate=config.XGB_LEARNING_RATE,
            subsample=config.XGB_SUBSAMPLE,
            colsample_bytree=config.XGB_COLSAMPLE_BYTREE,
            objective="multi:softmax",
            num_class=self.num_classes,
            random_state=seed,
            n_jobs=-1,
            eval_metric="mlogloss",
        )
        self.model.fit(X_train, y_train_idx)
        self.epochs_run = None
        return self

    def predict(self, X_test):
        return self.model.predict(X_test)


# ---------------------------------------------------------------------------
# LightGBM
# ---------------------------------------------------------------------------

class LightGBMModel:
    name = "lightgbm"

    def __init__(self, input_dim=None, num_classes=None):
        self.model = None
        self.num_classes = num_classes
        self.epochs_run = None

    def fit(self, X_train, y_train_idx, seed):
        self.model = lgb.LGBMClassifier(
            n_estimators=config.LGB_N_ESTIMATORS,
            max_depth=config.LGB_MAX_DEPTH,
            learning_rate=config.LGB_LEARNING_RATE,
            subsample=config.LGB_SUBSAMPLE,
            colsample_bytree=config.LGB_COLSAMPLE_BYTREE,
            objective="multiclass",
            num_class=self.num_classes,
            random_state=seed,
            n_jobs=-1,
            verbosity=-1,
        )
        self.model.fit(X_train, y_train_idx)
        self.epochs_run = None
        return self

    def predict(self, X_test):
        return self.model.predict(X_test)


# ---------------------------------------------------------------------------
# Logistic Regression (convex baseline)
# ---------------------------------------------------------------------------

class LogisticRegressionModel:
    """
    A CONVEX baseline, deliberately included to test the mechanistic
    hypothesis (Discussion) that DNN instability stems from non-convex
    optimization path-dependency. Uses the 'saga' solver specifically,
    NOT the default 'lbfgs', because lbfgs is a deterministic optimizer
    with no dependence on random_state at all (verified directly: lbfgs
    gives byte-identical coefficients across different seeds), which
    would make this a meaningless, trivial test rather than a real one.
    """

    name = "logistic_regression"

    def __init__(self, input_dim=None, num_classes=None):
        self.model = None
        self.epochs_run = None

    def fit(self, X_train, y_train_idx, seed):
        self.model = LogisticRegression(
            solver=config.LOGREG_SOLVER,
            max_iter=config.LOGREG_MAX_ITER,
            C=config.LOGREG_C,
            random_state=seed,
        )
        self.model.fit(X_train, y_train_idx)
        self.epochs_run = None
        return self

    def predict(self, X_test):
        return self.model.predict(X_test)


# ---------------------------------------------------------------------------
# Shallow MLP (single hidden layer)
# ---------------------------------------------------------------------------

class _ShallowMLPModule(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_size, dropout):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        return self.network(x)


class ShallowMLPModel:
    """
    A single-hidden-layer neural network, an intermediate point on the
    "degree of non-convexity" spectrum between logistic regression
    (convex) and the full 3-hidden-layer DNN (strongly non-convex).
    Otherwise trained identically to IDSClassifierDNN (same validation
    split discipline, same early stopping logic), so any difference in
    seed-driven instability between this and the full DNN is attributable
    to capacity/depth, not to a difference in training procedure.
    """

    name = "shallow_mlp"

    def __init__(self, input_dim, num_classes):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.model = None
        self.epochs_run = None
        self.best_val_loss = None

    def fit(self, X_train, y_train_idx, seed):
        set_full_determinism(seed)

        n = len(X_train)
        n_val = int(n * config.SHALLOW_MLP_VALIDATION_FRACTION)
        split_rng = np.random.default_rng(config.RANDOM_STATE_FOR_SPLIT)
        perm = split_rng.permutation(n)
        val_idx = perm[:n_val]
        train_idx = perm[n_val:]

        X_tr, y_tr = X_train[train_idx], y_train_idx[train_idx]
        X_val, y_val = X_train[val_idx], y_train_idx[val_idx]

        X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
        y_tr_t = torch.tensor(y_tr, dtype=torch.long)
        X_val_t = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
        y_val_t = torch.tensor(y_val, dtype=torch.long).to(DEVICE)

        train_dataset = TensorDataset(X_tr_t, y_tr_t)
        generator = torch.Generator()
        generator.manual_seed(seed)
        train_loader = DataLoader(
            train_dataset, batch_size=config.SHALLOW_MLP_BATCH_SIZE, shuffle=True,
            generator=generator,
        )

        self.model = _ShallowMLPModule(
            self.input_dim, self.num_classes,
            config.SHALLOW_MLP_HIDDEN_SIZE, config.SHALLOW_MLP_DROPOUT,
        ).to(DEVICE)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=config.SHALLOW_MLP_LEARNING_RATE)
        criterion = nn.CrossEntropyLoss()

        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        for epoch in range(config.SHALLOW_MLP_MAX_EPOCHS):
            self.model.train()
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
                optimizer.zero_grad()
                loss = criterion(self.model(batch_X), batch_y)
                loss.backward()
                optimizer.step()

            self.model.eval()
            with torch.no_grad():
                val_logits = self.model(X_val_t)
                val_loss = criterion(val_logits, y_val_t).item()

            if val_loss < best_val_loss - 1e-5:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= config.SHALLOW_MLP_EARLY_STOP_PATIENCE:
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        self.epochs_run = epoch + 1
        self.best_val_loss = best_val_loss
        return self

    def predict(self, X_test):
        self.model.eval()
        X_test_t = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            logits = self.model(X_test_t)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
        return preds


ARCHITECTURE_REGISTRY = {
    "dnn": IDSClassifierDNN,
    "random_forest": RandomForestModel,
    "xgboost": XGBoostModel,
    "lightgbm": LightGBMModel,
    "logistic_regression": LogisticRegressionModel,
    "shallow_mlp": ShallowMLPModel,
}


def build_model(architecture_name, input_dim, num_classes):
    if architecture_name not in ARCHITECTURE_REGISTRY:
        raise ValueError(
            f"Unknown architecture '{architecture_name}'. "
            f"Available: {list(ARCHITECTURE_REGISTRY.keys())}"
        )
    model_class = ARCHITECTURE_REGISTRY[architecture_name]
    return model_class(input_dim=input_dim, num_classes=num_classes)