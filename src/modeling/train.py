# src/modeling/train.py
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier


def train_baseline_model(X_train, y_train):
    model = LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        solver="liblinear",
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


def train_tree_model(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=16,
        min_samples_leaf=5,
        min_samples_split=10,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample"
    )
    model.fit(X_train, y_train)
    return model


def train_boosted_model(X_train, y_train):
    model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=300,
        max_depth=10,
        min_samples_leaf=20,
        l2_regularization=1.0,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model