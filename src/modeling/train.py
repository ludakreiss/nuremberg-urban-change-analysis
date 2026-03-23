from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


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
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample"
    )
    model.fit(X_train, y_train)
    return model