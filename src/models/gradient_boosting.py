from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


def train_gradient_boosting(
    X_train,
    y_train,
    *,
    n_estimators: int = 100,
    learning_rate: float = 0.05,
    max_depth: int = 3,
    random_state: int = 42,
) -> Pipeline:
    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "classifier",
                GradientBoostingClassifier(
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    max_depth=max_depth,
                    random_state=random_state,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)
    return model
