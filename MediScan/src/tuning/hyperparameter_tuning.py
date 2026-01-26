
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression

def tune_logistic_regression(X_train, y_train):
    lr = LogisticRegression()

    lr_param_dist = {
        "C": [0.01, 0.1, 1, 10, 100],
        "penalty": ["l2"],
        "solver": ["liblinear", "lbfgs"],
        "max_iter": [500, 1000]
    }

    lr_search = RandomizedSearchCV(
        lr,
        param_distributions=lr_param_dist,
        n_iter=20,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
        random_state=42
    )

    lr_search.fit(X_train, y_train)

    return (
        lr_search.best_estimator_,
        lr_search.best_score_,
        lr_search.best_params_
    )

def tune_random_forest(X_train , y_train):
    rf = RandomForestClassifier(random_state= 42)

    rf_param_dist = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"]
    }

    rf_search = RandomizedSearchCV(
        rf,
        param_distributions=rf_param_dist,
        n_iter=30,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
        random_state=42
    )

    rf_search.fit(X_train, y_train)

    return (
        rf_search.best_estimator_,
        rf_search.best_score_,
        rf_search.best_params_
    )

def best_hyperparameter(lr_estimator , lr_score , lr_params , rf_estimator , rf_score , rf_params):
    final_model = lr_estimator if lr_score >= rf_score else rf_estimator
    print(type(final_model))
    return final_model
