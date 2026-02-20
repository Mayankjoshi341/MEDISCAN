
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression


def tune_logistic_regression(X_train, y_train):
    lr = LogisticRegression()

    lr_param_dist = {
        "C": [0.01, 0.1, 1],
        "solver": ["lbfgs" , "saga"],
        "max_iter": [500, 1000]
    }

    lr_search = RandomizedSearchCV(
        lr,
        param_distributions=lr_param_dist,
        n_iter=12,
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
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
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

def best_hyperparameters(lr_model, lr_score, rf_model, rf_score):
    if lr_score > rf_score:
        print(f"Best Model: Logistic Regression with score {lr_score}")
        return lr_model
    else:
        print(f"Best Model: Random Forest with score {rf_score}")
        return rf_model
