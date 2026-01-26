from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import numpy as np

def model_selection(X , y):
    models = {"logistic regression" : LogisticRegression(max_iter= 100),
           "Decision tree " : DecisionTreeClassifier(),
           "Random Forest" : RandomForestClassifier(),
           "XG-Boost" : XGBClassifier()}
    model_evaluate_dis = {}
    for name ,model in models.items():
        model.fit(X , y)
        scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
        print(f"{name}: mean={np.mean(scores)*100:.2f}, std={np.std(scores):.3f}")
        model_evaluate_dis.update({name :(np.mean(scores)*100)})
    print(model_evaluate_dis)
    best_key = max(model_evaluate_dis , key = model_evaluate_dis.get) # type: ignore
    best_model = f"{model_evaluate_dis[best_key]}()"
    return best_model , model