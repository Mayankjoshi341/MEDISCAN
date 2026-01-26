from sklearn.metrics import accuracy_score
def model_evaluate(x_test , y_test , model):
    y_pred = model.predict(x_test)
    score = accuracy_score(y_true=y_test , y_pred= y_pred)
    print(score)
    return score