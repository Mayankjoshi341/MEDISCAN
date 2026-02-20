from sklearn.metrics import accuracy_score , classification_report , confusion_matrix , recall_score
def model_evaluate(x_test , y_test , model):
    full_report = {}
    y_pred = model.predict(x_test)
    acc_score = accuracy_score(y_true=y_test , y_pred= y_pred)
    class_report = classification_report(y_true=y_test , y_pred= y_pred)
    conf_matrix = confusion_matrix(y_true=y_test , y_pred= y_pred)
    recall = recall_score(y_true=y_test , y_pred= y_pred , average="weighted")
    print("Classification Report:")
    print(class_report)
    print("Confusion Matrix:")
    print(conf_matrix)
    print("Recall Score:")
    print(recall)
    print("Accuracy Score:")
    print(acc_score)
    full_report.update({"accuracy_score" : acc_score , 
                        "classification_report" : class_report ,
                          "confusion_matrix" : conf_matrix , 
                          "recall_score" : recall})
    return full_report

def evaluate_model_graphs(model , x_test , y_test):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from src.utils.config import REPORTS_DIR
    y_pred = model.predict(x_test)
    conf_matrix = confusion_matrix(y_true=y_test , y_pred= y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(REPORTS_DIR/"confusion_matrix.png")
    return "Confusion matrix saved as 'confusion_matrix.png' in the reports directory."
