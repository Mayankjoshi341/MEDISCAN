def model_train(X_train, y_train , model , para):
    model = model(para)
    model.fit(X_train, y_train)
    return model
