from sklearn.model_selection import train_test_split
import pandas as pd
def train_test_split_data(X , y , train_size = 0.8):
    x_train , x_test , y_train , y_test = train_test_split(X, y , train_size= train_size)
    return x_train , x_test , y_train , y_test
