import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

from data_loader import get_data

def mse(Y, Yhat):
    """
    Calculates mean squared error between ground-truth Y and predicted Y
    """
    return np.mean((Y-Yhat)**2)


def main():
    """
    """
    Xmat_train, Xmat_test, Xmat_val, Y_train, Y_test, Y_val = get_data()
    #n, d = Xmat_train.shape

    #linear regression 
    print("Linear Regression:")
    model = LinearRegression().fit(Xmat_train, Y_train)
    
    train_acc = mse(Y_train, model.predict(Xmat_train))  
    print("training error", train_acc)

    val_acc = mse(Y_val, model.predict(Xmat_val))  
    print("validation error", val_acc)

    test_acc = mse(Y_test, model.predict(Xmat_test))  
    print("test error", test_acc)   



if __name__ == "__main__":
    main()





