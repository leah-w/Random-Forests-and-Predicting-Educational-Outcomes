import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, plot_tree

from data_loader import get_data

def graph_scatter(depth,y_train, y_val):
    
    plt.scatter(depth, y_train, label = "Training")
    plt.scatter(depth, y_val, label = "Validation")

    #labels and title
    plt.xlabel("Depth of Decision Tree")
    plt.ylabel("Mean Squared Error")
    plt.title("Depth of Decision Tree vs Error")
    plt.legend()
    plt.show()   

def graph_tree(tree,x,):
    col_names = ['Hours_Studied','Attendance','Parental_Involvement','Access_to_Resources','Extracurricular_Activities','Sleep_Hours','Previous_Scores','Motivation_Level','Internet_Access','Tutoring_Sessions','Family_Income','Teacher_Quality','School_Type','Peer_Influence','Physical_Activity','Learning_Disabilities','Parental_Education_Level','Distance_from_Home','Gender']

    plt.figure(figsize=(12, 8))
    plot_tree(tree,
            feature_names = col_names, 
            filled=True, 
            rounded=True)
    plt.title("Decision Tree")
    plt.show()
    
def mse(Y, Yhat):
    """
    Calculates mean squared error between ground-truth Y and predicted Y
    """
    return np.mean((Y-Yhat)**2)

def main():
    """
    """
    Xmat_train, Xmat_test, Xmat_val, Y_train, Y_test, Y_val = get_data()
    n, d = Xmat_train.shape

    #regression tree  
    #random.seed(42)
    print("Tree:")    
    depth = [];  train_mse = []; val_mse = []
    #d = 8
    for i in range(d):
        model = DecisionTreeRegressor(max_depth = i+1).fit(Xmat_train, Y_train) 
        depth.append(i)
    
        train_mse.append(mse(Y_train, model.predict(Xmat_train)))

        val_mse.append(mse(Y_val, model.predict(Xmat_val)))        

    graph_scatter(depth, train_mse, val_mse)
    
    #final tree
    tree = DecisionTreeRegressor(max_depth = 5).fit(Xmat_train, Y_train) 
    
    y_train_mse = mse(Y_train, tree.predict(Xmat_train))  
    print("Training Error", y_train_mse)

    y_val_mse = mse(Y_val, tree.predict(Xmat_val))  
    print("Validation Error", y_val_mse)
    
    test_mse = mse(Y_test, tree.predict(Xmat_test))  
    print("test accuracy", test_mse)

    #graph tree
    graph_tree(tree, Xmat_train)
    


if __name__ == "__main__":
    main()
