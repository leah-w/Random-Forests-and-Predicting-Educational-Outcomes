import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "browser"

#from mpl_toolkits.mplot3d import Axes3D

from data_loader import get_data

def graph_bar(indep, measure):
    indep_str = [str(x) for x in indep] 
    #print(est_str)
    #print(y_train)
    plt.bar(indep_str, measure)
    #plt.scatter(est, y_val, label = "Validation")

    #labels and title
    plt.xlabel("Models")
    plt.ylabel("Mean Squared Error ")
    plt.title("Model vs Error")
    #plt.legend()
    plt.show()   

def graph_tree(tree,x):
    df = pd.DataFrame(x)

    plt.figure(figsize=(12, 8))
    plot_tree(tree,
            feature_names = df.columns, 
            filled=True, 
            rounded=True)
    plt.title("Decision Tree")
    plt.show()

def predict_variance(model, X):
    pred = np.array([tree.predict(X) for tree in model.estimators_])
    var = np.var(pred, axis = 0)
    return var

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

    #random forest 
    #random.seed(42)
    print("Forest:")
    train_mse = []; val_mse = []
    features = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, "sqrt", "log2", None]
    for feat in features:
        model = RandomForestRegressor(max_features = feat, random_state=25, n_estimators=500, bootstrap=True)
        #Xmat_train = Xmat_train[:, np.random.permutation(Xmat_train.shape[1])]
        model.fit(Xmat_train, Y_train) 

        train_mse.append(mse(Y_train, model.predict(Xmat_train)))   

        val_mse.append(mse(Y_val, model.predict(Xmat_val)))     

    graph_bar(features, train_mse)
    graph_bar(features, val_mse)
    
    #final forest    
    forest = RandomForestRegressor(max_features = 'sqrt', random_state=25, n_estimators=500, bootstrap=True)
    forest.fit(Xmat_train, Y_train) 
    
    y_train_mse = mse(Y_train, forest.predict(Xmat_train))  
    print("training error", y_train_mse)

    y_val_mse = mse(Y_val, forest.predict(Xmat_val))  
    print("validation error", y_val_mse)

    testMSE = mse(Y_test, forest.predict(Xmat_test))  
    print(" test accuracy", testMSE)
    
    
    #graph all errors
    models = ["Linear Regression", "Decision Tree", "Random Forest"]
    error = [5.090229000076439, 10.909087886763114, 6.52193581068525]
    graph_bar(models, error)
    
    # ablation 
    print(forest.feature_importances_)
    feature_import = {}

    for feature in Xmat_train.columns:
        Xtest_copy = Xmat_test.copy()
        Xtest_copy[feature] = np.random.permutation(Xmat_test[feature].values)

        testMSE_perm = mse(Y_test, forest.predict(Xtest_copy))
        feature_import[feature] = testMSE - testMSE_perm

    print("feature_import:", feature_import)    

    features = ["Hours_Studied", "Attendance", "Access_to_Resources"]
    
    models = {}
    #no features 
    models["none"] = [Y_train.mean()] * len(Y_train)

    #one feature
    for feature in features:
        X = Xmat_train[feature].values.reshape(-1, 1)
        m = RandomForestRegressor(random_state=0).fit(X, Y_train)
        models[feature] = m.predict(X)

    #two features
    for i, feature1 in enumerate(features):
        for feature2 in features[i+1:]:
            X = Xmat_train[[feature1, feature2]].values
            m = RandomForestRegressor(random_state=0).fit(X, Y_train)
            models[f"{feature1}, {feature2}"] = m.predict(X)

    #all features
    X_all = Xmat_train[features]
    m = RandomForestRegressor(random_state=0).fit(X_all, Y_train)
    models["all"] = m.predict(X_all)

    SVHours = (1/3 * (models["Hours_Studied"] - models["none"]) + 1/6 * (models["Hours_Studied, Attendance"] - models["Attendance"]) + 1/6 * (models["Hours_Studied, Access_to_Resources"] - models["Access_to_Resources"]) + 1/3 * (models["all"] - models["Attendance, Access_to_Resources"]))

    SVAtt = (1/3 * (models["Attendance"] - models["none"]) + 1/6 * (models["Hours_Studied, Attendance"] - models["Hours_Studied"]) + 1/6 * (models["Attendance, Access_to_Resources"] - models["Access_to_Resources"]) + 1/3 * (models["all"] - models["Hours_Studied, Access_to_Resources"]))

    SVRes = (1/3 * (models["Access_to_Resources"] - models["none"]) + 1/6 * (models["Attendance, Access_to_Resources"] - models["Attendance"]) + 1/6 * (models["Hours_Studied, Access_to_Resources"] - models["Hours_Studied"]) + 1/3 * (models["all"] - models["Hours_Studied, Attendance"]))

    fig = go.Figure(
        go.Waterfall(
            name="waterfall",
            orientation="h",
            y=features,
            x=[SVHours[i], SVAtt[i], SVRes[i]],
            base=models["none"][i],
            decreasing=dict(marker=dict(color="#fb0655")),
            increasing=dict(marker=dict(color="#008bfb"))
            )
        )
    fig.update_layout(
        title=f"Base Prediction: {models['none'][i]}, Predicted probability: {models['all'][i]}", width=1000, height=500, font=dict(size=14),autosize=True)

    fig.show()



if __name__ == "__main__":
    main()

