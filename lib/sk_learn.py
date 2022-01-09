import pandas as p
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def Regression_logistic(data, label, name_analyze, max_iter=100, *regularization):
    """
    Creation of a simple model with logistic regression to classify the data

    :parameters:
        data: Dataframe
            gene expression data for analysis
        label: Series
            Classification of tumor types according to samples
        name_analyze: str
            name of the condition analyzed
        max_iter: int
            number of iterations (Default=100)
        regularization: int
            regularization factor 1/lambda (optionnel)
        

    """
    print(name_analyze)
    if regularization:
        regul = regularization[0]
        # Creation of the logistic regression model for classification with regularization C
        logreg = linear_model.LogisticRegression(max_iter=max_iter, C=regul) 
        # Added max_iter parameter because by default max_iter=100 (insufficient for an optimal result)
    else:
        # Creation of the logistic regression model for classification
        logreg = linear_model.LogisticRegression(max_iter=max_iter)
    # Separation of our training and test samples
    x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.30, random_state=42, stratify=label)

    # Training of our sample data
    logreg.fit(x_train, y_train)

    # Recovery of our predictions
    y_predict = logreg.predict(x_test)

    # Recovery of the model accuracy for the test phase and the training phase
    accuracy_test = accuracy_score(y_test, y_predict)
    print("Accuracy_test:", accuracy_test)
    accuracy_training = accuracy_score(y_train, logreg.predict(x_train))
    print("Accuracy_training:", accuracy_training)

    # Evaluation of our model
    b = p.crosstab(y_predict, y_test)
    print(b)

    # Verification that the model was effective
    print("Expected number for the type LUAD:", (y_test == "LUAD").sum(), "\n")

    return x_train, y_train, x_test, y_test

def learning_curve(x_train, y_train, x_test, y_test, fig, num_fig, title_graph, max_iter=100, *regularization):
    """
    creation  of the evolution curve of the accuracy of the model 
    according to increase of the dataset

    :parameters:
        x_train: Dataframe
            training data
        y_train: Series
            training classification
        x_test: Dataframe
            test data
        y_test: Dataframe
            test classification
        fig: object
            matplotlib figure for all the graphs
        num_fig: int
            placement number of the graph on the figure
        title_graph: str
            title given to the graph
        max_iter: int
            number of iterations for training (default=100)
        regularization: int
            regularization factor to compensate for model errors
    :return:
        ax: object
            the graph of the variation of the accuracy of the model on the test and training data 
    """
    list_accuracy_test = []
    list_accuracy_training = []
    nb_samples = [20, 50, 80, 120, 150, 200, 250, 300, 340, 400, 450, 500]
    for nb in nb_samples:
        if regularization:
            regul = regularization[0]
            logreg = linear_model.LogisticRegression(max_iter=max_iter, C=regul)
        else:
            logreg = linear_model.LogisticRegression(max_iter=max_iter)
        logreg.fit(x_train.iloc[1:nb,:], y_train.iloc[1:nb])
        accuracy_test = accuracy_score(y_test, logreg.predict(x_test))
        accuracy_train = accuracy_score(y_train.iloc[1:nb], logreg.predict(x_train.iloc[1:nb,:]))
        list_accuracy_test.append(accuracy_test)
        list_accuracy_training.append(accuracy_train)

    
    # Creation of the graph added to our matplotlib figure
    graph = fig.add_subplot(num_fig, title=title_graph)
    graph = plt.plot(nb_samples, list_accuracy_test, label="Test")
    graph = plt.plot(nb_samples, list_accuracy_training, label="Training")
    return graph

def analyze():
    """
    Run the different logistic regression analyses to see the differences according to the parameters 

    :parameters:
        None
    :return:
        Display figure with all the graphics
    """
    # transformation of CSV files into DataFrame allowing the processing
    df_data = p.read_csv("../data/data.csv", header=1)
    df_label = p.read_csv("../data/labels.csv", header=1)
    fig = plt.figure()

    #optimal conditions
    #Recovery of data without indexes
    Y = df_label.iloc[:,1]
    X = df_data.iloc[:,1:]
    max_iter = 1000
    x_train, y_train, x_test, y_test = Regression_logistic(X, Y, "Model simple", max_iter)
    graph1 = learning_curve(x_train, y_train, x_test, y_test, fig, 321, "Model simple", max_iter)
    fig.legend(loc= 'lower right')
    
    # Underfitting
    # Reduction of our data by removing characteristics (data for some genes)
    Y_under = df_label.iloc[:,1]
    X_under = df_data.iloc[:,1:10]
    max_iter = 10000
    x_train_under, y_train_under, x_test_under, y_test_under =Regression_logistic(X_under, Y_under, "Underfitting", max_iter)
    graph2 = learning_curve(x_train_under, y_train_under, x_test_under, y_test_under, fig, 322, "Underfitting", max_iter)
    
    # It can be noticed that the comparison of y_predicted versus y_actual
    # is rather weak due to the lack of features to be taken into account

    #Overfitting
    # Reduction of the number of samples (many characteristics for a low number of samples)
    Y_over = df_label.iloc[40:60,1]
    X_over = df_data.iloc[40:60,1:]
    x_train_over, y_train_over, x_test_over, y_test_over =Regression_logistic(X_over, Y_over, "Overfitting")
    graph3 = learning_curve(x_train_over, y_train_over, x_test_over, y_test_over, fig, 323, "Overfitting")
    
    
    #Test Regularization
    #Low level of Regularization:
    Y = df_label.iloc[:,1]
    X = df_data.iloc[:,1:]
    x_train_low, y_train_low, x_test_low, y_test_low = Regression_logistic(X, Y, "Low regulation", 1000, 1e20)
    graph4 = learning_curve(x_train_low, y_train_low, x_test_low, y_test_low, fig, 324, "Low regulation", 1000, 1e20)

    #High level of regulation
    Y = df_label.iloc[:,1]
    X = df_data.iloc[:,1:]
    x_train_high, y_train_high, x_test_high, y_test_high = Regression_logistic(X, Y, "High regulation", 1000, 1e-20)
    graph5 = learning_curve(x_train_high, y_train_high, x_test_high, y_test_high, fig, 325, "High regulation", 1000, 1e-20)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    plt.show()


if __name__ == "__main__":
    analyze()

    