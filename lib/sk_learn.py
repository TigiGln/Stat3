import pandas as p
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def Regression_logistic(data, label, max_iter=100, *regularization):
    if regularization:
        regul = regularization[0]
        #Création du modèle de régression logistique pour la classification
        logreg = linear_model.LogisticRegression(max_iter=max_iter, C=regul) 
        #ajout du paramètre max_iter car par défaut max_iter=100 (insuffisant pour un résultat optimal)
    else:
        logreg = linear_model.LogisticRegression(max_iter=max_iter)
    #Séparation de nos échantillons d'entrainement et de test
    x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.33, random_state=42, stratify=label)

    #entrainement de notre échantillon de données
    logreg.fit(x_train, y_train)

    #récupération de notre prédiction
    y_predict = logreg.predict(x_test)

    accuracy_test = accuracy_score(y_test, y_predict)
    print("Accuracy_test:", accuracy_test)
    accuracy_training = accuracy_score(y_train, logreg.predict(x_train))
    print("Accuracy_training:", accuracy_training)

    #Evaluation de notre modèle
    b = p.crosstab(y_predict, y_test)
    print(b)

    # Vérification que le modèle a été efficace
    print("Expected number for the type LUAD:", (y_test == "LUAD").sum(), "\n")

    return x_train, y_train, x_test, y_test

def learning_curve(x_train, y_train, x_test, y_test,fig, num_fig, title_graph, max_iter=100):
    list_accuracy_test = []
    list_accuracy_training = []
    nb_samples = [30, 60, 90, 120, 150, 180, 200]
    for nb in nb_samples:
        logreg = linear_model.LogisticRegression(max_iter=max_iter)
        logreg.fit(x_train.iloc[1:nb,:], y_train.iloc[1:nb])
        accuracy_test = accuracy_score(y_test, logreg.predict(x_test))
        accuracy_train = accuracy_score(y_train.iloc[1:nb], logreg.predict(x_train.iloc[1:nb,:]))
        list_accuracy_test.append(accuracy_test)
        list_accuracy_training.append(accuracy_train)

    
    ax = fig.add_subplot(num_fig, title=title_graph)
    ax = plt.plot(nb_samples, list_accuracy_test, label="Test")
    ax = plt.plot(nb_samples, list_accuracy_training, label="Training")
    



if __name__ == "__main__":
    # transformation des fichiers CSV en DataFrame permettant le traitement
    df_data = p.read_csv("../data/data.csv", header=1)
    df_label = p.read_csv("../data/labels.csv", header=1)
    fig = plt.figure()

    #Condition normal
    #Récupération des données sans les index
    Y = df_label.iloc[:,1]
    X = df_data.iloc[:,1:]
    max_iter = 1000
    x_train, y_train, x_test, y_test = Regression_logistic(X, Y, max_iter)
    ax1 = learning_curve(x_train, y_train, x_test, y_test, fig, 221, "Model simple", max_iter)
    fig.legend(loc= 'lower right')
    #ax1.title.set_text("first")
    # Underfitting
    # Diminution de nos données en enlevant des caractéristiques(données pour certains gènes)
    Y_under = df_label.iloc[:,1]
    X_under = df_data.iloc[:,1:10]
    max_iter = 10000
    x_train_under, y_train_under, x_test_under, y_test_under =Regression_logistic(X_under, Y_under, max_iter)
    ax2 = learning_curve(x_train_under, y_train_under, x_test_under, y_test_under, fig, 222, "Underfitting", max_iter)
    #ax2.title.set_text("second")
    # On peut remarquer que la comparaison des y_prédit par rapport au y_réel est assez faible 
    # par le manque de caractéristiques à prendre en compte

    #Overfitting
    # Diminution du nombres d'échantillons (beaucoup de caractéristiques pour un nombre d'échantillons faible)
    Y_over = df_label.iloc[40:60,1]
    X_over = df_data.iloc[40:60,1:]
    x_train_over, y_train_over, x_test_over, y_test_over =Regression_logistic(X_over, Y_over)
    ax3 = learning_curve(x_train_over, y_train_over, x_test_over, y_test_over, fig, 223, "Overfitting")
    
    
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    plt.show()
    #Test Regularization
    #Regularisation faible:
    Y = df_label.iloc[:,1]
    X = df_data.iloc[:,1:]
    Regression_logistic(X, Y, 1000, 1e40)

    #Forte régularisation
    Y = df_label.iloc[:,1]
    X = df_data.iloc[:,1:]
    Regression_logistic(X, Y, 1000, 1e-40)

    