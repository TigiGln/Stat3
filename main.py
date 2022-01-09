import pandas as p

from lib.sk_learn import analyze

from lib.keras_tensorflow import neural_network
# transformation of CSV files into DataFrame allowing the processing
df_data = p.read_csv("./data/data.csv", header=1)
df_label = p.read_csv("./data/labels.csv", header=1)

analyze(df_data, df_label)# Launching the analysis with logistic regression


Y = df_label.iloc[:,1]
X = df_data.iloc[:,1:]
neural_network(X, Y) # Launching the analysis with the neural network