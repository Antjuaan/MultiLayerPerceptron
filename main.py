from MultiLayerPerceptron import MultiLayerPerceptron
import numpy as np
import pandas as pd

# Estraggo i dati dal dataset
data = pd.read_csv("MLP/iris.data", header=None)

# Estraggo i dati relativi ai petali e ai sepali
X = data.iloc[:, :-1].values

# Estraggo i dati relativi alle classi in one-hot encoding
y = pd.get_dummies(data.iloc[:, -1]).values

# Suddivido i dati in training e test set (80% training, 20% test)
X_train = np.concatenate((X[:40], X[50:90], X[100:140]))
y_train = np.concatenate((y[:40], y[50:90], y[100:140]))

X_test = np.concatenate((X[40:50], X[90:100], X[140:150]))
y_test = np.concatenate((y[40:50], y[90:100], y[140:150]))

# Creo la rete neurale
nn = MultiLayerPerceptron(4, 8, 3, learning_rate=0.01, momentum=0.9)

# Addestramento e Test
nn.train_test(X_train, y_train, X_test, y_test, epochs=100, debug=True)