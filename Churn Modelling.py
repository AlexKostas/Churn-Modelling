# Implemeting Churn Model using an Artificial Neural Network 
# By training this ANN with real data you can predict when a customer of a bank is about to leave 
# The prediction and trainign is based on the data the bank has collected about the customers
# This ANN model has applications to a variety of cases, feel free to modify and use it as you wish
# In order for the script to compile you must have the Keras package installed

# Data preprocessing ------------

# Importing the libraries
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Splitting the dataset into Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# -----------------------------------

# Artificial Neural Network ---------

# Import Keras and related packages
from keras.models import Sequential
from keras.layers import Dense

# Initialise Artificial Neural Network
classifier = Sequential()

# Add input and first hidden layer
classifier.add(Dense(activation="relu", input_dim=11, units=6, kernel_initializer="uniform"))

# Add second hidden layer
classifier.add(Dense(activation="relu", units=6, kernel_initializer="uniform"))

# Add output layer
classifier.add(Dense(activation="sigmoid", units=1, kernel_initializer="uniform")) 

# Compile ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Connect ANN and Training Set
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)

# ---------------------------------------

# Make predictions and evaluate model ---

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# ----------------------------------