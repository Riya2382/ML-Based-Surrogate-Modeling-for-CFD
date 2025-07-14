# Script to train ML surrogate model for CFD
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

data = pd.read_csv("data/airfoil_dataset.csv")
X = data[["camber", "thickness", "aoa", "mach", "Re"]]
y = data[["Cl", "Cd"]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential([
    Dense(64, activation='relu', input_shape=(5,)),
    Dense(64, activation='relu'),
    Dense(2)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.fit(X_train, y_train, epochs=200, validation_split=0.1, callbacks=[EarlyStopping(patience=10)])

model.save("models/trained_model.h5")
