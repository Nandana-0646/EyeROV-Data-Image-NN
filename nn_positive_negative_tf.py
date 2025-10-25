# nn_positive_negative_tf.py
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# create dataset: integers -100..100
X = np.arange(-100,101).astype(np.float32).reshape(-1,1)
y = (X > 0).astype(int).ravel()  # 1 if positive, 0 if zero or negative

# normalize
X_mean = X.mean(); X_std = X.std()
Xn = (X - X_mean) / X_std

model = models.Sequential([
    layers.Input(shape=(1,)),
    layers.Dense(8, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(Xn, y, epochs=40, batch_size=16, verbose=0)

# test
tests = np.array([[-5.],[0.],[1.],[50.]], dtype=np.float32)
tests_norm = (tests - X_mean) / X_std
preds = model.predict(tests_norm)
print("Tests:", tests.ravel().tolist())
print("Predicted probabilities:", preds.ravel().tolist())
print("Predicted classes:", (preds.ravel() > 0.5).astype(int).tolist())
