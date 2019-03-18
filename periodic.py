### Long Short-Term Memory (LSTM) Model with Keras/Tensorflow

### Model and predict a simple periodic function
### We want the LSTM to learn the periodic function using windows of a given size
### and then predict the next N-steps in the series

import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

### Reproducible results 
np.random.seed(42)

### Generate data
x = np.linspace(0, 50, 500)
period = 7
y = np.sin(2*np.pi*x/period)
# y = np.abs(y)

pl.plot(x, y, 'r-')
pl.show()

### Keras requires a 3-D array of (N, W, F) where N is the number of training sequences (windows)
### W is the sequence length (i.e. window size) and F is the number of features for each sequence/window (equal to 1 here since there is 1 predicted value)
window_length = 20
nwindows = len(y)-window_length

pl.plot(x[:window_length], y[:window_length], 'r-')
pl.show()

### Create X, Y training data windows
### X are the features we want to use to base our prediction on - the values of the periodic function that corresponding to time steps
### Y is the next value of the function that comes after the previous group in X
### Sequences are sliding windows and are shifted by 1 time unit each step, and so overlap with prior windows
data = []
for i in range(len(y)-window_length):
		data.append(y[i:i+window_length])
data = np.array(data)

trainingfraction = 0.9
train_size = round(len(y) * trainingfraction)

train = data[:train_size,:]
# np.random.shuffle(train)
train_X = train[:,:-1] ### include window_length - 1
train_Y = train[:,-1] ### use last point in window
test_X = data[train_size:,:-1]
test_Y = data[train_size:,-1]

train_X = np.reshape(train_X, (train_X.shape[0], train_X.shape[1], 1))
test_X = np.reshape(test_X, (test_X.shape[0], test_X.shape[1], 1))

### Build model
model = Sequential()
model.add(LSTM(100, input_shape=(window_length-1, 1)))#, return_sequences=True))
# model.add(Dropout(rate=0.2))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

epochs = 3
batch_size = 10
model.fit(train_X, train_Y, epochs=epochs, batch_size=batch_size, verbose=2)


### Predict on a point-by-point basis - only predict a single point ahead of the training window
def predict_point(model, data):
	predicted = model.predict(data)
	predicted = np.reshape(predicted, (predicted.size,))
	return predicted

### Predict a full sequence from an initial training window - the model predicts the next point (as with point by point method)
### then the window is shifted to the next step so the previously predicted point is now included. After 3 steps, there will be 
### two previous predictions included in the window. After W predictions the model will be entirely based on previous predictions
### and will start to become less accurate
def predict_sequence(model, data, window_size):
	# Shift the window by 1 new prediction each time, re-run predictions on new window
	predicted = []
	current_window = data[0]
	for i in range(len(data)):
		predicted.append(model.predict(current_window[np.newaxis,:,:])[0,0])
		current_window = current_window[1:]
		current_window = np.insert(current_window, [window_size-2], predicted[-1], axis=0)
	return predicted

### Predict the training and test data
trainPredict = predict_sequence(model, train_X, window_length)
testPredict = predict_sequence(model, test_X, window_length)

trainScore = np.sqrt(mean_squared_error(train_Y, trainPredict))
print("Train Score: {:.6f} RMS".format(trainScore))
testScore = np.sqrt(mean_squared_error(test_Y, testPredict))
print("Test Score: {:.6f} RMS".format(testScore))

padding = [np.nan for p in range(window_length)]
train_x_plot = x[:train_size+window_length]
test_x_plot = x[train_size+window_length:]

pl.plot(x, y, 'k-',label='Original data')
pl.plot(train_x_plot, padding+trainPredict,'b-', label='Train Predictions')
pl.plot(test_x_plot, testPredict,'r-', label='Test Predictions')
pl.legend()
pl.show()