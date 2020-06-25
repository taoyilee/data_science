# lstm for time series forecasting
from numpy import sqrt
from numpy import asarray
from pandas import read_csv
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from matplotlib import pyplot

# split a uni-variate sequence into samples
def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix > len(sequence)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return asarray(X), asarray(y)

# a = [1, 2, 3, 4]
# b = a[1:] + [5]

# load the dataset
path = 'http://localhost:8000/covid_19_daily_fatality.csv'
df = read_csv(path, header=0, index_col=0, squeeze=True)
print(df.index)

# retrieve the values
values = df.values.astype('float32')
# specify the window size
n_steps = 7

# split into samples
X, y = split_sequence(values, n_steps)
# reshape into [samples, time steps, features]
X = X.reshape((X.shape[0], X.shape[1], 1))

# split into train/test
n_test = 14
X_train, X_test, y_train, y_test = X[:-n_test], X[-n_test:], y[:-n_test], y[-n_test:]
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
# define model
model = Sequential()
model.add(LSTM(200, activation='relu', kernel_initializer='he_normal', input_shape=(n_steps,1)))
model.add(Dense(100, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(100, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(1))
# compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
# fit the model
history = model.fit(X_train, y_train, epochs=400, batch_size=32, verbose=2, validation_data=(X_test, y_test))
# evaluate the model
mse, mae = model.evaluate(X_test, y_test, verbose=0)
print('MSE: %.3f, RMSE: %.3f, MAE: %.3f' % (mse, sqrt(mse), mae))

# plot learning curves
pyplot.title('Learning Curves')
pyplot.xlabel('Epoch')
pyplot.ylabel('Loss')
pyplot.plot(history.history['loss'], label='loss')
pyplot.plot(history.history['mae'], label='mae')
pyplot.legend()
pyplot.show()

# make a prediction
# 6/3-6/9/2020 (7 days)
s = [df['6/3/20'], df['6/4/20'], df['6/5/20'], df['6/6/20'], df['6/7/20'], df['6/8/20'], df['6/9/20']]

# Predict for 6/10 through 6/15/2020 - 5 days
for i in range(5):
	row = asarray(s).reshape((1, n_steps, 1))
	yhat = model.predict(row)
	print('Predicted: %.3f' % (yhat))

	p = yhat[0][0]
	s = s[1:] + [p]

