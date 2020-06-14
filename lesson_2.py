from numpy import argmax
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from matplotlib import pyplot

# load the dataset
path = 'http://localhost:8000/loan_performance.csv'
df = read_csv(path, header=None)

le1 = LabelEncoder()
le2 = LabelEncoder()
le3 = LabelEncoder()
le1.fit_transform(df[3])
le2.fit_transform(df[4])
le3.fit_transform(df[5])
df[3] = le1.transform(df[3])
df[4] = le2.transform(df[4])
df[5] = le3.transform(df[5])

# split into input and output columns
X, y = df.values[:, :-1], df.values[:, -1]
# ensure all data are floating point values
X = X.astype('float32')

# encode strings to integer
le_y = LabelEncoder()
y = le_y.fit_transform(y)



# split into train and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
# determine the number of input features
n_features = X_train.shape[1]

# define model
model = Sequential()
model.add(Dense(20, activation='relu', kernel_initializer='he_normal', input_shape=(n_features,)))
model.add(Dense(16, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(3, activation='softmax'))

# compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# fit the model
history = model.fit(X_train, y_train, epochs=200, batch_size=32, verbose=0)

# plot learning curves
pyplot.title('Learning Curves')
pyplot.xlabel('Epoch')
pyplot.ylabel('Cross Entropy')
pyplot.plot(history.history['loss'], label='loss')
pyplot.plot(history.history['accuracy'], label='accuracy')
pyplot.legend()
pyplot.show()

# evaluate the model
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print('Test Accuracy: %.3f' % acc)

row = [3.625,350000,33, le1.transform(['N'])[0], le2.transform(['R'])[0], le3.transform(['SF'])[0]]
yhat = model.predict([row])
print('Predicted: %s (class=%d)' % (yhat, argmax(yhat)))
print('Predicted class: %s' % le_y.classes_[argmax(yhat)])
