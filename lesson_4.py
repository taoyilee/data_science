from numpy import argmax
from numpy import sqrt
from numpy import random
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model
from matplotlib import pyplot

#Change the path as per your local directory
df = read_csv('/Volumes/G-DRIVE mobile/Data/FannieMae/2019Q1/Acquisition_2019Q1.txt',
    delimiter='|', index_col=False,
    names=['loan_identifier', 'channel', 'seller_name', 'original_interest_rate',
           'original_upb', 'original_loan_term', 'origination_date', 'first_paymane_date',
           'ltv', 'cltv', 'number_of_borrowers', 'dti', 'borrower_credit_score',
           'first_time_home_buyer_indicator', 'loan_purpose', 'property_type', 'number_of_units',
           'occupancy_status', 'property_state', 'zip_3_digit', 'mortgage_insurance_percentage',
           'product_type', 'co_borrower_credit_score', 'mortgage_insurance_type',
           'relocation_mortgage_indicator'])

#Get the training data set form the original data
df_reg = df[['original_upb', 'cltv', 'dti', 'borrower_credit_score',
           'occupancy_status', 'property_state', 'original_interest_rate']]

#Transform categorical values
le_occupancy_status = LabelEncoder()
le_property_state = LabelEncoder()
le_occupancy_status.fit_transform(df['occupancy_status'])
le_property_state.fit_transform(df['property_state'])
df_reg['occupancy_status'] = le_occupancy_status.transform(df_reg['occupancy_status'])
df_reg['property_state'] = le_property_state.transform(df_reg['property_state'])

#random sampling
rnd = random.randint(0,df_reg.index.__len__(),1000)
df_reg = df_reg.iloc[rnd,:]

# split into input and output columns
X, y = df_reg.values[:, :-1], df_reg.values[:, -1]
# split into train and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
# determine the number of input features
n_features = X_train.shape[1]

# define model
model = Sequential()
model.add(Dense(20, activation='relu', kernel_initializer='he_normal', input_shape=(n_features,)))
model.add(Dense(16, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(1))
# compile the model
model.compile(optimizer='adam', loss='mse')

# fit the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)

# evaluate the model
error = model.evaluate(X_test, y_test, verbose=0)
print('MSE: %.3f, RMSE: %.3f' % (error, sqrt(error)))

# make a prediction
row = [100000.0, 85.0, 39.0, 652.0, le_occupancy_status.transform(['I'])[0], le_property_state.transform(['NJ'])[0]]

yhat = model.predict([row])
print('Predicted: %.3f' % yhat)

# save model to file
model.save('model.h5')

# load the model from file
model = load_model('model.h5')
# make a prediction
row = [150000.00, 90.00, 40.0, 720.0, 0, 32]
yhat = model.predict([row])
print('Predicted: %.3f' % yhat[0])

