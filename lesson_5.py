import pandas as pd
from matplotlib import pyplot
from numpy import argmax
from numpy import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dense


def add_status_column(result):
    status = []
    for i in result.index:
        if result['current_loan_delinquency_status'][i] == '0':
            status.append('NORMAL')
        elif result['current_loan_delinquency_status'][i] > '0' and result['current_loan_delinquency_status'][i] != 'X':
            status.append('DELINQUENT')
        elif str(result.at[i, 'zero_balance_code']) != 'nan':
            status.append('PAYOFF')
        else:
            status.append('OTHER')
    return status


def create_and_compile_nn(in_neurons, hidden_neurons, out_neurons, n_features, batch_normalize):
    # define model
    model = Sequential()
    model.add(Dense(in_neurons, activation='relu', kernel_initializer='he_normal', input_shape=(n_features,)))
    if batch_normalize:
        model.add(BatchNormalization())
    model.add(Dense(hidden_neurons, activation='relu', kernel_initializer='he_normal'))
    model.add(Dense(out_neurons, activation='softmax'))
    # compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def train_nn(df, test_size, num_epochs, batch_size, batch_normalize, nn_config):
    # split into input and output columns
    X, y = df.values[:, :-1], df.values[:, -1]
    # ensure all data are floating point values
    X = X.astype('float32')
    # encode strings to integer
    le_y = LabelEncoder()
    y = le_y.fit_transform(y)
    # split into train and test datasets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    n_features = X_train.shape[1]
    model = create_and_compile_nn(nn_config[0], nn_config[1], nn_config[2], n_features, batch_normalize)
    # fit the model
    history = model.fit(X_train, y_train, epochs=num_epochs, batch_size=batch_size, verbose=0)

    # evaluate the model
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print('Test Accuracy: %.3f' % acc)

    return model, history, loss, acc


def learn(df_train):
    models = []
    histories = []
    losses = []
    accuracies = []
    for i in range(0, 99):
        # random sampling
        rnd = random.randint(0, df_train.index.__len__(), 1000)
        df_rnd_sample = df_train.iloc[rnd, :]
        class_labels = df_train['status'].unique().tolist()
        model, history, loss, acc = train_nn(df_rnd_sample, 0.33, 50, 32, False, [20, 16, class_labels.__len__()])
        if pd.notnull(loss) and acc > 0.8:
            models.append(model)
            histories.append(history)
            losses.append(loss)
            accuracies.append(acc)
        if models.__len__() >= 3:
            break
    return models, losses, accuracies, histories


def plot_learning_curve(history):
    pyplot.title('Learning Curves')
    pyplot.xlabel('Epoch')
    pyplot.ylabel('Cross Entropy')
    pyplot.plot(history.history['loss'], label='loss')
    pyplot.plot(history.history['accuracy'], label='accuracy')
    pyplot.legend()
    pyplot.show()


if __name__ == "__main__":
    df = pd.read_csv('/Volumes/G-DRIVE mobile/Data/FannieMae/2019Q1/Performance_2019Q1.txt',
                     sep='|', index_col=False,
                     names=['loan_identifier', 'monthly_reporting_period', 'servicer_name', 'current_interest_rate',
                            'current_actual_upb', 'loan_age', 'remaining_months_to_legal_maturity',
                            'adjusted_remaining_months_to_maturity',
                            'maturity_date', 'msa', 'current_loan_delinquency_status', 'modification_flag',
                            'zero_balance_code',
                            'zero_balance_effective_date', 'last_paid_intallment_date', 'foreclosure_date',
                            'disposition_date',
                            'foreclosure_costs', 'property_preservation_and_repair_costs', 'asset_recovery_costs',
                            'miscellaneous_holding_expenses_and_credits',
                            'associated_taxes_for_holding_property', 'net_sale_proceeds',
                            'repurchase_make_whole_proceeds',
                            'other_foreclosure_proceeds', 'non_interest_bearing_upb', 'principal_foregiveness_upb',
                            'repurchase_make_whole_proceeds_flag', 'foreclosure_principal_write_off_amount',
                            'servicing_activity_indicator'])

    # Goal: Retain the latest monthly performance record for each loan.
    gr = df.groupby(['loan_identifier'])['monthly_reporting_period'].max()

    # gr is a series type object here. Convert it to data frame.
    df1 = pd.DataFrame(gr, columns=['monthly_reporting_period'])

    # Make index of df1 as column 'loan_identifier' and change index.
    df1['loan_identifier'] = df1.index
    df1.index = range(df1.shape[0])

    # Now join df1 with df on both columns 'loan_identifier' and 'monthly_reporting_period'.
    result = pd.merge(df, df1, on=['loan_identifier', 'monthly_reporting_period'])
    result['status'] = add_status_column(result)

    df_loans = pd.read_csv('/Volumes/G-DRIVE mobile/Data/FannieMae/2019Q1/Acquisition_2019Q1.txt',
                           sep='|', index_col=False,
                           names=['loan_identifier', 'channel', 'seller_name', 'original_interest_rate',
                                  'original_upb', 'original_loan_term', 'origination_date', 'first_paymane_date',
                                  'ltv', 'cltv', 'number_of_borrowers', 'dti', 'borrower_credit_score',
                                  'first_time_home_buyer_indicator', 'loan_purpose', 'property_type', 'number_of_units',
                                  'occupancy_status', 'property_state', 'zip_3_digit', 'mortgage_insurance_percentage',
                                  'product_type', 'co_borrower_credit_score', 'mortgage_insurance_type',
                                  'relocation_mortgage_indicator'])

    result = pd.concat([result, df_loans], axis=1, join='inner')

    df_train = result[
        ['current_interest_rate', 'current_actual_upb', 'loan_age', 'ltv', 'dti', 'property_state', 'status']]

    le_us_state = LabelEncoder()
    le_us_state.fit_transform(df_train['property_state'])
    df_train['property_state'] = le_us_state.transform(df_train['property_state'])

    # We need this later for getting categorical class label.
    le_status = LabelEncoder()
    le_status.fit_transform(df_train['status'])

    # Iteratively develop models - call learn
    models, losses, accurecies, histories = learn(df_train)
    # Predict
    row = [4.25, 384700.00, 31, 85, 45.0, 4]
    for i in range(models.__len__()):
        model = models[i]
        loss = losses[i]
        acc = accurecies[i]
        yhat = model.predict([row])
        print('Predicted: %s (class=%d)' % (yhat, argmax(yhat)))
        print('Predicted class: %s' % le_status.classes_[argmax(yhat)])
        print('Accuracy: %.3f Loss: %.3f' % (acc, loss))
        plot_learning_curve(histories[i])
