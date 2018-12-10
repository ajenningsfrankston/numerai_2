
import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.wrappers.scikit_learn import KerasClassifier
from check_consistency import check_consistency

from pridge_classifier import PRidgeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier

from numerapi import NumerAPI


def make_prediction(train,validation,tournament,comp_no,comp_names):

    # Construct a prediction for a particular competition
    #

    discard_core = ['id', 'era', 'data_type']

    comp_name = comp_names[comp_no]

    target_list = list()
    for name in comp_names:
        if name != comp_name:
            target_list.append('target_' + name)

    discard_list = target_list + discard_core

    train_comp = train.drop(discard_list,axis=1)

    # Transform the loaded CSV data into numpy arrays

    features = [f for f in list(train_comp) if "feature" in f]
    X = train_comp[features]
    Y = train_comp['target_' + comp_name]
    x_prediction = validation[features]
    ids = tournament['id']

    rdc = PRidgeClassifier(alpha=1.0)
    lrc = LogisticRegression()

    rdc.fit(X.values,Y.values)
    lrc.fit(X.values,Y.values)

    # keras parameters

    batch_size = 256
    epochs = 16

    def create_model(neurons=20, dropout=0.1):
        model = Sequential()
        # we add a vanilla hidden layer:
        model.add(Dense(neurons))
        model.add(Activation('sigmoid'))
        model.add(Dropout(dropout))
        model.add(Dense(neurons))
        model.add(Activation('sigmoid'))
        model.add(Dropout(dropout))

        model.add(Dense(1, activation='sigmoid', kernel_initializer='glorot_normal'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_crossentropy', 'accuracy'])
        return model

    keras_model = KerasClassifier(build_fn=create_model, epochs=epochs, batch_size=batch_size, verbose=2)

    keras_model.fit(X.values,Y.values)

    model_list = [lrc,rdc,keras_model]

    consistencies = [ check_consistency(model, validation, train) for model in model_list]

    print("Consistencies: ",format(consistencies))

    most_consistent_model = model_list[consistencies.index(max(consistencies))]

    # bagging classifier

    model = BaggingClassifier(base_estimator=most_consistent_model, n_estimators=10)
    model.fit(X.values,Y.values)

    # check consistency
    final_model = model
    consistency = check_consistency(final_model, validation, train)
    print("Consistency: {}".format(consistency))

    x_prediction = tournament[features]
    t_id = tournament["id"]
    raw_predict = final_model.predict_proba(x_prediction.values)
    y_prediction = raw_predict[:,1]
    results = np.reshape(y_prediction, -1)
    results_df = pd.DataFrame(data={'probability_'+comp_name: results})
    joined = pd.DataFrame(t_id).join(results_df)
    filename = comp_name + '_predictions.csv'
    path = './tmp/numerai_predictions/' + filename
    print()
    print("Writing predictions to " + path.strip())
    # # Save the predictions out to a CSV file
    joined.to_csv(path, float_format='%.5f', index=False)

    return path
