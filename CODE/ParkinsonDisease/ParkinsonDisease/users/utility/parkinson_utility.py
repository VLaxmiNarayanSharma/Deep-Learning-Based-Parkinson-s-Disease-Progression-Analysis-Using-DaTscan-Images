import os

import pandas as pd
from django.conf import settings


path = os.path.join(settings.MEDIA_ROOT, 'parkinsons.csv')
a = pd.read_csv(path)
a.head()
a.shape
a.dtypes
# print(a.head())

# sns.catplot(x='status', kind='count', data=a)
# for i in a:
# if i != 'status' and i != 'name':
# sns.catplot(x='status', y=i, kind='box', data=a)
b = a.drop(['name'], axis=1)
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score

features = a.drop(['status', 'name'], axis=1)
labels = a['status']
scaler = MinMaxScaler((-1, 1))
x = scaler.fit_transform(features)
y = labels
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=5)


def start_models():
    from sklearn.linear_model import LogisticRegression
    from xgboost import XGBRFClassifier, XGBClassifier
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, \
        GradientBoostingClassifier, RandomForestClassifier
    lr = cross_val_score(LogisticRegression(), x_train, y_train)
    xgbc = cross_val_score(XGBRFClassifier(), x_train, y_train)
    xgb = cross_val_score(XGBClassifier(), x_train, y_train)
    svm = cross_val_score(SVC(), x_train, y_train)
    dtc = cross_val_score(DecisionTreeClassifier(), x_train, y_train)
    adb = cross_val_score(AdaBoostClassifier(), x_train, y_train)
    bbc = cross_val_score(BaggingClassifier(), x_train, y_train)
    etc = cross_val_score(ExtraTreesClassifier(), x_train, y_train)
    gbc = cross_val_score(GradientBoostingClassifier(), x_train, y_train)
    rfc = cross_val_score(RandomForestClassifier(), x_train, y_train)

    print('log reg', lr, lr.mean())
    print('xgbd', xgbc, xgbc.mean())
    print('xgb', xgb, xgb.mean())
    print('svm', svm, svm.mean())
    print('dtc', dtc, dtc.mean())
    print('adb', adb, adb.mean())
    print('bbc', bbc, bbc.mean())
    print('etc', etc, etc.mean())
    print('gbc', gbc, gbc.mean())
    print('rfc', rfc, rfc.mean())

    model = XGBClassifier()
    model.fit(x_train, y_train)
    accuracy_dict = {}

    # y_predtr = model.predict(x_train)
    # print(accuracy_score(y_train, y_predtr) * 100)

    y_pred = model.predict(x_test)
    xg_accuracy = accuracy_score(y_test, y_pred) * 100
    accuracy_dict.update({'xg_accuracy': xg_accuracy})

    model = ExtraTreesClassifier()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    etc_accuracy = accuracy_score(y_test, y_pred) * 100
    accuracy_dict.update({'etc_accuracy': etc_accuracy})

    from sklearn.metrics import classification_report, confusion_matrix
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

    model = AdaBoostClassifier()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    ada_accuracy = accuracy_score(y_test, y_pred) * 100
    accuracy_dict.update({'ada_accuracy': ada_accuracy})
    model = SVC(kernel='rbf')
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    svc_accuracy = accuracy_score(y_test, y_pred) * 100
    accuracy_dict.update({'svc_accuracy': svc_accuracy})

    model = RandomForestClassifier()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    rf_accuracy = accuracy_score(y_test, y_pred) * 100
    accuracy_dict.update({'rf_accuracy': rf_accuracy})

    ## ANN
    import keras
    from keras.models import Sequential
    from keras.layers import Dense

    classifier = Sequential()

    classifier.add(Dense(output_dim=6, init='uniform', activation='relu', input_dim=22))
    classifier.add(Dense(output_dim=6, init='uniform', activation='relu'))
    classifier.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print(classifier.summary())
    classifier.fit(x_train, y_train, batch_size=10, nb_epoch=100)

    y_pred = classifier.predict(x_test)
    y_pred = (y_pred > 0.5)
    ann_accuracy = accuracy_score(y_test, y_pred) * 100
    accuracy_dict.update({'ann_accuracy': ann_accuracy})
    return accuracy_dict
