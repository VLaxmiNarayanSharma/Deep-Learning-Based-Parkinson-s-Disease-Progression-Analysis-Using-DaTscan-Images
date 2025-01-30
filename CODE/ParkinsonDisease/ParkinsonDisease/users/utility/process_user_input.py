import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from django.conf import settings

path = os.path.join(settings.MEDIA_ROOT, 'parkinsons.csv')
a = pd.read_csv(path)
a = a[['NHR', 'HNR', 'RPDE', 'DFA', 'PPE', 'status']]
a.head()
a.shape
a.dtypes
# print(a.head())

sns.catplot(x='status', kind='count', data=a)
plt.show()
for i in a:
    if i != 'status' and i != 'PPE':
        sns.catplot(x='status', y=i, kind='box', data=a)
        plt.show()
# b = a.drop(['name'], axis=1)
data = a.drop('status', axis=1)
sns.heatmap(data.corr(), annot=True)
plt.show()
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

features = a.drop(['status'], axis=1)
labels = a['status']
scaler = MinMaxScaler((-1, 1))
x = scaler.fit_transform(features)
y = labels
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=5)


def get_result(test_set):
    from xgboost import XGBClassifier
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier()
    model.fit(x_train, y_train)
    scaler = MinMaxScaler((-1, 1))
    x = scaler.fit_transform([test_set])
    y_pred = model.predict(x)
    # y_pred = model.predict(x_test)
    print(y_pred)
    return y_pred
