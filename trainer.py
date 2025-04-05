import pandas
from joblib import dump
from datetime import datetime

from numpy.f2py.crackfortran import verbose
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier


data = pandas.read_csv('public_data.csv')
data['class'] = data['class'].map({'GALAXY' : 0, 'STAR' : 1, 'QSO' : 2})
data = data.drop('Row_id', axis = 1)
print('Информация о датасете')
data.info()

x = data.drop(columns=['class'])
y = data['class']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

param_grid = {
    'n_estimators' : [100, 200, 300],
    'learning_rate' : [0.01, 0.05, 0.1],
    'max_depth' : [-1],
    'num_leaves' : [31, 127, 255],
    'objective' : ['multiclass'],
    'class_weight' : ['balanced'],
    'random_state' : [42],
    'verbose' : [-1]
}
model = GridSearchCV(
    estimator=LGBMClassifier(),
    param_grid=param_grid,
    cv=3,
    scoring='accuracy',
    n_jobs=-1
)

print('\n', 'Начало обучения')
start_time = datetime.now()
model.fit(pandas.DataFrame(X_train), y_train)
end_time = datetime.now()
print(f'Модель была обучена за {end_time - start_time}')

prediction = model.predict(pandas.DataFrame(X_test))
accuracy = accuracy_score(y_test, prediction)
print('\n', f'Точность модели: {accuracy * 100:.4f}%')

dump(model, 'model.joblib')
