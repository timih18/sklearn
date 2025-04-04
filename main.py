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

result = X_test
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

param_grid = {
    'n_estimators' : [100, 200, 300, 500],
    'learning_rate' : [0.05, 0.1, 0.3],
    'max_depth' : [-1],
    'num_leaves' : [31, 127, 255],
    'objective' : ['multiclass'],
    'reg_alpha' : [0, 0.1, 0.5],
    'reg_lambda' : [0, 0.1, 0.5],
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
print('\n', f'Точность модели {accuracy * 100:.4f}%')


processed_y_test = ['GALAXY' if i == 0 else 'STAR' if i == 1 else 'QSO' for i in y_test]
processed_prediction = ['GALAXY' if i == 0 else 'STAR' if i == 1 else 'QSO' for i in prediction]
order = ['Row_id', 'obj_ID', 'alpha', 'delta', 'u', 'g', 'r', 'i', 'z', 'run_ID', 'rerun_ID',
         'cam_col', 'field_ID', 'spec_obj_ID', 'class', 'redshift', 'plate', 'MJD', 'fiber_ID', 'predicted']
result['Row_id'] = result.index + 1
result['class'] = processed_y_test
result['predicted'] = processed_prediction
result = result.reindex(columns=order)
result.to_csv('output.csv')

dump(model, 'model.joblib')
