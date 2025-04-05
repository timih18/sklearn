import pandas
from joblib import load
from sklearn.preprocessing import StandardScaler

model = load('model.joblib')
data = pandas.read_csv('<PATH>')
result = data
data = data.drop(columns=['Row_id'])
data.info()

scaler = StandardScaler()
data = scaler.fit_transform(data)
prediction = model.predict(data)
result['predicted'] = prediction
result['predict'] = result['predicted'].map({0 : 'GALAXY', 1 : 'STAR', 2 : 'QSO'})
result.to_csv('output.csv', index=False)