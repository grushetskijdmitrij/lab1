import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

df = pd.read_csv("fifa_ranking_2022-10-06.csv")

print("Первые строки датасета:")
print(df.head())
print(df.info())

X = df[['previous_points','previous_rank']]

Y = df['points']

print("\nX:")
print(X.head())

print("\nY:")
print(Y.head())

X_train, X_test, Y_train, Y_test = train_test_split(
    X,Y,
    test_size=0.2,
    random_state=42
)

print("\nОбучающая выборка:",X_train.shape)
print("\nТестовая выборка", X_test.shape)

model = LinearRegression()
#после этой функции модель обучена
model.fit(X_train,Y_train)

y_pred = model.predict(X_test)

print("\nПредсказанные значения:")
print(y_pred[:5])

print("\nРеальные значения:")
print(Y_test[:5])

mae = mean_absolute_error(Y_test,y_pred)
mse = mean_squared_error(Y_test, y_pred)
r2 = r2_score(Y_test, y_pred)

print("MAE:",mae)
print("MSE:",mse)
print("R2",r2)

df['top_team'] = (df['rank'] <= 50).astype(int)

print(df[['rank','top_team']].head(10))

X_class = df[['previous_points','previous_rank','points']]

Y_class = df['top_team']

X_train_c,X_test_c,Y_train_c,Y_test_c = train_test_split(
    X_class, Y_class,
    test_size=0.2,
    random_state=42
)

log_model = LogisticRegression()
#также обучение
log_model.fit(X_train_c,Y_train_c)
