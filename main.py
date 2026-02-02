import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression


#задаем значения
X = np.linspace(5,100,20)
true_w1 = 13.5
true_w0 = -2.0
y = true_w0 + true_w1 * X
#добавление шума 
noise = np.random.normal(0,20,size=y.shape)
y = y + noise

#веса
w0 = 0.0
w1 = 0.0


#предсказание
def predict(X, w0,w1):
    return w0 + w1 * X

#ошибка
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)
#Функция градиента
def gradients(X,y_true,y_pred):
    error = y_pred - y_true
    dw0 = np.mean(error)
    dw1 = np.mean(error * X)
    return dw0, dw1

#Обучающий цикл
lr = 0.0001

loss_history = []

for epoch in range(1000):
    y_pred = predict(X, w0,w1)
    dw0,dw1 = gradients(X,y,y_pred)
    
    w0 -= lr * dw0
    w1 -= lr * dw1
    
    loss = mse(y,y_pred)
    loss_history.append(loss)

#Визуализация
plt.plot(loss_history)
plt.xlabel("Эпоха")
plt.ylabel("Ошибка")
plt.title("Изменения ошибки во время обучения")
plt.grid(True)
plt.show()

plt.scatter(X, y, color='blue', label='Real Data')
plt.plot(X, predict(X, w0,w1), color='red', label='Model')
plt.xlabel("Сложность босса")
plt.ylabel("Награда за босса")
plt.title("Модель vs Реальные данные")
plt.legend()
plt.grid()
plt.show()

print(f"Pred: {y_pred}\nOrg.y:{y}\nLoss:{mse(y,y_pred)}\nw1={w1}\nw0={w0}")


#Линейная регрессия
model = LinearRegression()
model.fit(X.reshape(-1,1),y)

print(f"sklearn w1: {model.coef_[0]}")
print(f"sklearn w0: {model.intercept_}")