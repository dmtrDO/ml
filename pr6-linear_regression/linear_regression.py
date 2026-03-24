
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# https://www.kaggle.com/datasets/abhishek14398/salary-dataset-simple-linear-regression?select=Salary_dataset.csv
file = "salary.csv"
df = pd.read_csv(file)

# перемішуємо дані
df = df.sample(frac=1, random_state=42)

# ділимо на ознаки та мітки
X = df.iloc[:, 0]
y = df.iloc[:, 1]

# ділимо на тренувальну та тестову вибірки
num_of_train = int(len(X) * 0.85)
X_train, y_train = X.iloc[:num_of_train].to_numpy(), y.iloc[:num_of_train].to_numpy()
X_test, y_test = X.iloc[num_of_train:len(X)].to_numpy(), y.iloc[num_of_train:len(y)].to_numpy()


class LinearRegression():
    def __init__(self, num_of_features):
        # псевдогенератор з деяким зерном щоб можна було
        # відтворити результати
        self.__rng = np.random.default_rng(seed=42)

        # ініціалізуємо ваги та зміщення
        self.__weights = self.__rng.random(num_of_features)
        self.__bias = self.__rng.random()

    def train(self, X_train, y_train):
        # y = w1x1 + w2x2 + w3x3 ... + wnxn + bias
        # l = loss function (mse)
        # l = 1/n * sum((yi - y_predi)**2) (i = 1...n)
        # grad = (dl/dw1; dl/dw2 ... ; dl/dwn)

        # для мінімізації функції втрат будемо використовувати
        # градієнтний спуск

        epochs = 10
        lr = 0.001
        print("Training...")
        for epoch in range(epochs):
            loss = 0 # виключно для логів на кожній епосі
            length = len(X_train)
            dw = np.zeros(self.__weights.shape[0])
            db = 0
            for i in range(length):
                error = y_train[i] - self.predict(X_train[i])
                dw += -2 * error * X_train[i]
                db += -2 * error
                loss += error ** 2
            
            loss /= length
            dw /= length
            db /= length

            print(f"epoch {epoch}: {loss}")

            self.__weights -= lr * dw
            self.__bias -= lr * db            
            
    # повертає значення функції y = kx + b (звичайна лінія) 
    def predict(self, x):
        return np.dot(x, self.__weights) + self.__bias


linear_regression_model = LinearRegression(1)
linear_regression_model.train(X_train, y_train)

# тестування та оцінка точності через MSE
y_pred = [linear_regression_model.predict(x) for x in X_test]
mse_test = np.mean((y_test - y_pred)**2)
print(f"Test MSE: {mse_test}")

# візуалізація
plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, color='blue', label='Train data', alpha=0.5)
plt.scatter(X_test, y_test, color='green', label='Test data', alpha=0.8)

X_line = np.array([X.min(), X.max()])
y_line = [linear_regression_model.predict(x) for x in X_line]

plt.plot(X_line, y_line, color='red', linewidth=3, label='Regression Line')

plt.title('Salary vs Experience')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.legend()
plt.grid(True)
plt.show()



