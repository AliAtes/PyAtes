import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class LinearRegression:

    def __int__(self):
        i = 1

    def multiple_array_linear_regression(self):
        data = {
            'b0': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            'x1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
            'x2': [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
            'y': [3.8, 6.5, 3.2, 5.1, 8.5, 5.2, 7.8, 9.5, 6.2, 7.8, 11.5, 9.2, 12.8, 11.5, 13.2, 13.8, 16.5, 13.2, 17.8, 16.3]}

        df = pd.DataFrame(data)
        X = df[['b0', 'x1', 'x2']]
        y = df['y']
        b = np.ones(X.shape[1])  # b0, b1, b2
        n = len(y)
        j_mae = np.ones(n)
        j_mse = np.ones(n)
        alpha = 0.001

        for r in range(1000):
            y_head = np.dot(X, b)
            error = y_head - y

            b = b - alpha * 2 / n * np.dot(X.T, error)

            j_mae = 1 / n * np.absolute(error)
            j_mse = 1 / n * np.square(error)

        print("my: sum_J_MAE(x) = ", np.sum(j_mae))
        print("my: sum_J_MSE(x) = ", np.sum(j_mse))

        y_head = np.dot(X, b)
        plt.scatter(X['x1'], y, color="green")
        plt.scatter(X['x2'], y, color="red")
        plt.plot(X['x1'], y_head, color="green")
        plt.plot(X['x2'], y_head, color="red")
        plt.show()

    def multiple_3D_linear_regression(self):

        x1 = [0.5, 0, 1, 5, 8, 4, 10, 7, 3, 2, 12, 10, 14, 6]
        x2 = [22, 21, 23, 25, 28, 23, 25, 29, 22, 23, 32, 30, 34, 27]
        y = [25, 22, 27, 80, 90, 69, 100, 85, 60, 35, 150, 130, 180, 75]

        n = len(x1)
        alpha = 0.001
        b0, b1, b2 = 1, 1, 1
        j_mae = np.ones(n)
        j_mse = np.ones(n)

        for r in range(10000):
            for i in range(n):
                y_head = b0 + b1 * x1[i] + b2 * x2[i]
                error = y_head - y[i]

                b0 = b0 - alpha * 2 / n * error * 1
                b1 = b1 - alpha * 2 / n * error * x1[i]
                b2 = b2 - alpha * 2 / n * error * x2[i]

                j_mae[i] = 1 / n * np.absolute(error)
                j_mse[i] = 1 / n * np.square(error)

        # sklearn library
        from sklearn.linear_model import LinearRegression
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_absolute_error, mean_squared_error

        linear_reg = LinearRegression()
        x_train = np.stack((x1, x2), axis=1)  # X = { [x1[0], x2[0]], [x1[1], x2[1]], [x1[2], x2[2]]... } 
        y_train = np.array(y)
        linear_reg.fit(x_train, y_train)

        y_true = np.array(y)
        y_predict = linear_reg.predict(x_train)

        mae = mean_absolute_error(y_true, y_predict)
        mse = mean_squared_error(y_true, y_predict)

        # print results
        print("my: sum_J_MAE(x) = ", np.sum(j_mae))
        print("sk: sum_J_MAE(x) = ", mae)
        print("my: sum_J_MSE(x) = ", np.sum(j_mse))
        print("sk: sum_J_MSE(x) = ", mse)
        print("my: sum_J_RMSE(x) = ", np.sqrt(np.sum(j_mse)))
        print("sk: sum_J_RMSE(x) = ", np.sqrt(mse))

        y_head = b0 + b1 * np.array(x1) + b2 * np.array(x2)
        plt.ylabel("Y-Label")
        plt.xlabel("X1 (G)   |   X2 (R)")
        plt.scatter(x1, y, color="green")
        plt.scatter(x1, y_head, color="blue")
        plt.scatter(x2, y, color="red")
        plt.scatter(x2, y_head, color="blue")
        plt.show()

    def simple_array_linear_regression(self):
        data = {
            'b0': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            'x':  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
            'y':  [3.8, 6.5, 3.2, 5.1, 8.5, 5.2, 7.8, 9.5, 6.2, 7.8, 11.5, 9.2, 12.8, 11.5, 13.2, 13.8, 16.5, 13.2, 17.8, 16.3]}

        df = pd.DataFrame(data)
        X = df[['b0', 'x']]
        y = df['y']
        b = np.ones(X.shape[1])  # b0, b1
        n = len(y)
        j_mae = np.ones(n)
        j_mse = np.ones(n)
        alpha = 0.001

        for r in range(1000):
            y_head = np.dot(X, b)
            error = y_head - y

            b = b - alpha * 2 / n * np.dot(X.T, error)

            j_mae = 1 / n * np.absolute(error)
            j_mse = 1 / n * np.square(error)

        print("my: sum_J_MAE(x) = ", np.sum(j_mae))
        print("my: sum_J_MSE(x) = ", np.sum(j_mse))

        y_head = np.dot(X, b)
        plt.scatter(X['x'], y, color="green")
        plt.plot(X['x'], y_head, color="red")
        plt.show()

    def simple_linear_regression(self):

        x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        y = [3.8, 6.5, 3.2, 5.1, 8.5, 5.2, 7.8, 9.5, 6.2, 7.8, 11.5, 9.2, 12.8, 11.5, 13.2, 13.8, 16.5, 13.2, 17.8, 16.3]

        n = len(x)  # veri seti boyutu
        a = b = 1  # y = ax + b doğrusunda; a: eğim, b: denklem sabiti
        j_mae = np.ones(n)  # Main Absolute Error ile hesaplanan hataların tutulduğu dizi
        j_mse = np.ones(n)  # Main Square Error ile hesaplanan hataların tutulduğu dizi
        alpha = 0.001  # Gradient Descent'te adım hassasiyeti

        for r in range(10000):
            for i in range(n):
                y_head = a * x[i] + b  # y[i]_predicted, optimize edilen a ve b değerleriyle x[i]'ye karşılık geldiği tahmin edilen y[i] değeri.
                error = y_head - y[i]

                b = b - alpha * 2 / n * error * 1  # b = b - alpha * dy.J/db
                a = a - alpha * 2 / n * error * x[i]  # a = a - alpha * dy.J/da

                j_mae[i] = 1 / n * np.absolute(error)  # Main Absolute Error func: J = 1/n * 1...n |(y_head - y[i])|
                j_mse[i] = 1 / n * np.square(error)  # Main Square Error func:   J = 1/n * 1...n (y_head - y[i])^2

                """print(r, ". sumJ(x) = ", np.sum(J))"""  # yavaş çalışıyor her seferinde çıktı verdiğinde.

        # sklearn library
        from sklearn.linear_model import LinearRegression
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_absolute_error, mean_squared_error

        linear_reg = LinearRegression()
        x_train = np.array(x).reshape((-1, 1))
        y_train = np.array(y)
        linear_reg.fit(x_train, y_train)

        y_true = np.array(y)
        y_predict = linear_reg.predict(x_train)

        mae = mean_absolute_error(y_true, y_predict)
        mse = mean_squared_error(y_true, y_predict)

        # print results
        print("my: sum_J_MAE(x) = ", np.sum(j_mae))
        print("sk: sum_J_MAE(x) = ", mae)
        print("my: sum_J_MSE(x) = ", np.sum(j_mse))
        print("sk: sum_J_MSE(x) = ", mse)
        print("my: sum_J_RMSE(x) = ", np.sqrt(np.sum(j_mse)))
        print("sk: sum_J_RMSE(x) = ", np.sqrt(mse))

        plt.scatter(x, y, color="green")
        plt.plot(x, linear_reg.predict(x_train), color="red")
        plt.plot(x, a * np.array(x) + b, color="yellow")
        plt.show()
