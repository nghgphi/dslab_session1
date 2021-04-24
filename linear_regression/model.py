import numpy as np
from numpy.lib.function_base import average

class RidgeRegression:
    def __init__(self):
        return
    def fit(self, x_train, y_train, LAMBDA):
        assert (len(x_train.shape) == 2) and (x_train.shape[0] == y_train.shape[0])
        w = np.linalg.inv(x_train.transpose().dot(x_train) +
        LAMBDA * np.identity(x_train.shape[1])).dot(x_train.transpose()).dot(y_train)
        return w
    def predict(self, w, x_new):
        x_new = np.array(x_new)
        y = x_new.dot(w)
        return y
    def compute_RSS(self, y_new, y_predicted):
        loss = 1 / y_new.shape[0] * (np.sum((y_new - y_predicted) ** 2))
        return loss
    def get_the_best_LAMBDA(self, x_train, y_train):
        def cross_validation(num_folds, LAMBDA):
            row_ids = np.array(range(x_train.shape[0]))
            valids_ids = np.split(row_ids[: len(row_ids) - len(row_ids) % num_folds], num_folds)
            valids_ids[-1] = np.append(valids_ids[-1], row_ids[len(row_ids) - len(row_ids) % num_folds:])
            train_ids = [[k for k in row_ids if k not in valids_ids[i]] for i in range(num_folds)]
            average_rss = 0

            for i in range(num_folds):
                valid_part = {'X': x_train[valids_ids[i]], 'Y': y_train[valids_ids[i]]}
                train_part = {'X': x_train[train_ids[i]], 'Y': y_train[train_ids[i]]}
                w = self.fit(train_part['X'], train_part['Y'], LAMBDA)
                y_predict = self.predict(w, valid_part['X'])
                average_rss += self.compute_RSS(valid_part['Y'], y_predict)
            return average_rss / num_folds
        def range_scan(best_lambda, minimun_rss, lambda_values):
            for current_lambda in lambda_values:
                aver_rss = cross_validation(num_folds= 5, LAMBDA= current_lambda)
                if aver_rss < minimun_rss:
                    best_lambda = current_lambda
                    minimun_rss = aver_rss
            return best_lambda, minimun_rss
        best_lambda, minimun_rss = range_scan(best_lambda= 0, minimun_rss= 10000 ** 2, lambda_values= range(50))

        lambda_values = [k * 1./1000 for k in range(
            max(0, (best_lambda - 1) * 1000, (best_lambda + 1) * 1000, 1))]
        best_lambda, minimun_rss = range_scan(best_lambda= best_lambda, minimun_rss= minimun_rss, lambda_values= lambda_values)

        return best_lambda
    def fit_gradient_descent(self, x_train, y_train, LAMBDA, learning_rate, max_num_epoch = 1000, batch_size = 6):
        w = np.random.randn(x_train.shape[1])
        last_lost = 10e+8
        for ep in range(max_num_epoch):
            arr = np.array(range(x_train.shape[0]))
            np.random.shuffle(arr)
            x_train = x_train[arr]
            y_train = y_train[arr]
            
            total_minibatch = int(np.ceil(x_train.shape[0] / batch_size))

            for i in range(total_minibatch):
                index = i * batch_size
                x_train_sub = x_train[index : index + batch_size]
                y_train_sub = y_train[index : index + batch_size]

                grad = x_train_sub.T.dot(x_train_sub.dot(w) - y_train_sub) + LAMBDA * w
                w = w - learning_rate * grad

            new_loss = self.compute_RSS(self.predict(w, x_train), y_train)
            if np.abs(new_loss - last_lost) <= 1e-5:
                break
            last_lost = new_loss
        return w
            
            


