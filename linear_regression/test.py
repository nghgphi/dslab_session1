from utils import *
from model import *
from prepare_data import *
from tqdm import tqdm



x, y = prepare(r'linear_regression/data.txt')

x = normalize_and_add_ones(x)
x_train, y_train = x[:50], y[:50]
x_test, y_test = x[50:], y[50:]

ridge_regression = RidgeRegression()
best_lambda = ridge_regression.get_the_best_LAMBDA(x_train= x_train, y_train= y_train)
print('Best lambda:', best_lambda)

# w_learned = ridge_regression.fit_gradient_descent(x_train, y_train, best_lambda, 0.01, max_num_epoch= 1000000, batch_size= 10)
w_learned = ridge_regression.fit(x_train, y_train, best_lambda)
y_predict = ridge_regression.predict(w_learned, x_test)

print(ridge_regression.compute_RSS(y_test, y_predict))
