from pycgp.benchmarks.symbolic import X, y
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor

rgr = MLPRegressor(max_iter=100000, solver='lbfgs', hidden_layer_sizes=1000)
print('start fit')
rgr.fit(X, y)
print(mean_squared_error(y, rgr.predict(X)))