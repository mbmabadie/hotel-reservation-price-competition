from sklearn.metrics import mean_squared_error
import pandas as pd

y1 = pd.read_csv('../results/deep_reg.csv')
y2 = pd.read_csv('../results/deep_reg_with_distance.csv')

print(mean_squared_error(y1['price'].values.ravel(), y2['price'].values.ravel(), squared=False))
