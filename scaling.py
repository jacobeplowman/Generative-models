import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Example dataset: 5 samples, 10 features
data = np.array([
    [1, 200, 3, 40, 5, 60, 7, 80, 9, 100],
    [2, 190, 4, 50, 6, 70, 8, 90, 10, 110],
    [3, 180, 5, 60, 7, 80, 9, 100, 11, 120],
    [4, 170, 6, 70, 8, 90, 10, 110, 12, 130],
    [5, 160, 7, 80, 9, 100, 11, 120, 13, 140]
])

# Use MinMaxScaler to scale each feature to [0, 1]
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

print("Original Data:")
print(data)

print("\nScaled Data:")
print(data_scaled)


#Once you have generated samples you can unscale the data back to the original scale using the inverse_transform method of the MinMaxScaler object. Here's how you can do it:

unscaled_data = scaler.inverse_transform(generated_data)
print(unscaled_data)