import numpy as np

a = np.array([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]], [[5, 5, 5], [6, 6, 6, ]], [[5, 5, 5], [6, 6, 6]]])
print(a.shape)

b = np.array([5, 5, 5])
print(np.all(a == b, axis=2))
print(np.any(np.all(a == b, axis=2)))