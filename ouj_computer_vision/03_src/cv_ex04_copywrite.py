# %%
import numpy as np
import matplotlib.pyplot as plt

# %% innner product of 2 vectors
x = np.array([1, 2, 3])
y = np.array([4, 5, 6])

print(np.dot(x, y))
# it is same to np.sum()
print(np.sum(x * y))

# %% norm
x = np.array([1, 1, -1, -1, 1, 1, -1, -1])
l1_norm = np.linalg.norm(x, 1)
l2_norm = np.linalg.norm(x, 2)

print(f'l1 norm: {l1_norm}\nl2 norm: {l2_norm}')# %%

# %% outer product
x = np.array([[1, 2, 3],
             [4, 5, 6]])
y = np.array([[2, 1],
             [2, 1],
             [2, 1]])

np.dot(x, y)

# %%
print(x.T)
# %% indentity matrix 
np.eye(2)

# %% inverse matrix
x = np.array([[2, 1],
              [1, 1]])

print(np.linalg.inv(x))

# %%
x = np.array([[2, 3],           # 2×2の正方行列
              [4, 5]])
print(np.linalg.det(x))         # 行列式が0にならない
y = np.array([[0, 2],           # 2×2の正方行列
              [0, 1]])
print(np.linalg.det(y))         # 行列式が0になる