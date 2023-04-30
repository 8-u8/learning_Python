# %%
import pulp

# %%
cost = {
    1:18,
    2:12,
    3:14,
    4:19
}

weight = {
    1:22,
    2:9,
    3:13,
    4:24
}

# %% 
x = {i:pulp.LpVariable(f'x{i}', cat=pulp.LpBinary) for i in cost.keys()}
x
# {1: x1, 2: x2, 3: x3, 4: x4, 5: x5}