# %% 
import pulp

# %% init

cost = {
    1:18,
    2:12,
    3:14,
    4:19,
    5:11,
    6:15
}

weight = {
    1:22,
    2:9,
    3:13,
    4:24,
    5:21,
    6:14
}

# %% 
x = {i:pulp.LpVariable(f'x{i}', cat=pulp.LpBinary) for i in cost.keys()}
x
# {1: x1, 2: x2, 3: x3, 4: x4, 5: x5}

# %%
problem = pulp.LpProblem('Knapsack Problem', sense=pulp.LpMaximize)
problem += pulp.lpSum(cost[i] * x[i] for i in x.keys()) , 'the sum value of objective'
problem += pulp.lpSum(weight[i] * x[i] for i in x.keys())<= 60, 'constraints for weights'

result = problem.solve()
# %% chack results
pulp.LpStatus[result]
pulp.value(problem.objective)
for v in problem.variables():
    print(f'{v} = {pulp.value(v):.0f}')
# %%
