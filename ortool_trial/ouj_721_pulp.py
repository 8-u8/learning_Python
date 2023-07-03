# %%
import pulp

# %% init
Task = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'T']
Time = {
    'A': 4,
    'B': 5,
    'C': 7,
    'D': 7,
    'E': 2,
    'F': 9,
    'G': 2
}

Pre = {
    'A': [],
    'B': [],
    'C': ['A', 'B'],
    'D': ['B'],
    'E': ['C'],
    'F': ['C', 'D'],
    'G': ['E', 'F'],
    'T': ['G']
}

# %% solver definition
solver = {
    t: pulp.LpVariable(f'solver{t}') for t in Task
}

p = pulp.LpProblem('most fast finish time', sense=pulp.LpMinimize)
p += solver['T'], 'objective the time of beginning of T'

for t in Task:
    if Pre[t] == []:
        p += solver[t] >= 0, f'work {t} does not have pre-work.'
    else:
        for pre in Pre[t]:
            p += solver[t] >= solver[pre] + Time[pre], f'work {t} will start after work {pre}'

p
# %%
result = p.solve()

# %% 
pulp.LpStatus[result]
# %%
print(p.objective)
for v in p.variables():
    print(f'{v} = {pulp.value(v)}')
