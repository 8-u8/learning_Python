# %%
from ortools.linear_solver import pywraplp


# %%

name_list =  ['xA', 'xB', 'xC', 'xD', 'xE', 'xF', 'xG', 'xT']
def create_model():
    data = {}
    # [xA, xB, xC, xD, xE, xF, xG, xT]
    data['constraint_coeffs'] = [
      # [xA, xB, xC, xD, xE, xF, xG, xT]
        [1, 0, 0, 0, 0, 0, 0, 0], # const xA
        [0, 1, 0, 0, 0, 0, 0, 0], # const xB
        [-1, 0, 1, 0, 0, 0, 0, 0], # const xC-A
        [0, -1, 1, 0, 0, 0, 0, 0], # const xC-B
        [0, -1, 0, 1, 0, 0, 0, 0], # const xD-B
        [0, 0, -1, 0, 1, 0, 0, 0], # const xE-C
        [0, 0, -1, 0, 0, 1, 0, 0], # const xF-C
        [0, 0, 0, -1, 0, 1, 0, 0], # const xF-D
        [0, 0, 0, 0, -1, 0, 1, 0], # const xG-E
        [0, 0, 0, 0, 0, -1, 1, 0], # const xG-F
        [0, 0, 0, 0, 0, 0, -1, 1]  # const xT
    ]
    data['bounds'] = [0, 0, 4, 5, 5, 7, 7, 7, 2, 9, 2]

    data['obj_coeffs'] = [1, 1, 1, 1, 1, 1, 1, 1]
    data['num_vars'] = 8
    data['num_constraints'] = 11

    return data


# %%
data = create_model()
solver = pywraplp.Solver.CreateSolver('SAT')

# init
infinity = solver.infinity()
x = {}

# define variables.
for i in range(data['num_vars']):
    x[i] = solver.IntVar(0, infinity, 'x[%i]' % i)
    tmp_keys = list(x.keys())[i]
    x[name_list[i]] = x.pop(tmp_keys)

print('Numver of variables = ', solver.NumVariables())


# %%
for i in range(data['num_constraints']):
    constraint = solver.RowConstraint(data['bounds'][i], infinity, '')

    for j in range(data['num_vars']):
        tmp_idx = list(x.keys())[j]
        constraint.SetCoefficient(x[tmp_idx], data['constraint_coeffs'][i][j])

print('Number of constraints = ', solver.NumConstraints())

# %%
objective = solver.Objective()
for j in range(data['num_vars']):
    tmp_idx = list(x.keys())[j]
    objective.SetCoefficient(x[tmp_idx], data['obj_coeffs'][j])

objective.SetMinimization()

# %%
status = solver.Solve()

# %%
if status == pywraplp.Solver.OPTIMAL:
    print('Objective value =', solver.Objective().Value())
    for j in name_list:
        tmp_idx = j
        print(j, ' = ', x[j].solution_value())
    print()
    print('Problem solved in %f milliseconds' % solver.wall_time())
    print('Problem solved in %d iterations' % solver.iterations())
    print('Problem solved in %d branch-and-bound nodes' % solver.nodes())
else:
    print('The problem does not have an optimal solution.')
# %%
