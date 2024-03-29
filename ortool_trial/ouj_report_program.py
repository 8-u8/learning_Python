# %%
from ortools.linear_solver import pywraplp

# %% create data 

def create_data_model():
    data = {}
    data['constraint_coeffs'] = [
        # [x11, x13,
        #  x22, x24, x25,
        #  x31, x33, x34,
        #  x43, x45,
        #  x51, x52, x55]
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # const1
        [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0], # const2
        [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0], # const3
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0], # const4
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1], # const5
        [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0], # const6
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], # const7
        [0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0], # const8
        [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0], # const9
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1]  # const10
    ]
    # the bounds of constraints.
    data['bounds'] = [1, 1, 1, 1, 1]
    # the coefficients of objective function
    data['obj_coeffs'] = [5, 1, 2, 8, 7, 4, 6, 7, 9, 4, 7, 8, 6]
    data['num_vars'] = 13
    data['num_constraints'] = 5
    return data

# %% 
data = create_data_model()

solver = pywraplp.Solver.CreateSolver('GLOP')

x = {}
name_list = ['x11', 'x13', 'x22', 'x24', 'x25', 'x31', 'x33', 'x34', 
             'x43', 'x45', 'x51', 'x52', 'x55']
for i in range(data['num_vars']):
    x[i] = solver.IntVar(0, 1, 'x[%i]' % i)
    tmp_keys = list(x.keys())[i]
    x[name_list[i]] = x.pop(tmp_keys)
print('Number of variables = ', solver.NumVariables())

# %% define constraints 
# 制約条件に対応する形で書けていない。
for i in range(data['num_constraints']):
    constraint = solver.RowConstraint(0, data['bounds'][i], '')
    for j in range(data['num_vars']):
        tmp_idx = list(x.keys())[j]
        constraint.SetCoefficient(x[tmp_idx], data['constraint_coeffs'][i][j])
print('Number of constraints = ', solver.NumConstraints())

# %% define objectives

objective = solver.Objective()
for j in range(data['num_vars']):
    tmp_idx = list(x.keys())[j]
    objective.SetCoefficient(x[tmp_idx], data['obj_coeffs'][j])

objective.SetMaximization()
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
