from ortools.linear_solver import pywraplp
from ortools.init import pywrapinit

def main():

    solver = pywraplp.Solver.CreateSolver('GLOP')
    if not solver:
        return
    
    x = solver.NumVar(0, 1, 'x')
    y = solver.NumVar(0, 2, 'y')

    print('Number of variables: ', solver.NumVariables())

    ct = solver.Constraint(0, 2, 'ct')
    ct.SetCoefficient(x, 1)
    ct.SetCoefficient(y, 1)

    print('Number of constraints: ', solver.NumConstraints())

    objective = solver.Objective()
    objective.SetCoefficient(x, 3)
    objective.SetCoefficient(y, 1)
    objective.SetMaximization()

    solver.Solve()
    x_out = x.solution_value()
    y_out = y.solution_value()
    print('Solution: ')
    print('Objective Value: ', objective.Value())
    print(f'x = {x_out}')
    print(f'y = {y_out}')


if __name__ == '__main__':
    pywrapinit.CppBridge.InitLogging('basic_example.py')
    cpp_flags = pywrapinit.CppFlags()
    cpp_flags.logtostderr = True
    cpp_flags.log_prefix = False
    pywrapinit.CppBridge.SetFlags(cpp_flags)

    main()