from ortools.linear_solver import pywraplp
# from ortools.init import pywrapinit


def LinearProgrammingExample():
    """線形計画問題の例"""
    # GLOPソルバでインスタンスを設定
    solver = pywraplp.Solver.CreateSolver('GLOP')
    if not solver:
        return

    # 非負の変数として変数xとyを定義する
    x = solver.NumVar(0, solver.infinity(), 'x')
    y = solver.NumVar(0, solver.infinity(), 'y')

    print('Number of variables =', solver.NumVariables())

    # 制約条件を与える
    ## x + 2y <= 14.
    solver.Add(x + 2 * y <= 14.0)

    ## 3x - y >= 0.
    solver.Add(3 * x - y >= 0.0)

    ## x - y <= 2.
    solver.Add(x - y <= 2.0)

    print('Number of constraints =', solver.NumConstraints())

    # 目的関数と最適化指標(最大化): 3x + 4y.
    solver.Maximize(3 * x + 4 * y)

    # Solve the system.
    status = solver.Solve()

    if status == pywraplp.Solver.OPTIMAL:
        print('Solution:')
        print('Objective value =', solver.Objective().Value())
        print('x =', x.solution_value())
        print('y =', y.solution_value())
    else:
        print('The problem does not have an optimal solution.')

    print('\nAdvanced usage:')
    print('Problem solved in %f milliseconds' % solver.wall_time())
    print('Problem solved in %d iterations' % solver.iterations())


if __name__ == '__main__':
    # pywrapinit.CppBridge.InitLogging('basic_example.py')
    # cpp_flags = pywrapinit.CppFlags()
    # cpp_flags.logtostderr = True
    # cpp_flags.log_prefix = False
    # pywrapinit.CppBridge.SetFlags(cpp_flags)

    LinearProgrammingExample()