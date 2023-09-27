import problems
from optimizer import Optimizer


def run(exact=False, max_iter=1000, print_progress=True):
    if exact:
        print('###################\n# Exact Solutions #\n###################')
    else:
        print('###################\n# Approx Solutions #\n###################')
    for f_name in problems.functions:
        print(f'Running: {f_name.capitalize()}\n===================')
        # Get The Problem
        problem = problems.functions[f_name]

        for x0 in problem['starting_points']:
            print(f'\t- Starting Point : {x0}\n\t{"-"*30}')
            # Get Function [Based on the starting point]
            f = problem['func'](x0)

            # Initiate Optimizer
            optimizer = Optimizer(f, exact=exact, max_iter=max_iter)

            # Run Each Method from Available Methods in the optimizer
            for method_name in ('newton_method', 'cg_linear', 'cg_nonlinear', 'qn'):
                # Newton Method
                if method_name == 'newton_method':
                    # optimizer.newton_method(eigen=False, print_progress=True)
                    optimizer.newton_method(eigen=True, print_progress=print_progress)
                    print(f'\t\t{"_" * 100}\n')
                # Conjugate Gradient [Linear]
                if method_name == 'cg_linear':
                    print('\t\t- Conjugate Gradient [Linear]:')
                    optimizer.cg_linear(print_progress=print_progress)
                # Conjugate Gradient [Non-Linear]
                if method_name == 'cg_nonlinear':
                    print('\t\t- Conjugate Gradient [Non-Linear]:')
                    print(f'\t\t\t- Fletcher-Reevers Method\n\t\t\t{"-" * 12}')
                    optimizer.cg_nonlinear(method='f_r', print_progress=print_progress)
                    print(f'\t\t\t- Polak-Ribiere Method\n\t\t\t{"-" * 12}')
                    optimizer.cg_nonlinear(method='p_r', print_progress=print_progress)
                # Quasi Newton
                if method_name == 'qn':
                    print('\t\t- Quasi Newton:')
                    print(f'\t\t\t- BFGS Method:\n\t\t\t{"-" * 15}')
                    optimizer.qn(method='bfgs', print_progress=print_progress)
                    print(f'\t\t\t- SR1 Method:\n\t\t\t{"-" * 12}')
                    optimizer.qn(method='sr1', print_progress=print_progress)
                    print(f'\t\t\t- SR1 Method + Trust Region:\n\t\t\t{"-" * 20}')
                    optimizer.qn(method='sr1', trust_region=True, print_progress=print_progress)
                else:
                    raise NotImplementedError('Method not supported')


if __name__ == "__main__":
    run(exact=True, max_iter=1000, print_progress=False)
    # run(exact=False, max_iter=1000, print_progress=False)
