import numpy as np

def rounded_nelder_mead(fun, x0, bounds, print_progress, decimal_precision1=4, decimal_precision2=4, decimal_precision3=4, maxiter=100, alpha=1.3487640718112321, gamma=2.139744408219208, rho=0.3740185211921033, sigma=0.5501390442296267, tol=1e-4):
    n = len(x0)

    simplex = np.zeros((n + 1, n))
    simplex[0] = x0

    for i in range(n):
        point = np.array(x0)
        point[i] = bounds[i][0]
        simplex[i + 1] = point

        point = np.array(x0)
        point[i] = bounds[i][1]
        simplex[i + 1] = point

    fs = np.zeros(n + 1)
    for i in range(n + 1):
        fs[i] = fun(simplex[i])

    nfev = n + 1  # Initialize the number of function evaluations

    optimized_solutions = []  # List to store optimized solutions

    for i in range(maxiter):
        # Sort simplex according to function values
        idx = np.argsort(fs)
        simplex = simplex[idx]
        fs = fs[idx]

        # Calculate the centroid of the simplex
        xbar = np.mean(simplex[:-1], axis=0)

        # Reflection
        xr = xbar + alpha * (xbar - simplex[-1])
        xr = np.clip(xr, bounds[:, 0], bounds[:, 1])
        xr[0] = np.round(xr[0], decimals=decimal_precision1)
        xr[1] = np.round(xr[1], decimals=decimal_precision2)
        xr[2] = np.round(xr[2], decimals=decimal_precision3)
        fxr = fun(xr)
        nfev += 1  # Increment the number of function evaluations

        if fs[0] <= fxr < fs[-2]:
            # Successful reflection
            simplex[-1] = xr
            fs[-1] = fxr

        elif fxr < fs[0]:
            # Expansion
            xe = xbar + gamma * (xr - xbar)
            xe = np.clip(xe, bounds[:, 0], bounds[:, 1])
            xe[0] = np.round(xe[0], decimals=decimal_precision1)
            xe[1] = np.round(xe[1], decimals=decimal_precision2)
            xe[2] = np.round(xe[2], decimals=decimal_precision3)
            fxe = fun(xe)
            nfev += 1  # Increment the number of function evaluations

            if fxe < fxr:
                simplex[-1] = xe
                fs[-1] = fxe
            else:
                simplex[-1] = xr
                fs[-1] = fxr

        else:
            # Contraction
            xc = xbar + rho * (simplex[-1] - xbar)
            xc = np.clip(xc, bounds[:, 0], bounds[:, 1])
            xc[0] = np.round(xc[0], decimals=decimal_precision1)
            xc[1] = np.round(xc[1], decimals=decimal_precision2)
            xc[2] = np.round(xc[2], decimals=decimal_precision3)
            
            fxc = fun(xc)
            nfev += 1  # Increment the number of function evaluations

            if fxc < fs[-1]:
                simplex[-1] = xc
                fs[-1] = fxc

            else:
                # Shrink
                simplex[1:] = simplex[0] + sigma * (simplex[1:] - simplex[0])
                for j in range(1, n + 1):
                    simplex[j][0] = np.round(simplex[j][0], decimals=decimal_precision1)
                    simplex[j][1] = np.round(simplex[j][1], decimals=decimal_precision2)
                    simplex[j][2] = np.round(simplex[j][2], decimals=decimal_precision3)
                    fs[j] = fun(simplex[j])
                    nfev += 1  # Increment the number of function evaluations
        
        if print_progress == 1:
            print('iteration:', i+1)
            print('best value of objective function so far:', fs[0])
            print('best candidate:', simplex[0])
            
        # Round and append optimized solution to the list with individual decimal precision for each parameter
        rounded_solution = (
            np.round(simplex[0][0], decimals=decimal_precision1),
            np.round(simplex[0][1], decimals=decimal_precision2),
            np.round(simplex[0][2], decimals=decimal_precision3)
        )
        optimized_solutions.append(rounded_solution)
                
        # Check termination criteria
        if np.max(np.abs(fs[:-1] - fs[-1])) < tol:
            break

    # Round optimized solution with individual decimal precision for each parameter
    optimized_solution = (
        np.round(simplex[0][0], decimals=decimal_precision1),
        np.round(simplex[0][1], decimals=decimal_precision2),
        np.round(simplex[0][2], decimals=decimal_precision3)
    )
    min_function_value = fs[0]
    success = True if i < maxiter - 1 else False
    termination_reason = 'Maximum number of iterations exceeded.' if i == maxiter - 1 else 'Maximum number of iterations not exceeded'
    num_iterations = i + 1

    return optimized_solution, min_function_value, nfev, success, termination_reason, num_iterations, optimized_solutions
