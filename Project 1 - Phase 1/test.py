import numpy as np

def f(x):
    return (150 * (x[0] * x[1]) ** 2) + (((0.5 * x[0]) + (2 * x[1]) - 2) ** 2)

def grad_f(x):
    grad = np.zeros(2)
    grad[0] = 600 * x[0] * x[1]**2 + 0.5 * (0.5 * x[0] + 2 * x[1] - 2)
    grad[1] = 300 * x[0]**2 * x[1] + 2 * (0.5 * x[0] + 2 * x[1] - 2)
    return grad

def hessian_f(x):
    hessian = np.zeros((2, 2))
    hessian[0, 0] = 1200 * x[1]**2 + 0.5
    hessian[0, 1] = hessian[1, 0] = 600 * x[0] * x[1] + 2
    hessian[1, 1] = 300 * x[0]**2 + 8
    return hessian

def linear_conjugate_gradient():
    x = np.array([-0.2, 1.2])  # Initial guess for x
    gradient = grad_f(x)
    d = -gradient
    max_iterations = 1000000
    for iterations in range(max_iterations):
        alpha = np.dot(gradient, gradient) / np.dot(d, np.dot(hessian_f(x), d))
        x = x + alpha * d
        prev_gradient = gradient
        gradient = grad_f(x)
        beta = np.dot(gradient, gradient) / np.dot(prev_gradient, prev_gradient)
        d = -gradient + beta * d
        if np.linalg.norm(gradient) < 1e-7:
            break
    print(iterations)
    return x

# Test the optimization
optimal_x = linear_conjugate_gradient()
optimal_value = f(optimal_x)
print("Optimal x:", optimal_x)
print("Optimal value:", optimal_value)
