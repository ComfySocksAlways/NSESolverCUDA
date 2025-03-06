import numpy as np

def jacobi_iteration(A, b, x, max_iter, tol):
    n = len(A)
    D = np.diag(np.diag(A))
    R = A - D
    x_old = np.copy(x)
    
    for k in range(max_iter):
        print(f"Iteration {k}")
        
        # Compute -(R * x)
        Rx = -np.dot(R, x_old)
        print(f"f: {Rx}")
        
        # Compute b - R * x
        b_minus_Rx = b + Rx
        print(f"g: {b_minus_Rx}")
        
        # Compute D inverse times (b - R * x)
        D_inv = np.linalg.inv(D)
        x = np.dot(D_inv, b_minus_Rx)
        print(f"x: {x}")
        
        # Check for convergence
        norm = np.linalg.norm(x - x_old, ord=2)
        if norm < tol:
            print(f"Converged after {k+1} iterations.")
            break
        
        x_old = np.copy(x)

    return x

# Define the system
A = np.array([
    [ 4, -1,  0,  0,  0],
    [-1,  4, -1,  0,  0],
    [ 0, -1,  4, -1,  0],
    [ 0,  0, -1,  4, -1],
    [ 0,  0,  0, -1,  3]
])

b = np.array([1, 2, 0, 1, 1])
x = np.array([1, 2, 0, 1, 1])  # Initial guess

# Parameters for the iteration
max_iter = 25
tol = 1e-6

# Perform Jacobi iteration
x_final = jacobi_iteration(A, b, x, max_iter, tol)

print(f"Final solution: {x_final}")
