import numpy as np

def conjugate_gradient_debug(A, b, x0, tol=1e-8, max_iter=2):
    x = x0
    r = b - A.dot(x)
    p = r
    rs_old = np.dot(r, r)
    print(f"first rs_old = {rs_old}")
    for i in range(max_iter):
        Ap = A.dot(p)
        alpha = rs_old / np.dot(p, Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = np.dot(r, r)

        print(f"Iteration {i + 1}")
        print(f"x = {x}")
        print(f"r = {r}")
        print(f"p = {p}")
        print(f"alpha = {alpha}")
        print(f"rs_old = {rs_old}")
        print(f"rs_new = {rs_new}")

        if np.sqrt(rs_new) < tol:
            print("Converged")
            break
        beta = rs_new / rs_old
        p = r + (rs_new / rs_old) * p
        print(f"updated p = {p}")
        print(f"beta = {beta}")
        print("-----------")


        rs_old = rs_new
    
    return x

# Toy 3x3 problem
A = np.array([[4, 1, 1],
              [1, 3, -1],
              [1, -1, 2]], dtype=float)

b = np.array([1, 2, 3], dtype=float)
x0 = b

# Solve using the Conjugate Gradient method with debug printing
x = conjugate_gradient_debug(A, b, x0)

print("Solution x =", x)
