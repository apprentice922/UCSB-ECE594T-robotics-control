import cvxpy as cp
import numpy as np

# ============================================================
# Constant settings
# ============================================================
N = 20                  # number of time steps
h = 0.1                 # time step
m = 1.0                 # mass
g = 10                # gravity
I = 0.1                # moment of inertia
r = 0.2                 # arm length / torque coefficient
u_max = 10.0            # max thrust per rotor
max_iters = 100         # alternating optimization iterations
tol = 1e-3              # convergence tolerance

delta_x = 2   # trust region size for state
delta_u = 4   # trust region size for control

# State: x = [q1, v1, q2, v2, q3, w]
x0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
xT = np.array([1.0, 0.0, 1.0, 0.0, 0.0, 0.0])

# ============================================================
# Initial guess
# ============================================================
x_bar = np.zeros((6, N + 1))
u1_bar = 0.5 * u_max * np.ones(N)
u2_bar = 0.5 * u_max * np.ones(N)

x_bar[:, 0] = x0
for k in range(N):
    # simple straight-line initial guess for positions
    alpha = (k + 1) / N
    x_bar[0, k + 1] = (1 - alpha) * x0[0] + alpha * xT[0]
    x_bar[2, k + 1] = (1 - alpha) * x0[2] + alpha * xT[2]
    x_bar[4, k + 1] = (1 - alpha) * x0[4] + alpha * xT[4]
    x_bar[1, k + 1] = (x_bar[0, k + 1] - x_bar[0, k]) / h
    x_bar[3, k + 1] = (x_bar[2, k + 1] - x_bar[2, k]) / h
    x_bar[5, k + 1] = (x_bar[4, k + 1] - x_bar[4, k]) / h

# ============================================================
# Alternating Optimization Loop
# ============================================================
for it in range(max_iters):
    # ========================================================
    # U-STEP: Optimize control u with state x_bar fixed
    # (Dynamics constraints become linear in u)
    # ========================================================
    u1 = cp.Variable(N)
    u2 = cp.Variable(N)

    u_constraints = []

    # Dynamics constraints with x_bar fixed (linear in u)
    for k in range(N):
        v1 = x_bar[1, k]
        v2 = x_bar[3, k]
        q3 = x_bar[4, k]
        w = x_bar[5, k]

        v1_next = x_bar[1, k + 1]
        v2_next = x_bar[3, k + 1]
        q3_next = x_bar[4, k + 1]
        w_next = x_bar[5, k + 1]

        sinq3 = np.sin(q3)
        cosq3 = np.cos(q3)

        # Linearized dynamics (with x fixed, sin/cos are constants):
        # v1_next = v1 + h * (1/m) * sin(q3) * (u1 + u2)
        # v2_next = v2 + h * ((1/m) * cos(q3) * (u1 + u2) - g)
        # q3_next = q3 + h * w  (always satisfied)
        # w_next = w + h * (r/I) * (u2 - u1)

        u_constraints += [
            v1_next == v1 + (h / m) * sinq3 * (u1[k] + u2[k]),
            v2_next == v2 + (h / m) * cosq3 * (u1[k] + u2[k]) - h * g,
            w_next == w + (h * r / I) * (u2[k] - u1[k]),
        ]

    # Control bounds
    u_constraints += [u1 >= 0, u1 <= u_max]
    u_constraints += [u2 >= 0, u2 <= u_max]

    # Trust region
    u_constraints += [
        u1 >= u1_bar - delta_u,
        u1 <= u1_bar + delta_u,
        u2 >= u2_bar - delta_u,
        u2 <= u2_bar + delta_u,
    ]

    # Objective: minimize control effort
    u_objective = cp.Minimize(cp.sum_squares(u1) + cp.sum_squares(u2))

    u_problem = cp.Problem(u_objective, u_constraints)
    u_problem.solve(solver=cp.OSQP, verbose=False)

    if u_problem.status not in ["optimal", "optimal_inaccurate"]:
        print(f"U-step iteration {it+1}: solver failed (status={u_problem.status})")
        u1_new = u1_bar.copy()
        u2_new = u2_bar.copy()
    else:
        u1_new = u1.value.copy()
        u2_new = u2.value.copy()

    # ========================================================
    # X-STEP: Optimize state x with control u fixed to u_new
    # ========================================================
    x = cp.Variable((6, N + 1))

    x_constraints = []

    # Boundary conditions
    x_constraints += [x[:, 0] == x0]
    x_constraints += [x[:, N] == xT]

    # Linearized dynamics constraints with u_bar = u_new fixed
    for k in range(N):
        # Nominal state (current estimate)
        q1b = x_bar[0, k]
        v1b = x_bar[1, k]
        q2b = x_bar[2, k]
        v2b = x_bar[3, k]
        q3b = x_bar[4, k]
        wb = x_bar[5, k]

        # Nominal controls (fixed to u_new)
        u1b = u1_new[k]
        u2b = u2_new[k]

        ub_sum = u1b + u2b
        db = u2b - u1b

        # Current variables
        q1 = x[0, k]
        v1 = x[1, k]
        q2 = x[2, k]
        v2 = x[3, k]
        q3 = x[4, k]
        w = x[5, k]

        dq3 = q3 - q3b

        # Linearize nonlinear terms around (q3b, u_sum_b)
        sinb = np.sin(q3b)
        cosb = np.cos(q3b)

        # sin(q3) * (u1 + u2) ≈ sin(q3b) * ub_sum + cos(q3b) * ub_sum * (q3 - q3b)
        sin_term = sinb * ub_sum + cosb * ub_sum * dq3

        # cos(q3) * (u1 + u2) ≈ cos(q3b) * ub_sum - sin(q3b) * ub_sum * (q3 - q3b)
        cos_term = cosb * ub_sum - sinb * ub_sum * dq3

        # Explicit Euler discretization
        q1_next = q1 + h * v1
        v1_next = v1 + h * (1.0 / m) * sin_term

        q2_next = q2 + h * v2
        v2_next = v2 + h * ((1.0 / m) * (cos_term - m * g))

        q3_next = q3 + h * w
        w_next = w + h * (r / I) * db

        x_constraints += [
            x[0, k + 1] == q1_next,
            x[1, k + 1] == v1_next,
            x[2, k + 1] == q2_next,
            x[3, k + 1] == v2_next,
            x[4, k + 1] == q3_next,
            x[5, k + 1] == w_next,
        ]

        # Trust region
        x_constraints += [
            x[:, k] >= x_bar[:, k] - delta_x,
            x[:, k] <= x_bar[:, k] + delta_x,
        ]

    x_constraints += [
        x[:, N] >= x_bar[:, N] - delta_x,
        x[:, N] <= x_bar[:, N] + delta_x,
    ]

    # Objective: minimize control effort (constant w.r.t. x, but include for completeness)
    x_objective = cp.Minimize(cp.sum_squares(u1_new) + cp.sum_squares(u2_new))

    x_problem = cp.Problem(x_objective, x_constraints)
    x_problem.solve(solver=cp.OSQP, verbose=False)

    if x_problem.status not in ["optimal", "optimal_inaccurate"]:
        print(f"X-step iteration {it+1}: solver failed (status={x_problem.status})")
        x_new = x_bar.copy()
    else:
        x_new = x.value.copy()

    # ========================================================
    # Convergence check and update
    # ========================================================
    dx = np.max(np.abs(x_new - x_bar))
    du1 = np.max(np.abs(u1_new - u1_bar))
    du2 = np.max(np.abs(u2_new - u2_bar))
    du = max(du1, du2)

    cost = np.sum(u1_new**2) + np.sum(u2_new**2)
    print(f"Iteration {it+1}: cost = {cost:.6f}, dx = {dx:.6e}, du = {du:.6e}")

    x_bar = x_new.copy()
    u1_bar = u1_new.copy()
    u2_bar = u2_new.copy()

    if max(dx, du) < tol:
        print("Converged.")
        break

# ============================================================
# Results
# ============================================================
print("\nFinal solution:")
print("x shape:", x_bar.shape)
print("u1 shape:", u1_bar.shape)
print("u2 shape:", u2_bar.shape)
print("\nFinal cost:", np.sum(u1_bar**2) + np.sum(u2_bar**2))
print("\nFirst few steps of u1:", u1_bar[:5])
print("First few steps of u2:", u2_bar[:5])
