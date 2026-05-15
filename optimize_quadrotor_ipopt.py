"""
Optimize a planar quadrotor trajectory using cvxpy with IPOPT (nonlinear solver).

This script follows the setup in optim_test.ipynb but uses the full nonlinear
discrete dynamics in quadrotor_dynamics.f_discrete and solves the resulting
nonconvex NLP with IPOPT via cvxpy: prob.solve(nlp=True, solver=cp.IPOPT).

Requirements:
- cvxpy built with IPOPT support (cyipopt installed and cvxpy from a recent
  branch that includes IPOPT interface). If IPOPT is not available, the script
  will raise an informative error.

Usage: run as a script. It will print solver status and final trajectory and
controls.
"""
import numpy as np
import cvxpy as cp
from quadrotor_dynamics import f_discrete, m, g, I, r


def solve_quadrotor(N=300, h=0.01, u_max=10.0):
    # State: x = [q1, v1, q2, v2, q3, w]
    x0 = np.array([0.25, 0.0, 0.5, 0.0, 0.0, 0.0])
    xT = np.array([0.75, 0.0, 1.5, 0.0, 0.0, 0.0])

    # Decision variables
    x = cp.Variable((6, N + 1))
    u1 = cp.Variable(N)
    u2 = cp.Variable(N)

    constraints = []
    constraints += [x[:, 0] == x0]
    constraints += [x[:, N] == xT]

    # Dynamics constraints using the nonlinear discrete dynamics
    for k in range(N):
        xk = x[:, k]
        uk = cp.hstack([u1[k], u2[k]])
        x_next = f_discrete(xk, uk, h)

        constraints += [x[:, k + 1] == x_next]

    # Control bounds
    constraints += [u1 >= 0, u1 <= u_max]
    constraints += [u2 >= 0, u2 <= u_max]

    # Objective: minimize control effort
    objective = cp.Minimize(cp.sum_squares(u1) + cp.sum_squares(u2))

    problem = cp.Problem(objective, constraints)

    print("Solving with IPOPT (nlp=True, solver=cp.IPOPT). This may take a while...")
    # Solve as a nonlinear program
    try:
        problem.solve(nlp=True, solver=cp.IPOPT, verbose=True)
    except Exception as e:
        raise RuntimeError(
            "IPOPT solve failed. Ensure cyipopt and a compatible cvxpy build are installed. "
            f"Original error: {e}"
        )

    print(f"Status: {problem.status}, Objective: {problem.value}")
    return x.value, u1.value, u2.value


if __name__ == '__main__':
    x_traj, u1_traj, u2_traj = solve_quadrotor(N=200, h=0.01, u_max=10.0)
    print("x shape:", x_traj.shape)
    print("u1 shape:", u1_traj.shape)
    print("u2 shape:", u2_traj.shape)
