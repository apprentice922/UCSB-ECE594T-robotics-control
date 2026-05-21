"""
Optimize a planar quadrotor trajectory using cvxpy with IPOPT (nonlinear solver).
`
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


def solve_quadrotor(x0 = np.array([0.25, 0.0, 0.5, 0.0, 0.0, 0.0]),
                    xT = np.array([0.75, 0.0, 1.5, 0.0, 0.0, 0.0]),
                    obstacles_height = np.array([.2, .6]),  #upper height and lower height of obstacle
                    slack_traj_num = 10,
                    N=100, h=0.01, u_max=10.0,
                    border_x = np.array([0, 4]),
                    border_y = np.array([0,2])):
    # State: x = [q1, v1, q2, v2, q3, w]
    
    # Decision variables
    x = cp.Variable((6, N + 1))
    u1 = cp.Variable(N)
    u2 = cp.Variable(N)

    constraints = []
    constraints += [x[:, 0] == x0]
    constraints += [x[0, N] == xT[0]]

    # Dynamics constraints using the nonlinear discrete dynamics
    # Choose an obstacle x-coordinate instance (midpoint of the x-bounds)
    # The user asked to "take an instance" - we use the midpoint of border_x.
    x_obs = float((border_x[0] + border_x[1]) / 2.0)

    # Relaxation magnitude for the big-M style relaxation. Keep it similar to
    # the canvas height so the relaxation does not explode numerically.
    M_relax = float(border_y[1] - border_y[0]) + 1.0

    # Gate bounds derived from obstacles_height and canvas top
    y_gate_low = obstacles_height[1]
    y_gate_high = border_y[1] - obstacles_height[0]

    for k in range(N):
        xk = x[:, k]
        uk = cp.hstack([u1[k], u2[k]])
        x_next = f_discrete(xk, uk, h)

        constraints += [x[:, k + 1] == x_next]

        constraints += [x[0, k+1] >= border_x[0]]
        constraints += [x[0, k+1] <= border_x[1]]

        constraints += [x[2, k+1] >= border_y[0]]
        constraints += [x[2, k+1] <= border_y[1]]

        # Hard gate constraint activated when the quadrotor is within r (arm
        # length) in x of the obstacle x_obs. We implement this as a
        # continuous relaxation: s = pos(r - |q1 - x_obs|) gives a weight in
        # [0, r]; alpha = s / r in [0,1]. When alpha==1 (centered on the
        # obstacle) the gate bounds are enforced exactly. When alpha==0 the
        # constraint is relaxed by up to M_relax (the canvas height + 1), so
        # the existing canvas bounds still apply.
        #
        # Note: this uses nonconvex cvxpy atoms (abs, pos) and will be solved
        # as an NLP via IPOPT (nlp=True).
        q1_next = x[0, k+1]
        q2_next = x[2, k+1]
        s = cp.pos(r - cp.abs(q1_next - x_obs))
        alpha = s / r

        constraints += [q2_next >= y_gate_low - (1 - alpha) * M_relax]
        constraints += [q2_next <= y_gate_high + (1 - alpha) * M_relax]

    # Border avoid constraints
    for k in range(N-slack_traj_num, N):
        constraints += [x[2, k+1] >= obstacles_height[1]]
        constraints += [x[2, k+1] <= border_y[1] - obstacles_height[0]]

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
    return x.value, u1.value, u2.value, problem


if __name__ == '__main__':
    cost = 0
    x_traj = np.array([]); u1_traj = np.array([]); u2_traj = np.array([])

    _x_traj, _u1_traj, _u2_traj, problem = solve_quadrotor(N=100, h=0.01, u_max=10.0)
    x_traj = _x_traj; u1_traj = _u1_traj; u2_traj = _u2_traj;
    cost += problem.value

    #_x_traj, _u1_traj, _u2_traj, problem = solve_quadrotor(x0=_x_traj[:, -1], xT=[2], N=200, h=0.01, u_max=10.0, obstacles_height=[1.25, 0])
    #x_traj = np.hstack((x_traj, _x_traj)); u1_traj = np.hstack((u1_traj, _u1_traj)); u2_traj = np.hstack((u2_traj, _u2_traj));
    #cost += problem.value


    print("x shape:", x_traj.shape)
    print("u1 shape:", u1_traj.shape)
    print("u2 shape:", u2_traj.shape)
