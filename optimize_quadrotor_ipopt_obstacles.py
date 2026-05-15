"""
Optimize a planar quadrotor trajectory with gate obstacles using IPOPT via cvxpy.

Adds soft penalty constraints for gate passage. Gates are modeled as vertical
obstacles at x = obs['x'] with a gap between [bh, height - uh]. The penalty is
applied when the quadrotor is near the gate in x using a smooth window.
"""
import numpy as np
import cvxpy as cp

from quadrotor_dynamics import f_discrete
from plot_utils import plot_trajectory


def _gate_penalty(q1, q2, obstacles, height, gate_sigma, margin, weight):
    """
    Soft penalty for violating gate gaps.

    For each obstacle i with gap [bh, height - uh], define violations:
        v_low  = max(0, (bh + margin) - q2)
        v_high = max(0, q2 - (height - uh - margin))

    A smooth window w_i(q1) = exp(-((q1 - x_i)/gate_sigma)^2) weights penalties
    near the gate x_i so the cost activates only around the obstacle.
    """
    penalties = []
    for obs in obstacles:
        x_i = obs["x"]
        uh = obs["uh"]
        bh = obs["bh"]

        w_i = cp.exp(-cp.square((q1 - x_i) / gate_sigma))
        v_low = cp.pos((bh + margin) - q2)
        v_high = cp.pos(q2 - (height - uh - margin))
        penalties.append(weight * w_i * (v_low + v_high))

    if len(penalties) == 0:
        return 0.0
    return cp.sum(cp.hstack(penalties))


def solve_quadrotor_with_obstacles(
    N=200,
    h=0.01,
    u_max=10.0,
    height=2.0,
    width=2.0,
    obstacles=None,
    gate_sigma=0.05,
    margin=0.02,
    obs_weight=200.0,
    q3_max=0.8,
    w_max=6.0,
):
    """
    Solve trajectory optimization with soft obstacle penalties.
    """
    if obstacles is None:
        obstacles = [
            {"x": 0.7, "uh": 0.5, "bh": 0.5},
            {"x": 1.2, "uh": 0.4, "bh": 0.6},
        ]

    # State: x = [q1, v1, q2, v2, q3, w]
    x0 = np.array([0.25, 0.0, 0.5, 0.0, 0.0, 0.0])
    xT = np.array([1.75, 0.0, 1.5, 0.0, 0.0, 0.0])

    # Decision variables
    x = cp.Variable((6, N + 1))
    u1 = cp.Variable(N)
    u2 = cp.Variable(N)

    constraints = []
    constraints += [x[:, 0] == x0]
    constraints += [x[:, N] == xT]

    # Dynamics constraints
    for k in range(N):
        xk = x[:, k]
        uk = cp.hstack([u1[k], u2[k]])
        x_next = f_discrete(xk, uk, h)
        constraints += [x[:, k + 1] == x_next]

    # Control bounds
    constraints += [u1 >= 0, u1 <= u_max]
    constraints += [u2 >= 0, u2 <= u_max]

    # Workspace bounds
    constraints += [x[0, :] >= 0, x[0, :] <= width]
    constraints += [x[2, :] >= 0, x[2, :] <= height]

    # Attitude bounds (optional but recommended)
    constraints += [cp.abs(x[4, :]) <= q3_max]
    constraints += [cp.abs(x[5, :]) <= w_max]

    # Objective: minimize control effort + soft obstacle penalties
    effort = cp.sum_squares(u1) + cp.sum_squares(u2)
    obstacle_costs = []
    for k in range(N + 1):
        q1 = x[0, k]
        q2 = x[2, k]
        obstacle_costs.append(
            _gate_penalty(q1, q2, obstacles, height, gate_sigma, margin, obs_weight)
        )
    obstacle_cost = cp.sum(cp.hstack(obstacle_costs))
    objective = cp.Minimize(effort + obstacle_cost)

    problem = cp.Problem(objective, constraints)

    print("Solving with IPOPT (nlp=True, solver=cp.IPOPT). This may take a while...")
    try:
        problem.solve(nlp=True, solver=cp.IPOPT, verbose=True)
    except Exception as e:
        raise RuntimeError(
            "IPOPT solve failed. Ensure cyipopt and a compatible cvxpy build are installed. "
            f"Original error: {e}"
        )

    print(f"Status: {problem.status}, Objective: {problem.value}")
    return x.value, u1.value, u2.value, obstacles


if __name__ == "__main__":
    x_traj, u1_traj, u2_traj, obstacles = solve_quadrotor_with_obstacles()

    trajectory = np.vstack([x_traj[0, :], x_traj[2, :], x_traj[4, :]]).T
    thrusts = np.vstack([u1_traj, u2_traj]).T
    setup = {"m": 1.0, "g": 10.0, "r": 0.2, "I": 0.1, "T": 2.0}
    plot_trajectory(height=2.0, width=2.0, setup_dict=setup,
                    obstacles=obstacles, trajectory=trajectory, thrusts=thrusts)
