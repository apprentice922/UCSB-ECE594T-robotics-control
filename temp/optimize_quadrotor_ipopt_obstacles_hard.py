"""
Optimize a planar quadrotor trajectory with hard gate constraints using cvxpy.

Gates are modeled as vertical obstacles at x = obs["x"] with a gap between
[bh, height - uh]. The quadrotor is treated as a disk with radius r, so the
allowed gap becomes [bh + r, height - uh - r] when the quadrotor is inside the
gate slab |x - x_i| <= r. Mixed-integer constraints enforce the piecewise
workspace logic.
"""
import numpy as np
import cvxpy as cp

from quadrotor_dynamics import f_discrete
from plot_utils import plot_trajectory


def _add_gate_constraints(
    constraints,
    x,
    obstacles,
    height,
    quadrotor_radius,
    big_m,
    slab_margin=0.0,
):
    if obstacles is None or len(obstacles) == 0:
        return constraints

    num_obs = len(obstacles)
    num_steps = x.shape[1]
    z_left = cp.Variable((num_obs, num_steps), boolean=True)
    z_inside = cp.Variable((num_obs, num_steps), boolean=True)
    z_right = cp.Variable((num_obs, num_steps), boolean=True)

    slab_half_width = quadrotor_radius + slab_margin

    for i, obs in enumerate(obstacles):
        x_i = obs["x"]
        uh = obs["uh"]
        bh = obs["bh"]

        gap_low = bh + quadrotor_radius
        gap_high = height - uh - quadrotor_radius
        if gap_low >= gap_high:
            raise ValueError(
                "Gate gap is too small for the quadrotor radius. "
                f"Computed gap [{gap_low}, {gap_high}]."
            )

        for k in range(num_steps):
            q1 = x[0, k]
            q2 = x[2, k]

            constraints += [z_left[i, k] + z_inside[i, k] + z_right[i, k] == 1]

            constraints += [
                q1 <= x_i - slab_half_width + big_m * (1 - z_left[i, k])
            ]
            constraints += [
                q1 >= x_i - slab_half_width - big_m * (1 - z_inside[i, k])
            ]
            constraints += [
                q1 <= x_i + slab_half_width + big_m * (1 - z_inside[i, k])
            ]
            constraints += [
                q1 >= x_i + slab_half_width - big_m * (1 - z_right[i, k])
            ]

            constraints += [q2 >= gap_low - big_m * (1 - z_inside[i, k])]
            constraints += [q2 <= gap_high + big_m * (1 - z_inside[i, k])]

    return constraints


def solve_quadrotor_with_obstacles_hard(
    N=200,
    h=0.01,
    u_max=10.0,
    height=2.0,
    width=2.0,
    obstacles=None,
    quadrotor_radius=0.5,
    big_m=None,
    slab_margin=0.0,
    solver="ECOS_BB",
    q3_max=0.8,
    w_max=6.0,
):
    """
    Solve trajectory optimization with hard gate avoidance constraints.
    """
    if obstacles is None:
        obstacles = [
            {"x": 0.7, "uh": 0.5, "bh": 0.5},
            {"x": 1.2, "uh": 0.4, "bh": 0.6},
        ]

    if big_m is None:
        big_m = max(width, height) + 2.0 * quadrotor_radius

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

    constraints = _add_gate_constraints(
        constraints,
        x,
        obstacles,
        height,
        quadrotor_radius,
        big_m,
        slab_margin=slab_margin,
    )

    # Objective: minimize control effort
    effort = cp.sum_squares(u1) + cp.sum_squares(u2)
    objective = cp.Minimize(effort)

    problem = cp.Problem(objective, constraints)

    print(
        "Solving mixed-integer problem with cvxpy. "
        f"solver={solver}, big_m={big_m}"
    )
    try:
        problem.solve(solver=solver, verbose=True)
    except Exception as e:
        raise RuntimeError(
            "MI solve failed. Ensure a mixed-integer solver is installed "
            f"and supported by cvxpy. Original error: {e}"
        )

    print(f"Status: {problem.status}, Objective: {problem.value}")
    return x.value, u1.value, u2.value, obstacles


if __name__ == "__main__":
    x_traj, u1_traj, u2_traj, obstacles = solve_quadrotor_with_obstacles_hard()

    trajectory = np.vstack([x_traj[0, :], x_traj[2, :], x_traj[4, :]]).T
    thrusts = np.vstack([u1_traj, u2_traj]).T
    setup = {"m": 1.0, "g": 10.0, "r": 0.2, "I": 0.1, "T": 2.0}
    plot_trajectory(
        height=2.0,
        width=2.0,
        setup_dict=setup,
        obstacles=obstacles,
        trajectory=trajectory,
        thrusts=thrusts,
    )
