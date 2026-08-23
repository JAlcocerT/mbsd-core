"""Slider-crank kinematic analysis with validation guardrails.

This example uses only the public MBSD 0.1 planar API. It is intentionally
written as an agent-friendly analysis: parameters are explicit, the model reads
as mechanism declarations, and the output includes residual checks before
interpreting the motion.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from mbsd import Mechanism


def build_slider_crank() -> tuple[object, np.ndarray, dict[str, float]]:
    crank_radius = 0.35
    rod_length = 1.15
    angular_speed = 2.0 * np.pi

    mechanism = Mechanism.planar(gravity=(0.0, 0.0))
    ground = mechanism.ground()
    crank = mechanism.body("crank", mass=0.5, inertia=0.01)
    rod = mechanism.body("connecting_rod", mass=1.0, inertia=0.03)
    slider = mechanism.body("slider", mass=1.5, inertia=0.02)

    mechanism.pin(ground, crank, point_a=(0.0, 0.0), point_b=(0.0, 0.0))
    mechanism.pin(crank, rod, point_a=(crank_radius, 0.0), point_b=(0.0, 0.0))
    mechanism.pin(rod, slider, point_a=(rod_length, 0.0), point_b=(0.0, 0.0))
    mechanism.slider(ground, slider, axis=(1.0, 0.0))
    mechanism.motor(crank, omega=angular_speed)

    q0 = np.zeros(mechanism.ncoord)
    q0[6:9] = [crank_radius, 0.0, 0.0]
    q0[9:12] = [crank_radius + rod_length, 0.0, 0.0]

    params = {
        "crank_radius": crank_radius,
        "rod_length": rod_length,
        "angular_speed": angular_speed,
    }
    return mechanism, q0, params


def run_analysis(nsteps: int = 361) -> dict[str, float]:
    mechanism, q0, params = build_slider_crank()
    t = np.linspace(0.0, 1.0, nsteps)
    result = mechanism.solve_kinematics(t, q0=q0)

    slider_x = result.q[9, :]
    slider_y = result.q[10, :]
    slider_theta = result.q[11, :]
    crank_theta = result.q[5, :]

    velocity_fd = np.gradient(slider_x, t, edge_order=2)
    acceleration_fd = np.gradient(result.v[9, :], t, edge_order=2)
    interior = slice(2, -2)

    metrics = {
        "coordinates": float(mechanism.ncoord),
        "constraints": float(mechanism.nrestr),
        "crank_radius": params["crank_radius"],
        "rod_length": params["rod_length"],
        "angular_speed": params["angular_speed"],
        "max_constraint_residual": mechanism.max_constraint_residual(result),
        "max_drive_error": float(np.max(np.abs(crank_theta - params["angular_speed"] * t))),
        "max_slider_y_error": float(np.max(np.abs(slider_y))),
        "max_slider_theta_error": float(np.max(np.abs(slider_theta))),
        "slider_x_min": float(np.min(slider_x)),
        "slider_x_max": float(np.max(slider_x)),
        "slider_stroke": float(np.ptp(slider_x)),
        "max_slider_speed": float(np.max(np.abs(result.v[9, :]))),
        "velocity_fd_error": float(np.max(np.abs(velocity_fd[interior] - result.v[9, interior]))),
        "acceleration_fd_error": float(
            np.max(np.abs(acceleration_fd[interior] - result.a[9, interior]))
        ),
    }
    return metrics


def print_report(metrics: dict[str, float]) -> None:
    print("Slider-crank analysis")
    print("  model:")
    print(f"    coordinates / constraints: {metrics['coordinates']:.0f} / {metrics['constraints']:.0f}")
    print(f"    crank radius:              {metrics['crank_radius']:.3f} m")
    print(f"    rod length:                {metrics['rod_length']:.3f} m")
    print(f"    angular speed:             {metrics['angular_speed']:.6f} rad/s")
    print("  guardrails:")
    print(f"    max constraint residual:   {metrics['max_constraint_residual']:.3e}")
    print(f"    max drive error:           {metrics['max_drive_error']:.3e}")
    print(f"    max slider |y|:            {metrics['max_slider_y_error']:.3e}")
    print(f"    max slider |theta|:        {metrics['max_slider_theta_error']:.3e}")
    print(f"    velocity FD error:         {metrics['velocity_fd_error']:.3e}")
    print(f"    acceleration FD error:     {metrics['acceleration_fd_error']:.3e}")
    print("  motion:")
    print(f"    slider x min/max:          {metrics['slider_x_min']:.6f} / {metrics['slider_x_max']:.6f} m")
    print(f"    slider stroke:             {metrics['slider_stroke']:.6f} m")
    print(f"    max slider speed:          {metrics['max_slider_speed']:.6f} m/s")


def save_plot(path: Path, nsteps: int) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("Plotting requires: pip install 'mbsd[plot]'") from exc

    mechanism, q0, _params = build_slider_crank()
    t = np.linspace(0.0, 1.0, nsteps)
    result = mechanism.solve_kinematics(t, q0=q0)

    fig, axes = plt.subplots(3, 1, figsize=(8.0, 7.0), sharex=True)
    fig.suptitle("MBSD slider-crank kinematics")
    axes[0].plot(t, result.q[9, :], color="#2563eb")
    axes[0].set_ylabel("slider x [m]")
    axes[1].plot(t, result.v[9, :], color="#dc2626")
    axes[1].set_ylabel("slider vx [m/s]")
    axes[2].plot(t, result.a[9, :], color="#16a34a")
    axes[2].set_ylabel("slider ax [m/s^2]")
    axes[2].set_xlabel("time [s]")
    for ax in axes:
        ax.grid(True, alpha=0.35)
    fig.tight_layout()

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nsteps", type=int, default=361)
    parser.add_argument("--plot", type=Path, default=None)
    args = parser.parse_args()

    metrics = run_analysis(nsteps=args.nsteps)
    print_report(metrics)
    if args.plot is not None:
        save_plot(args.plot, nsteps=args.nsteps)
        print(f"  plot: {args.plot}")


if __name__ == "__main__":
    main()
