import math

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


TARGET_START = (0.0, 0.0)
TARGET_VELOCITY = (0.045, 0.018)
INTERCEPTOR_START = (-4.0, -3.0)
INTERCEPTOR_SPEED = 0.09
INTERCEPT_DISTANCE = 0.2
MAX_STEPS = 400


def main() -> None:
    target = [TARGET_START[0], TARGET_START[1]]
    interceptor = [INTERCEPTOR_START[0], INTERCEPTOR_START[1]]
    trail_target_x = [target[0]]
    trail_target_y = [target[1]]
    trail_interceptor_x = [interceptor[0]]
    trail_interceptor_y = [interceptor[1]]

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(-5, 8)
    ax.set_ylim(-5, 5)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Drone Interception Simulation")
    ax.grid(True, linestyle="--", alpha=0.35)

    target_dot, = ax.plot([], [], "bo", markersize=10, label="Target")
    interceptor_dot, = ax.plot([], [], "ro", markersize=10, label="Interceptor")
    target_path, = ax.plot([], [], "b-", linewidth=1, alpha=0.5)
    interceptor_path, = ax.plot([], [], "r-", linewidth=1, alpha=0.5)
    status_text = ax.text(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )
    ax.legend(loc="lower right")

    intercepted = {"value": False}

    def update(_frame: int):
        if intercepted["value"]:
            return target_dot, interceptor_dot, target_path, interceptor_path, status_text

        target[0] += TARGET_VELOCITY[0]
        target[1] += TARGET_VELOCITY[1]

        dx = target[0] - interceptor[0]
        dy = target[1] - interceptor[1]
        distance = math.hypot(dx, dy)

        if distance > 0:
            step = min(INTERCEPTOR_SPEED, distance)
            interceptor[0] += step * dx / distance
            interceptor[1] += step * dy / distance

        trail_target_x.append(target[0])
        trail_target_y.append(target[1])
        trail_interceptor_x.append(interceptor[0])
        trail_interceptor_y.append(interceptor[1])

        target_dot.set_data([target[0]], [target[1]])
        interceptor_dot.set_data([interceptor[0]], [interceptor[1]])
        target_path.set_data(trail_target_x, trail_target_y)
        interceptor_path.set_data(trail_interceptor_x, trail_interceptor_y)

        distance = math.hypot(target[0] - interceptor[0], target[1] - interceptor[1])
        status_text.set_text(
            f"Target: ({target[0]:.2f}, {target[1]:.2f})\n"
            f"Interceptor: ({interceptor[0]:.2f}, {interceptor[1]:.2f})\n"
            f"Distance: {distance:.2f}"
        )

        if distance < INTERCEPT_DISTANCE:
            intercepted["value"] = True
            status_text.set_text(
                status_text.get_text() + "\nIntercepted!"
            )
            print("Intercepted!", flush=True)
            plt.close(fig)

        return target_dot, interceptor_dot, target_path, interceptor_path, status_text

    anim = FuncAnimation(fig, update, frames=MAX_STEPS, interval=40, blit=False, repeat=False)
    plt.show()
    return anim


if __name__ == "__main__":
    main()
