import math
import subprocess
import sys


def ensure_matplotlib():
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        print("matplotlib not found. Installing with pip...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib"])


ensure_matplotlib()

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class Drone:
    def __init__(self, x, y, speed):
        self.x = x
        self.y = y
        self.speed = speed

    def position(self):
        return (self.x, self.y)

    def move_towards(self, target_x, target_y):
        dx = target_x - self.x
        dy = target_y - self.y
        distance_to_target = math.hypot(dx, dy)
        if distance_to_target == 0:
            return
        step = min(self.speed, distance_to_target)
        self.x += step * dx / distance_to_target
        self.y += step * dy / distance_to_target

    def move_in_pattern(self, step_count):
        # Preserve the original alternating pattern and make motion continuous.
        if step_count % 2 == 0:
            self.x += self.speed
        else:
            self.y += self.speed


def distance(d1, d2):
    return math.hypot(d1.x - d2.x, d1.y - d2.y)


def main():
    target = Drone(0.0, 0.0, 0.12)
    chaser = Drone(-6.0, -6.0, 0.18)

    fig, ax = plt.subplots()
    fig.canvas.manager.set_window_title("Drone Chase Simulation")
    ax.set_title("Drone Chase Simulation")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-7, 8)
    ax.set_ylim(-7, 8)

    target_dot, = ax.plot([], [], "bo", markersize=8, label="Target Drone")
    chaser_dot, = ax.plot([], [], "ro", markersize=8, label="Chaser Drone")
    status_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top")
    ax.legend(loc="lower right")

    state = {"step_count": 0, "intercepted": False, "animation": None}

    def init():
        target_dot.set_data([target.x], [target.y])
        chaser_dot.set_data([chaser.x], [chaser.y])
        current_distance = distance(target, chaser)
        status_text.set_text(f"Step: 0\nDistance: {current_distance:.3f}")
        return target_dot, chaser_dot, status_text

    def update(_frame):
        if state["intercepted"]:
            return target_dot, chaser_dot, status_text

        step_count = state["step_count"]
        target.move_in_pattern(step_count)
        chaser.move_towards(target.x, target.y)
        current_distance = distance(target, chaser)

        target_dot.set_data([target.x], [target.y])
        chaser_dot.set_data([chaser.x], [chaser.y])
        status_text.set_text(f"Step: {step_count}\nDistance: {current_distance:.3f}")

        print(
            f"Step {step_count}: Target at {target.position()}, "
            f"Chaser at {chaser.position()}, Distance {current_distance:.3f}"
        )

        pad = 1.5
        min_x = min(target.x, chaser.x) - pad
        max_x = max(target.x, chaser.x) + pad
        min_y = min(target.y, chaser.y) - pad
        max_y = max(target.y, chaser.y) + pad
        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)

        if current_distance < 0.2:
            state["intercepted"] = True
            print("Intercepted!")
            status_text.set_text(
                f"Step: {step_count}\nDistance: {current_distance:.3f}\nIntercepted!"
            )
            if state["animation"] is not None:
                state["animation"].event_source.stop()
            close_timer = fig.canvas.new_timer(interval=300)
            close_timer.add_callback(lambda: plt.close(fig))
            close_timer.start()
            fig.canvas.draw_idle()

        state["step_count"] += 1
        return target_dot, chaser_dot, status_text

    animation = FuncAnimation(
        fig,
        update,
        init_func=init,
        interval=50,
        blit=False,
        cache_frame_data=False,
    )
    state["animation"] = animation
    plt.show()


if __name__ == "__main__":
    main()
