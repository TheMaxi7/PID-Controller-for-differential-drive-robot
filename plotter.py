import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage


class Plotter:
    def __init__(self, fig=None, ax=None):
        base_path = os.path.dirname(os.path.abspath(__file__))
        img_path = os.path.join(base_path, "drone.png")
        self.image = plt.imread(img_path)

        if fig is not None and ax is not None:
            self.fig = fig
            self.ax = ax
            self.embedded = True  # GUI mode
        else:
            self.fig, self.ax = plt.subplots(figsize=(10, 10))
            self.embedded = False  # Standalone mode

    def drawPlot(self, x, y, theta, waypoints=None, show=False, clear=True):
        if clear:
            self.ax.cla()  # clear previous drawings

        # Draw trajectory
        self.ax.plot(x, y, 'b-', label='Trajectory')  # path line
        self.ax.plot(x, y, 'ob')                      # waypoints

        # Draw robot image
        robot_xlim = 1.5
        robot_ylim = 1.5
        angle_deg = np.degrees(theta[-1])
        rotated_img = ndimage.rotate(self.image, -90 + angle_deg) * 255

        self.ax.imshow(
            rotated_img.astype(np.uint8),
            extent=[
                x[-1] - robot_xlim, x[-1] + robot_xlim,
                y[-1] - robot_ylim, y[-1] + robot_ylim
            ],
            zorder=5
        )

        # Draw waypoints
        if waypoints is not None:
            for i, wp in enumerate(waypoints):
                self.ax.plot(wp.x, wp.y, 'r*', markersize=18 if i == len(waypoints) - 1 else 10)
                label = "END" if i == len(waypoints) - 1 else str(i + 1)
                self.ax.text(wp.x, wp.y + 1.0, label, color='red',
                             fontsize=12 if label == "END" else 10,
                             ha='center', fontweight='bold' if label == "END" else 'normal')

        # Configure plot appearance
        self.ax.grid(True)
        self.ax.set_xlim(-25, 25)
        self.ax.set_ylim(-25, 25)
        self.ax.set_aspect('equal', adjustable='box')

        # Render the figure
        if self.embedded:
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
        elif show:
            plt.show()
