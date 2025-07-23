import numpy as np

class State:
    def __init__(self, x_, y_, theta_):
        self.x = x_ if x_ is not None else 0
        self.y = y_ if y_ is not None else 0
        self.theta = theta_ if theta_ is not None else 0

    def __str__(self):
        return f"{self.x},{self.y},{self.theta}"


class Controller:
    def __init__(
        self, start_, goal_, R_=0.0325, L_=0.1,
        kP_ang=2.0, kI_ang=0.01, kD_ang=0.05,
        kP_dist=0.3, kI_dist=0.02, kD_dist=0.05,
        dT=0.1, max_v=1, max_w=1.5,
        arrive_distance=0.5, simulation_flag=None):

        self.current = start_
        self.goal = goal_

        self.R = R_
        self.L = L_

        # PID gains
        self.Kp_ang, self.Ki_ang, self.Kd_ang = kP_ang, kI_ang, kD_ang
        self.Kp_dist, self.Ki_dist, self.Kd_dist = kP_dist, kI_dist, kD_dist

        self.E_ang = self.old_e_ang = 0
        self.E_dist = self.old_e_dist = 0

        self.dt = dT
        self.max_v = max_v
        self.max_w = max_w
        self.arrive_distance = arrive_distance
        self.simulation_flag = simulation_flag or (lambda: True)

    def fixAngle(self, angle):
        return np.arctan2(np.sin(angle), np.cos(angle))

    def iteratePID(self):
        dx = self.goal.x - self.current.x
        dy = self.goal.y - self.current.y
        distance = np.hypot(dx, dy)
        target_theta = np.arctan2(dy, dx)
        error_theta = self.fixAngle(target_theta - self.current.theta)

        # Linear PID control
        e_dist = distance
        self.E_dist += e_dist * self.dt
        d_e_dist = (e_dist - self.old_e_dist) / self.dt
        v = self.Kp_dist * e_dist + self.Ki_dist * self.E_dist + self.Kd_dist * d_e_dist
        v = np.clip(v, 0.0, self.max_v)
        self.old_e_dist = e_dist

        # Angular PID control
        e_ang = error_theta
        self.E_ang += e_ang * self.dt
        d_e_ang = (e_ang - self.old_e_ang) / self.dt
        w = self.Kp_ang * e_ang + self.Ki_ang * self.E_ang + self.Kd_ang * d_e_ang
        w = np.clip(w, -self.max_w, self.max_w)
        self.old_e_ang = e_ang

        return v, w

    def makeAction(self, v, w):
        self.current.x += v * np.cos(self.current.theta) * self.dt
        self.current.y += v * np.sin(self.current.theta) * self.dt
        self.current.theta = self.fixAngle(self.current.theta + w * self.dt)

    def isArrived(self):
        dx = self.goal.x - self.current.x
        dy = self.goal.y - self.current.y
        return (dx**2 + dy**2) < self.arrive_distance**2

    def runPID(self, plotter=None, waypoints=None, full_path=None):
        x, y, theta = [self.current.x], [self.current.y], [self.current.theta]

        while not self.isArrived():
            if not self.simulation_flag():
                break

            v, w = self.iteratePID()
            self.makeAction(v, w)
            x.append(self.current.x)
            y.append(self.current.y)
            theta.append(self.current.theta)

            if plotter:
                path = (full_path[0] + x[1:], full_path[1] + y[1:], full_path[2] + theta[1:]) if full_path else (x, y, theta)
                plotter.drawPlot(*path, waypoints=waypoints)
                if hasattr(plotter, 'fig') and plotter.embedded:
                    plotter.fig.canvas.draw_idle()
                    plotter.fig.canvas.flush_events()

        return x, y, theta
