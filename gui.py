import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from pid import Controller, State
from plotter import Plotter

class SimulationGUI:
    def __init__(self, root):
        self.simulation_running = False
        self.root = root
        self.root.title("PID Controller for DDR")

        self.default_params = {
            "kP_ang": 2.0, "kI_ang": 0.01, "kD_ang": 0.05,
            "kP_dist": 0.3, "kI_dist": 0.02, "kD_dist": 0.05,
            "dT": 0.1, "max_v": 1.0, "max_w": 1.5,
            "arrive_distance": 0.5, "R": 0.0325, "L": 0.1
        }
        self.params = self.default_params.copy()
        self.entries = {}

        self.setup_gui()

    def setup_gui(self):
        ctrl_frame = ttk.Frame(self.root)
        ctrl_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        ttk.Label(ctrl_frame, text="PID Parameters").pack()

        for key in self.params:
            row = ttk.Frame(ctrl_frame)
            row.pack(fill='x', pady=2)
            ttk.Label(row, text=key, width=15).pack(side=tk.LEFT)
            entry = ttk.Entry(row, width=10)
            entry.insert(0, str(self.params[key]))
            entry.pack(side=tk.LEFT)
            self.entries[key] = entry

        ttk.Button(ctrl_frame, text="Run Simulation", command=self.run_simulation).pack(pady=10)
        ttk.Button(ctrl_frame, text="Reset Parameters", command=self.reset_parameters).pack()

        self.param_display = tk.Text(ctrl_frame, height=20, width=25, state='disabled', bg='#f0f0f0')
        self.param_display.pack(pady=10)

        self.fig, self.ax = plt.subplots(figsize=(5, 5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    def reset_parameters(self):
        self.params = self.default_params.copy()
        for key in self.params:
            self.entries[key].delete(0, tk.END)
            self.entries[key].insert(0, str(self.params[key]))
        self.update_param_display()

    def update_param_display(self):
        self.param_display.config(state='normal')
        self.param_display.delete('1.0', tk.END)
        for key, val in self.params.items():
            self.param_display.insert(tk.END, f"{key}: {val:.4f}\n")
        self.param_display.config(state='disabled')

    def run_simulation(self):
        for key in self.params:
            try:
                self.params[key] = float(self.entries[key].get())
            except ValueError:
                self.params[key] = self.default_params[key]
                self.entries[key].delete(0, tk.END)
                self.entries[key].insert(0, str(self.default_params[key]))

        self.update_param_display()

        if self.simulation_running:
            self.simulation_running = False
            self.root.after(100, self._start_simulation)
            return

        self.ax.clear()
        self._start_simulation()

    def _start_simulation(self):
        self.simulation_running = True

        start = State(-20.0, 15.0, np.radians(90))
        waypoints = [
            State(0, 20, 0), State(20, 10, 0), State(0, 5, 0),
            State(-10, -15, 0), State(0, -10, 0), State(8, -10, 0)
        ]
        plotter = Plotter(fig=self.fig, ax=self.ax)

        def create_controller(start_state, goal_state):
            return Controller(
                start_state, goal_state,
                R_=self.params["R"], L_=self.params["L"],
                kP_ang=self.params["kP_ang"], kI_ang=self.params["kI_ang"], kD_ang=self.params["kD_ang"],
                kP_dist=self.params["kP_dist"], kI_dist=self.params["kI_dist"], kD_dist=self.params["kD_dist"],
                dT=self.params["dT"],
                max_v=self.params["max_v"], max_w=self.params["max_w"],
                arrive_distance=self.params["arrive_distance"],
                simulation_flag=lambda: self.simulation_running
            )

        current_state = start
        full_x, full_y, full_theta = [start.x], [start.y], [start.theta]

        for goal in waypoints:
            if not self.simulation_running:
                break

            controller = create_controller(current_state, goal)
            traj_x, traj_y, traj_theta = controller.runPID(
                plotter, waypoints, (full_x, full_y, full_theta)
            )
            full_x.extend(traj_x[1:])
            full_y.extend(traj_y[1:])
            full_theta.extend(traj_theta[1:])
            current_state = State(traj_x[-1], traj_y[-1], traj_theta[-1])

        if self.simulation_running:
            plotter.drawPlot(full_x, full_y, full_theta, waypoints=waypoints, clear=False)
            self.canvas.draw()

        self.simulation_running = False

