#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.ticker import MaxNLocator
import os
import glob
import numpy as np
import time
from collections import deque

LOG_DIRECTORY = "/home/varun/ws_offboard_control/firedronex_plots"
MAX_DATA_POINTS_TRAJECTORY = 2000 # Max data points to keep in memory for trajectory/altitude/state plots for performance
MAX_DATA_POINTS_DETECTIONS = 500 # Max fire detection points to keep in memory for performance
PLOT_UPDATE_INTERVAL_MS = 1000  # Update plots every 1 second

class LivePlotterNode(Node):
    def __init__(self):
        super().__init__('firedronex_live_plotter_node')
        self.get_logger().info(f"Log directory set to: {LOG_DIRECTORY}")

        # Figure 1: 2D Trajectory & Fire Map (Separate Window)
        self.fig_traj_map, self.ax_traj_map = plt.subplots(figsize=(10, 8)) # Larger figure size
        self.ax_traj_map.set_xlabel("X (m)")
        self.ax_traj_map.set_ylabel("Y (m)")
        self.ax_traj_map.set_title("2D Trajectory & Fire Map")
        self.ax_traj_map.axis('equal')
        self.drone_path_line, = self.ax_traj_map.plot([], [], 'b-', label="Drone Path")
        self.drone_current_pos_scatter = self.ax_traj_map.scatter([], [], c='blue', marker='o', s=50, label="Drone Current")
        self.fire_detections_scatter = self.ax_traj_map.scatter([], [], c='red', marker='x', s=40, label="Fire Detections")
        self.current_target_scatter = self.ax_traj_map.scatter([], [], c='magenta', marker='*' , s=150, label="Current Target") # Corrected marker syntax
        self.ax_traj_map.legend(loc='upper right', fontsize='small')
        self.ax_traj_map.grid(True)

        # Figure 2: Altitude Profile and Mission State (Combined in one window)
        self.fig_other_plots, self.axs_other = plt.subplots(2, 1, figsize=(10, 8)) # 2 rows, 1 col
        plt.figure(self.fig_other_plots.number) # Ensure subsequent plt commands target this figure
        plt.subplots_adjust(hspace=0.4)

        # Setup for Graph 2: Altitude Profile (now on axs_other[0])
        self.ax_alt = self.axs_other[0]
        self.ax_alt.set_xlabel("Time (s)")
        self.ax_alt.set_ylabel("Altitude (Z, m)")
        self.ax_alt.set_title("Altitude Profile")
        self.altitude_line, = self.ax_alt.plot([], [], 'g-', label="Altitude")
        self.ax_alt.legend(loc='upper right', fontsize='small')
        self.ax_alt.grid(True)

        # Setup for Graph 3: Mission State Over Time (now on axs_other[1])
        self.ax_state = self.axs_other[1]
        self.ax_state.set_xlabel("Time (s)")
        self.ax_state.set_ylabel("Mission State")
        self.ax_state.set_title("Mission State Over Time")
        self.mission_state_line, = self.ax_state.plot([], [], 'r-', drawstyle='steps-post', label="State")
        self.ax_state.legend(loc='upper right', fontsize='small')
        self.ax_state.grid(True)
        
        # Data storage using deque for efficient appends and pops
        self.timestamps = deque(maxlen=MAX_DATA_POINTS_TRAJECTORY)
        self.odom_x_data = deque(maxlen=MAX_DATA_POINTS_TRAJECTORY)
        self.odom_y_data = deque(maxlen=MAX_DATA_POINTS_TRAJECTORY)
        self.odom_z_data = deque(maxlen=MAX_DATA_POINTS_TRAJECTORY)
        self.mission_states_numeric = deque(maxlen=MAX_DATA_POINTS_TRAJECTORY) # Store numeric representation of states
        
        self.fire_x_data = deque(maxlen=MAX_DATA_POINTS_DETECTIONS)
        self.fire_y_data = deque(maxlen=MAX_DATA_POINTS_DETECTIONS)
        
        self.current_target_x = None
        self.current_target_y = None

        self.mission_state_map = {} # To map state strings to numbers for plotting
        self.next_mission_state_idx = 0
        
        self.latest_traj_log_file = None
        self.latest_det_log_file = None
        self.last_traj_read_time = 0
        self.last_det_read_time = 0
        self.start_time_abs = time.time() # For relative time on plots if timestamps are absolute

        # Animation will be driven by the `fig_other_plots` (Altitude/State)
        # The update_plots function will handle updating both figures.
        self.ani = animation.FuncAnimation(self.fig_other_plots, self.update_plots, init_func=self.init_plots, 
                                           interval=PLOT_UPDATE_INTERVAL_MS, blit=False, cache_frame_data=False)
        
        self.get_logger().info("Live plotter initialized. Waiting for log files...")
        plt.show(block=True) # This will show all figures and block until the one with animation is closed.

    def find_latest_log_files(self):
        # Attempt to find files
        traj_log_pattern = os.path.join(LOG_DIRECTORY, "trajectory_log.csv")
        det_log_pattern = os.path.join(LOG_DIRECTORY, "detection_log.csv")

        found_traj_files = glob.glob(traj_log_pattern)
        found_det_files = glob.glob(det_log_pattern)

        if not found_traj_files and self.latest_traj_log_file is None:
            self.get_logger().info("No trajectory log files found yet.", throttle_duration_sec=5)
            return False
        if not found_det_files and self.latest_det_log_file is None :
            self.get_logger().info("No detection log files found yet.", throttle_duration_sec=5)
            return False

        if found_traj_files:
            latest_traj = max(found_traj_files, key=os.path.getctime)
            if self.latest_traj_log_file != latest_traj:
                self.get_logger().info(f"New trajectory log file: {latest_traj}")
                self.latest_traj_log_file = latest_traj
                self.last_traj_read_time = 0 # Reset read time for new file
                 # Reset data deques when a new file is chosen, to avoid mixing old and new session data
                self.timestamps.clear()
                self.odom_x_data.clear()
                self.odom_y_data.clear()
                self.odom_z_data.clear()
                self.mission_states_numeric.clear()
                self.start_time_abs = time.time() # Reset relative start time

        if found_det_files:
            latest_det = max(found_det_files, key=os.path.getctime)
            if self.latest_det_log_file != latest_det:
                self.get_logger().info(f"New detection log file: {latest_det}")
                self.latest_det_log_file = latest_det
                self.last_det_read_time = 0
                self.fire_x_data.clear()
                self.fire_y_data.clear()
        
        return self.latest_traj_log_file is not None or self.latest_det_log_file is not None


    def load_new_data(self):
        new_data_loaded = False
        if not self.find_latest_log_files() and (self.latest_traj_log_file is None and self.latest_det_log_file is None) :
            self.get_logger().debug("load_new_data: find_latest_log_files returned false and no log files cached.", throttle_duration_sec=5)
            return False

        # Helper function for robust float conversion
        def robust_float_conversion(value, column_name, row_idx, logger):
            try:
                if pd.isna(value):
                    logger.warn(f"NaN value found in column '{column_name}' at row {row_idx}", throttle_duration_sec=10)
                    return np.nan
                float_val = float(value)
                if not np.isfinite(float_val):
                    logger.warn(f"Non-finite value {value} found in column '{column_name}' at row {row_idx}", throttle_duration_sec=10)
                    return np.nan
                return float_val
            except (ValueError, TypeError) as e:
                # Log only once per unique problematic column to avoid spam
                log_key = f"conversion_error_{column_name}"
                if not hasattr(self, '_logged_conversion_errors'):
                    self._logged_conversion_errors = set()
                if log_key not in self._logged_conversion_errors:
                    logger.warn(f"Could not convert value '{value}' in column '{column_name}' (row approx {row_idx}) to float: {str(e)}", throttle_duration_sec=10)
                    self._logged_conversion_errors.add(log_key)
                return np.nan

        # Load Trajectory Data
        if self.latest_traj_log_file:
            try:
                current_mod_time = os.path.getmtime(self.latest_traj_log_file)
                if current_mod_time > self.last_traj_read_time:
                    self.get_logger().info(f"Attempting to read trajectory log: {self.latest_traj_log_file}")
                    traj_df = pd.read_csv(self.latest_traj_log_file)
                    if not traj_df.empty:
                        self.get_logger().info(f"Trajectory log loaded. Shape: {traj_df.shape}. Header: {traj_df.columns.tolist()}")
                        self.get_logger().info(f"Trajectory log head:\n{traj_df.head().to_string()}")
                        traj_df.columns = traj_df.columns.str.strip()
                        
                        # Log data types and sample values before conversion
                        for col in ['drone_odom_x', 'drone_odom_y', 'drone_odom_z']:
                            if col in traj_df.columns:
                                self.get_logger().info(f"Column {col} dtype before conversion: {traj_df[col].dtype}")
                                if not traj_df[col].empty:
                                    self.get_logger().info(f"Sample values from {col}: {traj_df[col].head().tolist()}")

                        self.get_logger().info(f"Reloading {self.latest_traj_log_file} ({len(traj_df)} rows)", throttle_duration_sec=2)
                        
                        first_traj_row = traj_df.iloc[0]
                        self.get_logger().info(
                            f"First trajectory row data sample (raw): "
                            f"actual_odom_x='{first_traj_row.get('drone_odom_x', 'N/A')}', "
                            f"actual_odom_y='{first_traj_row.get('drone_odom_y', 'N/A')}', "
                            f"actual_odom_z='{first_traj_row.get('drone_odom_z', 'N/A')}'", 
                            throttle_duration_sec=10
                        )

                        self.timestamps.clear()
                        self.odom_x_data.clear()
                        self.odom_y_data.clear()
                        self.odom_z_data.clear()
                        self.mission_states_numeric.clear()

                        file_start_time = traj_df['timestamp_ros'].iloc[0] if 'timestamp_ros' in traj_df.columns and not traj_df.empty else self.start_time_abs
                        
                        for idx, row in traj_df.iterrows():
                            ts = robust_float_conversion(row.get('timestamp_ros', 0), 'timestamp_ros', idx, self.get_logger()) - file_start_time
                            x = robust_float_conversion(row.get('drone_odom_x'), 'drone_odom_x', idx, self.get_logger())
                            y = robust_float_conversion(row.get('drone_odom_y'), 'drone_odom_y', idx, self.get_logger())
                            z = robust_float_conversion(row.get('drone_odom_z'), 'drone_odom_z', idx, self.get_logger())
                            
                            # Log first few converted values
                            if idx < 5:
                                self.get_logger().info(f"Row {idx} converted values: X={x:.6f}, Y={y:.6f}, Z={z:.6f}")
                            
                            self.timestamps.append(ts)
                            self.odom_x_data.append(x)
                            self.odom_y_data.append(y)
                            self.odom_z_data.append(z)
                            
                            state_str = row.get('mission_state', 'UNKNOWN')
                            if state_str not in self.mission_state_map:
                                self.mission_state_map[state_str] = self.next_mission_state_idx
                                self.next_mission_state_idx += 1
                            self.mission_states_numeric.append(self.mission_state_map[state_str])

                        self.get_logger().info(f"Processed {len(traj_df)} rows from trajectory log. Odom X deque size: {len(self.odom_x_data)}")

                        if 'target_fire_x' in traj_df.columns and pd.notna(traj_df['target_fire_x'].iloc[-1]) and \
                           'target_fire_y' in traj_df.columns and pd.notna(traj_df['target_fire_y'].iloc[-1]):
                            self.current_target_x = robust_float_conversion(traj_df['target_fire_x'].iloc[-1], 'target_fire_x_last', -1, self.get_logger())
                            self.current_target_y = robust_float_conversion(traj_df['target_fire_y'].iloc[-1], 'target_fire_y_last', -1, self.get_logger())
                        else:
                            self.current_target_x = None
                            self.current_target_y = None
                        
                        # Check for all-NaN columns
                        if all(pd.isna(x) for x in self.odom_x_data): self.get_logger().warn("odom_x_data is all NaN after loading trajectory.", throttle_duration_sec=10)
                        if all(pd.isna(y) for y in self.odom_y_data): self.get_logger().warn("odom_y_data is all NaN after loading trajectory.", throttle_duration_sec=10)
                        if all(pd.isna(z) for z in self.odom_z_data): self.get_logger().warn("odom_z_data is all NaN after loading trajectory.", throttle_duration_sec=10)

                        new_data_loaded = True
                    self.last_traj_read_time = current_mod_time
            except Exception as e:
                self.get_logger().error(f"Error loading trajectory CSV '{self.latest_traj_log_file}': {e}", throttle_duration_sec=5)

        if self.latest_det_log_file:
            try:
                current_mod_time = os.path.getmtime(self.latest_det_log_file)
                if current_mod_time > self.last_det_read_time:
                    self.get_logger().info(f"Attempting to read detection log: {self.latest_det_log_file}")
                    det_df = pd.read_csv(self.latest_det_log_file)
                    det_df.columns = det_df.columns.str.strip()
                    self.get_logger().info(f"Detection log loaded. Shape: {det_df.shape}. Header: {det_df.columns.tolist()}")
                    self.get_logger().info(f"Detection log head:\n{det_df.head().to_string()}")
                    
                    fire_df = det_df[det_df['detection_label'] == 'fire'].copy()
                    self.get_logger().info(f"Found {len(fire_df)} 'fire' entries in detection log.")
                    
                    if not fire_df.empty:
                        self.get_logger().info(f"Fire Detection CSV columns (filtered for 'fire'): {fire_df.columns.tolist()}", throttle_duration_sec=10)
                        self.get_logger().info(f"Reloading {self.latest_det_log_file} ({len(fire_df)} fire rows)", throttle_duration_sec=2)
                        
                        first_fire_row = fire_df.iloc[0]
                        self.get_logger().info(
                            f"First fire row data sample (raw): "
                            f"object_world_x='{first_fire_row.get('object_world_x', 'N/A')}', "
                            f"object_world_y='{first_fire_row.get('object_world_y', 'N/A')}'",
                            throttle_duration_sec=10
                        )

                        self.fire_x_data.clear()
                        self.fire_y_data.clear()
                        for idx, row in fire_df.iterrows():
                            self.fire_x_data.append(robust_float_conversion(row.get('object_world_x'), 'object_world_x', idx, self.get_logger()))
                            self.fire_y_data.append(robust_float_conversion(row.get('object_world_y'), 'object_world_y', idx, self.get_logger()))
                        
                        self.get_logger().info(f"Processed {len(fire_df)} fire rows from detection log. Fire X deque size: {len(self.fire_x_data)}")

                        if all(pd.isna(x) for x in self.fire_x_data): self.get_logger().warn("fire_x_data is all NaN after loading detections.", throttle_duration_sec=10)
                        if all(pd.isna(y) for y in self.fire_y_data): self.get_logger().warn("fire_y_data is all NaN after loading detections.", throttle_duration_sec=10)
                        
                        new_data_loaded = True
                    self.last_det_read_time = current_mod_time
            except Exception as e:
                self.get_logger().error(f"Error loading detection CSV '{self.latest_det_log_file}': {e}", throttle_duration_sec=5)
        
        return new_data_loaded

    def init_plots(self):
        # Initialize artists for trajectory map figure
        self.drone_path_line.set_data([], [])
        self.drone_current_pos_scatter.set_offsets(np.empty((0, 2)))
        self.fire_detections_scatter.set_offsets(np.empty((0, 2)))
        self.current_target_scatter.set_offsets(np.empty((0, 2)))
        
        # Set initial limits and properties for trajectory map
        self.ax_traj_map.set_xlim(-10, 10)
        self.ax_traj_map.set_ylim(-10, 10)
        self.ax_traj_map.set_aspect('equal', adjustable='box')
        self.ax_traj_map.grid(True)
        self.ax_traj_map.set_xlabel('X (meters)')
        self.ax_traj_map.set_ylabel('Y (meters)')
        self.ax_traj_map.set_title('2D Trajectory & Fire Map')
        self.ax_traj_map.autoscale(False)  # Disable autoscaling
        
        # Initialize artists for altitude plot
        self.altitude_line.set_data([], [])
        self.ax_alt.set_xlim(0, 10)
        self.ax_alt.set_ylim(-10, 0)
        self.ax_alt.set_xlabel('Time (s)')
        self.ax_alt.set_ylabel('Altitude (m)')
        self.ax_alt.set_title('Altitude Profile')
        self.ax_alt.grid(True)
        
        # Initialize artists for mission state plot
        self.mission_state_line.set_data([], [])
        self.ax_state.set_xlim(0, 10)
        self.ax_state.set_ylim(-0.5, 4.5)
        self.ax_state.set_xlabel('Time (s)')
        self.ax_state.set_ylabel('Mission State')
        self.ax_state.set_title('Mission State Over Time')
        self.ax_state.grid(True)
        
        # Adjust layout to prevent overlapping
        self.fig_traj_map.tight_layout()
        self.fig_other_plots.tight_layout()
        
        # Return all artists that will be updated and blitted
        return (self.drone_path_line, self.drone_current_pos_scatter, 
                self.fire_detections_scatter, self.current_target_scatter, 
                self.altitude_line, self.mission_state_line)

    def update_plots(self, frame):
        # --- Graph 1: 2D Trajectory & Fire Map (on self.ax_traj_map) ---
        all_plot_x_coords = []
        all_plot_y_coords = []

        # Debug print raw data before filtering
        self.get_logger().info(f"Raw data points before filtering - X: {len(self.odom_x_data)}, Y: {len(self.odom_y_data)}")
        if len(self.odom_x_data) > 0:
            self.get_logger().info(f"Raw data sample - First point: X={list(self.odom_x_data)[0]}, Y={list(self.odom_y_data)[0]}")
            self.get_logger().info(f"Raw data sample - Last point: X={list(self.odom_x_data)[-1]}, Y={list(self.odom_y_data)[-1]}")

        # Filter out NaN values for trajectory plotting
        valid_traj_points = [(x, y) for x, y in zip(self.odom_x_data, self.odom_y_data) 
                            if np.isfinite(x) and np.isfinite(y) and abs(x) < 1000 and abs(y) < 1000]  # Basic sanity check
        self.get_logger().info(f"Valid trajectory points after filtering: {len(valid_traj_points)} points")

        # Initialize plot limits
        x_min, x_max = -10, 10  # Default limits
        y_min, y_max = -10, 10  # Default limits
        data_present = False

        if valid_traj_points:
            traj_x, traj_y = zip(*valid_traj_points)
            all_plot_x_coords.extend(traj_x)
            all_plot_y_coords.extend(traj_y)
            
            # Update trajectory line
            self.drone_path_line.set_data(traj_x, traj_y)
            
            # Update current position marker (if available)
            if traj_x and traj_y:
                self.drone_current_pos_scatter.set_offsets(np.array([[traj_x[-1], traj_y[-1]]]))
                data_present = True
                
                # Update limits based on trajectory
                x_min = min(x_min, min(traj_x))
                x_max = max(x_max, max(traj_x))
                y_min = min(y_min, min(traj_y))
                y_max = max(y_max, max(traj_y))
        else:
            self.get_logger().warn("No valid trajectory data to plot")
            self.drone_path_line.set_data([], [])
            self.drone_current_pos_scatter.set_offsets(np.empty((0, 2)))

        # Add fire detection points
        valid_fire_points = [(x, y) for x, y in zip(self.fire_x_data, self.fire_y_data) 
                           if np.isfinite(x) and np.isfinite(y) and abs(x) < 1000 and abs(y) < 1000]
        if valid_fire_points:
            fire_x, fire_y = zip(*valid_fire_points)
            all_plot_x_coords.extend(fire_x)
            all_plot_y_coords.extend(fire_y)
            self.fire_detections_scatter.set_offsets(np.array(list(zip(fire_x, fire_y))))
            data_present = True
            
            # Update limits based on fire detections
            x_min = min(x_min, min(fire_x))
            x_max = max(x_max, max(fire_x))
            y_min = min(y_min, min(fire_y))
            y_max = max(y_max, max(fire_y))
        else:
            self.fire_detections_scatter.set_offsets(np.empty((0, 2)))

        # Add current target
        if (self.current_target_x is not None and self.current_target_y is not None and
            np.isfinite(self.current_target_x) and np.isfinite(self.current_target_y) and
            abs(self.current_target_x) < 1000 and abs(self.current_target_y) < 1000):
            
            all_plot_x_coords.append(self.current_target_x)
            all_plot_y_coords.append(self.current_target_y)
            self.current_target_scatter.set_offsets(np.array([[self.current_target_x, self.current_target_y]]))
            data_present = True
            
            # Update limits based on target
            x_min = min(x_min, self.current_target_x)
            x_max = max(x_max, self.current_target_x)
            y_min = min(y_min, self.current_target_y)
            y_max = max(y_max, self.current_target_y)
        else:
            self.current_target_scatter.set_offsets(np.empty((0, 2)))

        # Calculate and set axis limits with padding
        if data_present:
            # Calculate range and add padding
            x_range = x_max - x_min
            y_range = y_max - y_min
            
            # Minimum range to prevent too small plots
            min_range = 2.0  # At least 2 meters range
            x_range = max(x_range, min_range)
            y_range = max(y_range, min_range)
            
            # Add padding (20% of range)
            x_padding = x_range * 0.2
            y_padding = y_range * 0.2
            
            # Set new limits
            self.ax_traj_map.set_xlim(x_min - x_padding, x_max + x_padding)
            self.ax_traj_map.set_ylim(y_min - y_padding, y_max + y_padding)
        else:
            # Default limits if no data
            self.ax_traj_map.set_xlim(-10, 10)
            self.ax_traj_map.set_ylim(-10, 10)

        # Ensure equal aspect ratio and grid
        self.ax_traj_map.set_aspect('equal', adjustable='box')
        self.ax_traj_map.grid(True)

        # --- Graph 2: Altitude Profile (on self.ax_alt) ---
        valid_alt_points = [(t, z) for t, z in zip(self.timestamps, self.odom_z_data) 
                          if np.isfinite(t) and np.isfinite(z) and abs(z) < 1000]
        if valid_alt_points:
            alt_t, alt_z = zip(*valid_alt_points)
            self.altitude_line.set_data(alt_t, alt_z)
            self.ax_alt.set_xlim(min(alt_t), max(alt_t))
            z_min, z_max = min(alt_z), max(alt_z)
            z_padding = max(0.1 * (z_max - z_min), 0.5)  # At least 0.5m padding
            self.ax_alt.set_ylim(z_min - z_padding, z_max + z_padding)
        else:
            self.altitude_line.set_data([], [])
            self.ax_alt.set_xlim(0, 10)
            self.ax_alt.set_ylim(-10, 0)

        # --- Graph 3: Mission State (on self.ax_state) ---
        valid_state_points = [(t, s) for t, s in zip(self.timestamps, self.mission_states_numeric) 
                            if np.isfinite(t) and np.isfinite(s)]
        if valid_state_points:
            state_t, state_s = zip(*valid_state_points)
            self.mission_state_line.set_data(state_t, state_s)
            self.ax_state.set_xlim(min(state_t), max(state_t))
            self.ax_state.set_ylim(-0.5, max(state_s) + 0.5)

            # Set Y-tick labels to state names
            if self.mission_state_map:
                inv_mission_state_map = {v: k for k, v in self.mission_state_map.items()}
                # Use unique states present in the current data for ticks
                unique_numeric_states_in_plot = sorted(list(set(s for t_val, s in valid_state_points if np.isfinite(s))))

                if unique_numeric_states_in_plot:
                    self.ax_state.set_yticks(unique_numeric_states_in_plot)
                    state_string_labels = [inv_mission_state_map.get(num_state, str(num_state)) for num_state in unique_numeric_states_in_plot]
                    self.ax_state.set_yticklabels(state_string_labels)
                else:
                    # Clear ticks if no valid states to plot or map is empty
                    self.ax_state.set_yticks([])
                    self.ax_state.set_yticklabels([])
            else:
                # Clear ticks if map is empty
                self.ax_state.set_yticks([])
                self.ax_state.set_yticklabels([])

        else:
            self.mission_state_line.set_data([], [])

        # Load new data for next update
        self.load_new_data()

        # Return all artists that were modified
        return [self.drone_path_line, self.drone_current_pos_scatter, 
                self.fire_detections_scatter, self.current_target_scatter,
                self.altitude_line, self.mission_state_line]

def main(args=None):
    rclpy.init(args=args)
    plotter_node = None # Define beforehand for access in finally
    
    try:
        plotter_node = LivePlotterNode() # plt.show() is in its __init__
        # If plt.show() returns normally (window closed by user, not Ctrl+C), 
        # execution continues here.
    except KeyboardInterrupt:
        if plotter_node:
            plotter_node.get_logger().info("Keyboard interrupt detected during plotting. Shutting down.")
        else:
            # This case might occur if interrupt happens very early in plotter_node's __init__
            print("Keyboard interrupt detected during plotter initialization. Shutting down.")
    except Exception as e: # Catch other potential errors during init/plotting
        if plotter_node:
            plotter_node.get_logger().error(f"An unexpected error occurred: {e}. Shutting down.", exc_info=True)
        else:
            print(f"An unexpected error occurred during plotter initialization: {e}")
            import traceback
            traceback.print_exc()
    finally:
        if plotter_node:
            plotter_node.get_logger().info("Plotter shutting down. Closing figures.")
            # Check if figures exist before trying to close
            if hasattr(plotter_node, 'fig_traj_map') and plotter_node.fig_traj_map is not None:
                try:
                    plt.close(plotter_node.fig_traj_map)
                except Exception as e_fig_traj:
                    plotter_node.get_logger().warn(f"Error closing trajectory map figure: {e_fig_traj}")
            
            if hasattr(plotter_node, 'fig_other_plots') and plotter_node.fig_other_plots is not None:
                try:
                    plt.close(plotter_node.fig_other_plots)
                except Exception as e_fig_other:
                    plotter_node.get_logger().warn(f"Error closing other plots figure: {e_fig_other}")

        if rclpy.ok():
            if plotter_node and hasattr(plotter_node, 'destroy_node') and callable(plotter_node.destroy_node):
                 plotter_node.get_logger().info("Destroying plotter ROS node.")
                 plotter_node.destroy_node()
            
            # Log before shutting down rclpy
            if plotter_node: 
                plotter_node.get_logger().info("Shutting down RCLPY.")
            else:
                print("Shutting down RCLPY.")
            rclpy.shutdown()
        
        if plotter_node:
            plotter_node.get_logger().info("Plotter shutdown complete.")
        else:
            print("Plotter shutdown complete.")

if __name__ == '__main__':
    main() 