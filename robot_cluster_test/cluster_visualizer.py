import matplotlib.pyplot as plt
import numpy as np

DRAW_SEPARATION = False  # Not useful

class FlockingVisualizer:
    def __init__(self, samples_data):
        """
        samples_data: list of tuples (robot_states, target_pos, controls)
        """
        self.samples_data = samples_data
        self.current_idx = 0
        self.max_idx = len(samples_data) - 1
        
        # Create figure and axis
        self.fig, self.ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # Connect keyboard events
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        # Show first sample
        self.update_plot()
        plt.show()
    
    def on_key_press(self, event):
        """Handle keyboard navigation."""
        if event.key == 'right' or event.key == 'n':  # Next
            if self.current_idx < self.max_idx:
                self.current_idx += 1
                self.update_plot()
        elif event.key == 'left' or event.key == 'p':  # Previous
            if self.current_idx > 0:
                self.current_idx -= 1
                self.update_plot()
    
    def update_plot(self):
        """Update the plot with current sample."""
        self.ax.clear()
        
        robot_states, target_pos, controls = self.samples_data[self.current_idx]
        
        # Set plot limits
        self.ax.set_xlim(-0.1, 1.1)
        self.ax.set_ylim(-0.1, 1.1)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlabel('X Position')
        self.ax.set_ylabel('Y Position')
        self.ax.set_title(f'Robot Flocking - Sample {self.current_idx + 1}/{len(self.samples_data)} (Use ← → or P/N to navigate)')
        
        # Draw robots
        for i in range(len(robot_states)):
            circle = plt.Circle(robot_states[i], 0.03, color='blue', alpha=0.7)
            self.ax.add_patch(circle)
            
            self.ax.text(robot_states[i, 0], robot_states[i, 1] + 0.05, 
                    f'R{i}', ha='center', fontsize=9, fontweight='bold')
            
            if np.linalg.norm(controls[i]) > 0:
                self.ax.arrow(robot_states[i, 0], robot_states[i, 1],
                        controls[i, 0] * 0.15, controls[i, 1] * 0.15,
                        head_width=0.03, head_length=0.02, 
                        fc='red', ec='red', alpha=0.7, linewidth=1.5)
            
            # Draw separation radius circle
            if DRAW_SEPARATION:
                separation_circle = plt.Circle(robot_states[i], 0.2,  # Considering 0.2 as separation radius
                                                color='blue', fill=False, 
                                                linestyle='--', alpha=0.3, linewidth=1)
                self.ax.add_patch(separation_circle)
        
        # Draw target
        if target_pos is not None:
            self.ax.plot(target_pos[0], target_pos[1], 'g*', markersize=20, 
                    label='Target', markeredgecolor='black', markeredgewidth=1)
        

        
        # Display parameters
        robot_poses_text = "\n".join([f"R{i}: ({robot_states[i,0]:.2f}, {robot_states[i,1]:.2f})" for i in range(len(robot_states))])
        target_pos_text = f"Target: ({target_pos[0]:.2f}, {target_pos[1]:.2f})" if target_pos is not None else "Target: N/A"
        param_text = (
            f"{target_pos_text}\n"
            f"Robot Poses:\n{robot_poses_text}\n"
        )
        self.ax.text(0.02, 0.98, param_text, transform=self.ax.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        if target_pos is not None:
            self.ax.legend(loc='upper right')
        self.fig.canvas.draw_idle()

if __name__ == "__main__":
    samples = []
    for i in range(3):
        robot_states = np.random.rand(3, 2)  # 3 robots with (x,y) positions
        target_pos = np.random.rand(2)  # Target position
        controls = np.random.randn(3, 2) * 0.1  # Random control vectors
        samples.append((robot_states, target_pos, controls))
    
    visualizer = FlockingVisualizer(samples)
