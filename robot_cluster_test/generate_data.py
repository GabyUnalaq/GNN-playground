import os
import numpy as np
import pickle
from typing import Optional

from graph import Graph
from .cluster_visualizer import FlockingVisualizer

ROBOTS_NUMBER = [3, 5]  # Number or range (min, max) of robots in the cluster
SAMPLES_NUM = 100  # Number of samples to generate
VIZ = False # Whether to visualize each sample
SAVE_DATA = True  # Whether to save generated data to a file

FILE_NAME = "test_data_{robots_num}r_{SAMPLES_NUM}s.pkl"
WITH_TARGET = True  # Whether to include target position in the data

script_dir = os.path.dirname(os.path.abspath(__file__))

def generate_flocking_control(robot_states: np.array, target_pos: Optional[np.array]):
    """A simple flocking control algorithm."""
    robots_num = robot_states.shape[0]
    controls = np.zeros((robots_num, 2))  # vx, vy for each robot

    # Flocking parameters
    cohesion_weight = 0.3
    separation_weight = 0.5
    target_weight = 0.0 if target_pos is None else 0.5
    separation_radius = 0.1

    for i in range(robots_num):
        # 1. Cohesion: move towards the center of mass of neighbors
        center_of_mass = np.mean(robot_states, axis=0)
        cohesion = center_of_mass - robot_states[i]
        
        # 2. Separation: avoid getting too close to neighbors
        separation = np.zeros(2)
        for j in range(robots_num):
            if i != j:
                diff = robot_states[i] - robot_states[j]
                distance = np.linalg.norm(diff)
                if distance < separation_radius and distance > 0:
                    separation += diff / distance
        
        # 3. Target seeking: move towards the target position
        target_direction = target_pos - robot_states[i] if WITH_TARGET else np.zeros(2)
        
        # Combine all components
        controls[i] = (cohesion_weight * cohesion +
                      separation_weight * separation +
                      target_weight * target_direction)
        
        # Normalize control to prevent extreme velocities
        control_magnitude = np.linalg.norm(controls[i])
        if control_magnitude > 1.0:
            controls[i] = controls[i] / control_magnitude

    return controls

if __name__ == "__main__":
    data = {
        "robots_number": ROBOTS_NUMBER,
        "samples_num": SAMPLES_NUM,
        "graphs_in": [],
        "graphs_out": [],
        "targets": [],
    }

    if VIZ:
        viz_samples = []
    for sample_idx in range(SAMPLES_NUM):
        if isinstance(ROBOTS_NUMBER, int):
            robots_num = ROBOTS_NUMBER
        else:
            robots_num = np.random.randint(ROBOTS_NUMBER[0], ROBOTS_NUMBER[1] + 1)

        robot_states = np.random.rand(robots_num, 2)  # x, y
        target_pos = np.random.rand(2) if WITH_TARGET else None # x, y

        controls = generate_flocking_control(robot_states, target_pos)

        if VIZ: 
            viz_samples.append((robot_states, target_pos, controls))
        
        in_graph = Graph({
            "num_nodes": robots_num,
            "node_feature_dim": 2,
            "edge_feature_dim": 0,
            "node_features": robot_states.tolist(),
            "edges": [[i, j] for i in range(robots_num) for j in range(robots_num) if i != j]
        })
        out_graph = Graph({
            "num_nodes": robots_num,
            "node_feature_dim": 2,
            "edge_feature_dim": 0,
            "node_features": controls.tolist(),
            "edges": [[i, j] for i in range(robots_num) for j in range(robots_num) if i != j]
        })

        data["graphs_in"].append(in_graph)
        data["graphs_out"].append(out_graph)
        data["targets"].append(target_pos)

    if VIZ:
        FlockingVisualizer(viz_samples)
    if SAVE_DATA:
        if isinstance(ROBOTS_NUMBER, int):
            robots_str = str(ROBOTS_NUMBER)
        else:
            robots_str = f"{ROBOTS_NUMBER[0]}-{ROBOTS_NUMBER[1]}"
        
        file_name = FILE_NAME.format(robots_num=robots_str, SAMPLES_NUM=SAMPLES_NUM)
        
        with open(os.path.join(script_dir, "data", file_name), "wb") as f:
            pickle.dump(data, f)
        print(f"Saved generated data to {file_name}")
