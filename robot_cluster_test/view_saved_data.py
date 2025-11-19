import os
import pickle
import numpy as np

from .cluster_visualizer import FlockingVisualizer
from graph import Graph

DATA_FILE = "test_data_3-5r_50s_no_targets.pkl"

script_dir = os.path.dirname(os.path.abspath(__file__))


if __name__ == "__main__":
    data_path = os.path.join(script_dir, "data", DATA_FILE)

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}.")

    with open(data_path, "rb") as f:
        data = pickle.load(f)

    samples_data = []
    for i in range(data["samples_num"]):
        robot_states = data["graphs_in"][i].node_features.numpy()
        target_pos = data["targets"][i]
        controls = data["graphs_out"][i].node_features.numpy()
        samples_data.append((robot_states, target_pos, controls))

    FlockingVisualizer(samples_data)
