# Graph Neural Network playground

Using this repository for playing around with the GNN theory and for testing small ideas.

## Python setup

I am using [uv](https://docs.astral.sh/uv/pip/environments/) to set my python env.

I am using `python 3.10` and `torch` with cuda support for cuda 12.1. For the Graph specific torch library, I had to install also `torch-geometric`.

## Scripts

- `run_gcn.py` - script that uses a simple graph database and a custom `GCN` net that contains 2 `GCNConv` layers which practically mean 2 message passing layers. It trains for 200 epochs and print accuracy. The time is shown as well.
- `run_simple_net.py` - script uses the custom message passing layer impl, doing (more or less) the same as the GCN one. It uses the same database and same metrics.

## Layers

The layers are found under the `net/layers.py` script. It contains both standard GNN layers and Adaptive GNN layers (that affect the edges).
