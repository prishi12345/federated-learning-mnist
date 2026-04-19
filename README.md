# Federated Learning Simulation on MNIST

## Overview
This project implements a basic Federated Learning simulation using PyTorch on the MNIST dataset. The system demonstrates how a global model can be trained collaboratively across multiple clients without sharing raw data, using Federated Averaging.

The project also includes controlled experiments with different data distribution strategies (IID and Non-IID style partitioning) to observe their impact on training behavior.

## Objective
The objective of this project is to implement a basic Federated Learning pipeline using PyTorch on the MNIST dataset and simulate a distributed training environment with multiple clients performing local training. The project aims to study how different data distribution strategies across clients affect the convergence behavior of the global model. It also focuses on understanding the role of Federated Averaging (FedAvg) in aggregating locally trained model updates to form a global model without sharing raw data.

### Model
A simple fully connected neural network is used for MNIST digit classification.

### Federated Setup
* Multiple clients simulate distributed devices
* Each client trains locally on its dataset
* A central model aggregates updates using FedAvg

### Data Distribution
* IID-style split: Random distribution of samples across clients
* Non-IID-style split: Clients receive biased subsets of digit classes 

## Observations
IID-style distribution leads to faster and more stable convergence
Non-IID-style distribution introduces slower convergence due to client data bias
Performance depends heavily on data heterogeneity across clients

## Tech Stack
- Python
- PyTorch
- TorchVision
- Matplotlib







