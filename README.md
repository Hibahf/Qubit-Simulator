# Quantum Circuit Simulator & Performance Benchmarking Framework

An independent research project investigating the computational trade-offs between classical simulations of quantum systems and executions on quantum simulator backends.

## Abstract

By utilizing NumPy and Qiskit, this script simulates basic quantum circuits. Furthermore, this script includes gate operations, noise modeling, and a benchmarking system that contrasts quantum circuit execution time with classic matrix operations. This was produced as an individual research endeavor to explore quantum computing fundamentals and performance analysis.

## Key Features

*   **From-Scratch Simulator:** Implements core quantum gates (H, X, Z, CNOT) and state evolution using linear algebra with NumPy.
*   **Industry-Standard Integration:** Leverages Qiskit and its Aer simulator for performance comparison and advanced features.
*   **Noise Modeling:** Integrates a customizable depolarizing noise model to study algorithmic robustness.
*   **Automated Benchmarking:** Systematically tests circuits of varying qubit count (2-4) and depth (5-20) to generate performance data.
*   **Extensible Framework:** Built with an object-oriented design for easy expansion with new gates, algorithms, and error models.

## Technical Insights & Analysis

See the full technical analysis and results here: [ANALYSIS.md](ANALYSIS.md)
