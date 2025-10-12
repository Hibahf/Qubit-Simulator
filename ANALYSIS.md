# Technical Analysis - Quantum vs. Classical Simulation Performance

## Hypothesis

For small-scaled quantum circuits (N < 10 qubits), highly-optimized classical simulators (like Qiskit Aer) are expected to outperform naive Python/Numpy implementations due to their C++ backends and advanced optimization techniques. This project aims to quantify that performance gap.

## Methodology

1. **Circuit Generation:* Generated random circuits parameterized by qubit count (2, 3, 4) and circuit depth (5, 10, 20).
2. **Execution Timing:**
     *  **Quantum Time:** Measurewd time to transpile and run the circuit on the Qiskit Aer `statevector` simulator.
     *  **Classical Time:** Measured time for the custom NumPy simulator to apply an equivalent sequence of gates via matrix multiplication.
3. **Noise Analysis:** Introduced a depolarizing noise model (5% 1-qubit error, 10% 2 qubit error) to observe its effect on circuit fidelity.

## Results and Discussion

### Benchmarking Data

### Key Findings

