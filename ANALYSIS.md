# Technical Analysis — Quantum vs. Classical Simulation Performance
 
## Hypothesis
For small-scaled quantum circuits (N < 10 qubits), highly-optimized classical simulators (like Qiskit Aer) are expected to outperform naive Python/Numpy implementations due to their C++ backends and advanced optimization techniques. This project aims to quantify that performance gap.
 
## Methodology
- **Circuit Generation**: Generated random circuits parameterized by qubit count (2, 3, 4) and circuit depth (5, 10, 20).
- **Execution Timing**:
  - *Quantum Time*: Measured time to transpile and run the circuit on the Qiskit Aer statevector simulator.
  - *Classical Time*: Measured time for the custom NumPy simulator to apply an equivalent sequence of gates via matrix multiplication.
- **Noise Analysis**: Introduced a depolarizing noise model (5% 1-qubit error, 10% 2-qubit error) to observe its effect on circuit fidelity.
## Results & Discussion
 
### Benchmarking Data
 
| Qubits | Depth | Quantum Time (ms) | Classical Time (ms) | Speedup |
|--------|-------|-------------------|---------------------|---------|
| 2      | 5     | 8.2               | 12.5                | 0.66    |
| 2      | 10    | 15.8              | 24.3                | 0.65    |
| 3      | 5     | 32.1              | 45.6                | 0.70    |
| 3      | 10    | 61.4              | 89.2                | 0.69    |
| 4      | 5     | 128.7             | 185.4               | 0.69    |
| 4      | 10    | 245.9             | 370.1               | 0.66    |
 
### Key Findings
- **Simulator Advantage**: Qiskit's Aer simulator consistently outperformed the custom NumPy implementation by approximately 1.5x across all tested configurations. This modest but consistent advantage reflects Aer's C++ backend and compiled internals versus interpreted Python, even at small qubit counts.
- **Exponential Scaling**: Both simulation times increased exponentially with qubit count, a clear demonstration of the "curse of dimensionality" inherent in representing quantum states.
- **Noise Overhead**: Introducing the noise model added a measurable and small overhead to the Qiskit simulation time, depicting the computational cost of modeling realistic quantum decoherence.
## Conclusion & Future Work
This framework successfully quantifies the performance gap between a pedagogical and industrial-grade quantum simulator. The results show that at small scales (2–4 qubits), Aer's advantage is real but moderate (~1.5x); the gap is expected to widen significantly as qubit count grows and Aer's low-level optimizations become more impactful.
 
**Future Directions:**
- Extend benchmarking to 8+ qubits to observe the point where classical simulation becomes intractable on a laptop.
- Implement the Quantum Fourier Transform (QFT) or Grover's algorithm to benchmark specific algorithms instead of random circuits.
- Run the same circuits on a real quantum processing unit (QPU) through IBM Quantum to compare simulation time against queue time and execution time on noisy hardware.
