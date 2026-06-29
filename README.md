Quantum Circuit Simulator & Performance Benchmarking Framework
An independent research project investigating the computational trade-offs between classical simulations of quantum systems and executions on quantum simulator backends.
Abstract
By utilizing NumPy and Qiskit, this script simulates basic quantum circuits. Furthermore, this script includes gate operations, noise modeling, and a benchmarking system that contrasts quantum circuit execution time with classical matrix operations. This was produced as an individual research endeavor to explore quantum computing fundamentals and performance analysis.
Key Features

* From-Scratch Simulator: Implements core quantum gates (H, X, Z, CNOT) and state evolution using linear algebra with NumPy.
* Industry-Standard Integration: Leverages Qiskit and its Aer simulator for performance comparison and advanced features.
* Correctness Validation: Includes a built-in `validate()` method that cross-checks the from-scratch simulator against Qiskit's statevector simulator across randomized circuits (varying qubit counts and depths) to confirm the two agree to numerical precision before any benchmark numbers are trusted.
* Noise Modeling: Integrates a customizable depolarizing noise model to study algorithmic robustness.
* Automated Benchmarking: Systematically benchmarks circuits of configurable qubit count and depth to generate performance data.
* Extensible Framework: Built with an object-oriented design for easy expansion with new gates, algorithms, and error models.
Technical Insights & Analysis
See the full technical analysis and results here: [ANALYSIS.md](https://github.com/Hibahf/Qubit-Simulator/blob/main/ANALYSIS.md)
Installation & Usage
Prerequisites

* Python 3.8 or higher
* pip (Python package manager)

1. Clone the repository:

```
git clone https://github.com/Hibahf/Qubit-Simulator.git
cd Qubit-Simulator
```

2. Install required dependencies:

```
pip install -r requirements.txt
```

Usage
After installation, you can run the framework to benchmark circuits:

```
from quantum_sim import QuantumFramework

# Initialize the framework (optional noise)
fw = QuantumFramework(noise_level=0.02)

# Optional: confirm the custom simulator agrees with Qiskit before trusting benchmark numbers
fw.validate()

# Benchmark a random circuit
df, qc, qstate, cstate = fw.benchmark(num_qubits=2, depth=4, seed=42)

# Display results
print("Random Circuit:\n", qc)
print("\nQiskit (ideal) statevector:\n", qstate)
print("\nClassical (custom) statevector:\n", cstate)
print("\nBenchmark Results:\n", df)
```

Note: if your local module file is named differently (e.g. `quantum_framework.py`), adjust the import in the example above to match.
