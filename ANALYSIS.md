# Technical Analysis — Quantum vs. Classical Simulation Performance

## Hypothesis

For small-scale quantum circuits (N < 10 qubits), it's reasonable to expect a highly-optimized simulator backend like Qiskit Aer (C++ internals, vectorized linear algebra) to outperform a naive Python/NumPy implementation. This project set out to quantify that gap. As shown below, the measured result is the opposite of that hypothesis at this scale, which is itself the more interesting finding.

## Methodology

- **Circuit Generation:** Random circuits parameterized by qubit count (2, 3, 4) and circuit depth (5, 10, 20), built from H, X, Z, and CNOT gates.
- **Quantum Time:** Wall-clock time to transpile and run the circuit on the Qiskit Aer statevector simulator (via `AerSimulator(method="statevector")`), including the `transpile()` call.
- **Classical Time:** Wall-clock time for the custom NumPy simulator to apply the equivalent gate sequence via direct matrix multiplication on the statevector.
- **Correctness Check:** Before trusting any timing comparison, every benchmarked circuit's classical output was checked against the Qiskit statevector output via state fidelity. All runs below report fidelity 1.0 (within floating-point precision) — the two simulators agree exactly. This is a hard prerequisite: a timing comparison between two simulators is meaningless if they aren't actually computing the same answer (see the project history for two real bit-ordering and transpiler-layout bugs that previously caused them not to).
- **Noise Analysis:** A depolarizing noise model (5% 1-qubit error, 10% 2-qubit error) is available via `QuantumFramework(noise_level=...)` and used for sampling noisy measurement counts (`simulate_qiskit_noisy_counts`). It is **not** applied to the ideal-statevector timing comparison below, since that path intentionally measures noiseless, exact evolution on both sides.

## Results & Discussion

### Benchmarking Data

*Each cell is the mean over 8 random circuits per configuration; timings in milliseconds.*

| Qubits | Depth | Quantum Time (ms) | Classical Time (ms) | Classical Speedup | Fidelity |
|---|---|---|---|---|---|
| 2 | 5  | 98.3 | 0.46 | 213x | 1.000000 |
| 2 | 10 | 91.2 | 0.39 | 234x | 1.000000 |
| 2 | 20 | 91.9 | 0.62 | 149x | 1.000000 |
| 3 | 5  | 91.5 | 0.30 | 308x | 1.000000 |
| 3 | 10 | 91.9 | 0.37 | 248x | 1.000000 |
| 3 | 20 | 92.1 | 0.63 | 147x | 1.000000 |
| 4 | 5  | 91.7 | 0.28 | 329x | 1.000000 |
| 4 | 10 | 93.2 | 0.42 | 223x | 1.000000 |
| 4 | 20 | 91.4 | 0.62 | 148x | 1.000000 |

### Key Findings

- **Custom simulator wins at this scale — by a lot.** Contrary to the initial hypothesis, the naive NumPy implementation outperformed Qiskit Aer by roughly **150x–330x** across every tested configuration. This isn't because NumPy's linear algebra is faster than Aer's — it's because Aer's per-call fixed overhead (Python↔C++ marshalling, circuit transpilation, backend job dispatch) is roughly ~90ms regardless of circuit size, and at 2-4 qubits the actual computation is trivially small (state vectors of length 4-16). The custom simulator has none of that overhead: it's a handful of small matrix multiplications directly in Python/NumPy.
- **Quantum time is roughly flat.** Aer's time barely changes across qubit count (2→4) or depth (5→20) in this range, consistent with the bottleneck being fixed per-call overhead rather than the size of the linear algebra itself.
- **Classical time grows mildly.** The custom simulator's time increases slightly with depth (more gate applications) and is largely insensitive to qubit count in this narrow 2-4 qubit range, since state vectors are still tiny (4 to 16 complex amplitudes).
- **Where this likely reverses.** Aer's real advantage — compiled tensor contractions, cache-aware memory layouts, multi-threading — should start to dominate once state vectors get large enough that raw compute time exceeds Aer's fixed overhead, likely somewhere in the 12-20+ qubit range, where state vectors reach 4,096–1,000,000+ amplitudes and the custom simulator's per-gate full-statevector reshape/transpose operations become the bottleneck instead.

## Conclusion & Future Work

At small scale (2-4 qubits) — a fixed-cost-dominated regime — the "industrial-grade" simulator backend is dramatically *slower* than a from-scratch implementation, not faster: a useful, somewhat counterintuitive result about where simulator overhead actually lives. The original hypothesis (a modest ~1.5x Aer advantage even at small scale) did not hold up against directly measured data; this revision replaces invented/assumed numbers with real measurements after fixing two correctness bugs in the simulator's gate-application logic that previously caused the two simulators to silently disagree.

**Future Directions:**
- Extend benchmarking well beyond this range (8, 12, 16, 20+ qubits) to find the actual crossover point where Aer's per-call overhead stops dominating and its low-level optimizations start winning.
- Separate Aer's transpilation time from its execution time, to see how much of the ~90ms fixed cost is transpilation versus simulation itself.
- Implement the Quantum Fourier Transform (QFT) or Grover's algorithm to benchmark specific algorithms instead of random circuits.
- Run the same circuits on a real quantum processing unit (QPU) through IBM Quantum to compare simulation time against queue time and execution time on noisy hardware.
