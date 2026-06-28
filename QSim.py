import numpy as np
import pandas as pd
import random
import time
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit.circuit.library import HGate, XGate, ZGate, CXGate


def kron_n(matrices):
    res = np.array([[1.0]])
    for m in matrices:
        res = np.kron(res, m)
    return res


def apply_gate_statevector(state, gate_matrix, targets, num_qubits):
    """
    Apply `gate_matrix` (a 2^k x 2^k unitary) to the qubits listed in
    `targets` (Qiskit qubit indices, given in qarg order), on a statevector
    of `num_qubits` qubits that follows Qiskit's bit-ordering convention:
    qubit 0 is the LEAST significant bit of the state index.

    Two bit-ordering subtleties matter here, and getting either wrong
    silently scrambles results for n > 1 qubits without raising any error:

    1. `state.reshape([2] * num_qubits)` puts the MOST significant bit on
       tensor axis 0 (numpy is row-major/C-order), so Qiskit qubit `q`
       lives on tensor axis `num_qubits - 1 - q`, not axis `q`.
    2. Qiskit's own gate matrices (via Gate.to_matrix()) use the convention
       that the FIRST qarg is the least-significant qubit of the local
       2^k-dim subspace. So when selecting axes for a multi-qubit gate,
       `targets` must be taken in REVERSED order to line up with that
       local matrix convention.
    """
    k = len(targets)
    if k == 0:
        return state

    n = num_qubits

    def axis_of(q):
        return n - 1 - q

    target_axes = [axis_of(q) for q in reversed(targets)]
    remaining_axes = [a for a in range(n) if a not in target_axes]

    state_tensor = state.reshape([2] * n)
    perm = target_axes + remaining_axes
    inv_perm = np.argsort(perm)

    permuted = np.transpose(state_tensor, axes=perm)
    dim_left = 2 ** k
    dim_right = 2 ** (n - k)
    reshaped = permuted.reshape((dim_left, dim_right))

    after = gate_matrix @ reshaped

    after_tensor = after.reshape([2] * n)
    final = np.transpose(after_tensor, axes=inv_perm)
    return final.reshape(2 ** n)


def random_gate(num_qubits):
    single_qubit_gates = [HGate(), XGate(), ZGate()]
    two_qubit_gates = [CXGate()]
    if num_qubits == 1 or random.random() < 0.7:
        gate = random.choice(single_qubit_gates)
        target = [random.randint(0, num_qubits - 1)]
        return gate, target
    else:
        gate = random.choice(two_qubit_gates)
        control = random.randint(0, num_qubits - 1)
        target = random.choice([q for q in range(num_qubits) if q != control])
        return gate, [control, target]


def random_circuit(num_qubits, depth):
    qc = QuantumCircuit(num_qubits)
    for _ in range(depth):
        gate, targets = random_gate(num_qubits)
        qc.append(gate, targets)
    return qc


class QuantumFramework:
    def __init__(self, noise_level=0.0):
        self.noise_level = float(noise_level)
        self._backend_sv = AerSimulator(method="statevector")
        self._backend_qasm = AerSimulator(method="density_matrix")
        self.noise_model = self._make_noise_model() if self.noise_level > 0 else None

    def _make_noise_model(self):
        nm = NoiseModel()
        error_1q = depolarizing_error(self.noise_level, 1)
        error_2q = depolarizing_error(min(1.0, self.noise_level * 2.0), 2)
        nm.add_all_qubit_quantum_error(error_1q, ['h', 'x', 'z'])
        nm.add_all_qubit_quantum_error(error_2q, ['cx'])
        return nm

    def simulate_qiskit_statevector(self, qc: QuantumCircuit):
        from qiskit import transpile
        qc_sv = qc.copy()
        qc_sv.save_statevector()
        tqc = transpile(qc_sv, self._backend_sv, optimization_level=0)
        result = self._backend_sv.run(tqc).result()
        return np.array(result.get_statevector(tqc))

    def simulate_qiskit_noisy_counts(self, qc: QuantumCircuit, shots=4096):
        from qiskit import transpile
        qc_copy = qc.copy()
        if qc_copy.num_clbits == 0:
            qc_copy = qc_copy.measure_all(inplace=False)
        tqc = transpile(qc_copy, self._backend_qasm)
        job = self._backend_qasm.run(tqc, noise_model=self.noise_model, shots=shots)
        return job.result().get_counts()

    def simulate_classical_statevector(self, qc: QuantumCircuit):
        num_qubits = qc.num_qubits
        state = np.zeros(2 ** num_qubits, dtype=complex)
        state[0] = 1.0
        for circuit_instruction in qc.data:
            instr = circuit_instruction.operation
            qargs = circuit_instruction.qubits
            qubit_indices = [qc.find_bit(q).index for q in qargs]
            assert len(set(qubit_indices)) == len(qubit_indices), "Duplicate target qubits"
            assert all(0 <= t < num_qubits for t in qubit_indices), "Target qubit out of range"
            if hasattr(instr, "to_matrix"):
                mat = instr.to_matrix()
            else:
                name = instr.name.lower()
                if name == 'h':
                    mat = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
                elif name == 'x':
                    mat = np.array([[0, 1], [1, 0]])
                elif name == 'z':
                    mat = np.array([[1, 0], [0, -1]])
                elif name in ('cx', 'cnot'):
                    mat = np.array([
                        [1, 0, 0, 0],
                        [0, 0, 0, 1],
                        [0, 0, 1, 0],
                        [0, 1, 0, 0]
                    ])
                else:
                    raise ValueError(f"Unsupported gate without to_matrix(): {instr.name}")
            state = apply_gate_statevector(state, mat, qubit_indices, num_qubits)
        return state

    @staticmethod
    def fidelity(state_a, state_b):
        na = np.linalg.norm(state_a)
        nb = np.linalg.norm(state_b)
        if na < 1e-12 or nb < 1e-12:
            return 0.0
        inner = np.vdot(state_a, state_b) / (na * nb)
        return float(np.abs(inner) ** 2)

    def benchmark(self, num_qubits=3, depth=6, seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        qc = random_circuit(num_qubits, depth)
        t0 = time.time()
        qstate = self.simulate_qiskit_statevector(qc)
        qt = time.time() - t0
        t0 = time.time()
        cstate = self.simulate_classical_statevector(qc)
        ct = time.time() - t0
        fid = self.fidelity(cstate, qstate)
        df = pd.DataFrame([{
            "Qubits": num_qubits,
            "Depth": depth,
            "Qiskit Time (s)": qt,
            "Classical Time (s)": ct,
            "Fidelity": fid
        }])
        return df, qc, qstate, cstate

    def validate(self, qubit_range=range(1, 6), depths=(3, 6, 10), trials=10, tol=1e-6):
        """
        Sweep many random circuits across qubit counts/depths and confirm the
        classical simulator agrees with Qiskit's statevector simulator
        (fidelity ~= 1.0 for all of them). Returns a results DataFrame and
        raises AssertionError on the first mismatch it finds.
        """
        rows = []
        for n in qubit_range:
            for depth in depths:
                for seed in range(trials):
                    df, qc, qstate, cstate = self.benchmark(num_qubits=n, depth=depth, seed=seed)
                    fid = df["Fidelity"].iloc[0]
                    rows.append({"Qubits": n, "Depth": depth, "Seed": seed, "Fidelity": fid})
                    assert fid > 1 - tol, (
                        f"Mismatch vs Qiskit at qubits={n}, depth={depth}, seed={seed}: "
                        f"fidelity={fid}"
                    )
        return pd.DataFrame(rows)


if __name__ == "__main__":
    fw = QuantumFramework(noise_level=0.02)

    print("Running correctness sweep against Qiskit (no noise)...")
    results = fw.validate()
    print(f"All {len(results)} trials matched Qiskit "
          f"(worst fidelity: {results['Fidelity'].min():.6f})\n")

    df, qc, qstate, cstate = fw.benchmark(num_qubits=2, depth=4, seed=42)
    print("Random Circuit:\n")
    print(qc)
    print("\nQiskit (ideal) statevector:\n", np.round(qstate, 5))
    print("\nClassical (custom) statevector:\n", np.round(cstate, 5))
    print("\nBenchmark:\n", df.to_string(index=False))
    noisy_counts = fw.simulate_qiskit_noisy_counts(qc, shots=2048)
    print("\nNoisy counts (qasm simulator with noise):\n", noisy_counts)
