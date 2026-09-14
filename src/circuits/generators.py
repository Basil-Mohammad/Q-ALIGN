"""
Concrete PennyLane circuit construction and training, isolated behind this
module so the quantum backend can be swapped (e.g. for Qiskit) without
touching any alignment math (src/alignment/*) -- manuscript
PHASE0_AUDIT.md decision #1.

Backend: PennyLane `default.qubit` (pure-Python statevector simulator).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Callable
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

from src.circuits.encodings import AngleEncodingCircuit
from src.circuits.complexity import CircuitComplexity


@dataclass
class QAlignCircuit:
    """A trainable PQC: one AngleEncodingCircuit (fixes Omega_C, i.e. the
    spectral inductive bias) + one entangling-layout list of (i, j, tau)
    tuples per layer (fixes G_C, i.e. the topological inductive bias).
    Trainable parameters are single-qubit rotation angles interleaved
    between data-encoding and entangling blocks (a standard data
    re-uploading ansatz, Perez-Salinas et al. 2020 / Schuld et al. 2021).
    """
    encoding: AngleEncodingCircuit
    entangling_layers: List[List[Tuple[int, int, float]]]  # one entry per encoding layer
    circuit_id: str

    def __post_init__(self):
        if len(self.entangling_layers) != len(self.encoding.layers):
            raise ValueError("Must provide one entangling layout per encoding layer.")

    @property
    def n_qubits(self) -> int:
        return self.encoding.n_qubits

    def n_parameters(self) -> int:
        # 3 trainable rotation angles per qubit per layer (a generic
        # single-qubit unitary decomposition), a standard convention.
        return 3 * self.n_qubits * len(self.encoding.layers)

    def complexity(self) -> CircuitComplexity:
        n_two_qubit = sum(len(layer) for layer in self.entangling_layers)
        return CircuitComplexity(
            n_qubits=self.n_qubits,
            n_parameters=self.n_parameters(),
            depth=len(self.encoding.layers),
            n_two_qubit_gates=n_two_qubit,
        )

    def qnode(self):
        dev = qml.device("default.qubit", wires=self.n_qubits)

        n_layers = len(self.encoding.layers)

        @qml.qnode(dev, diff_method="parameter-shift")
        def circuit(x: np.ndarray, theta: np.ndarray):
            # theta shape: (n_layers, n_qubits, 3)
            for l, enc_layer in enumerate(self.encoding.layers):
                for q in range(self.n_qubits):
                    var = enc_layer.qubit_variable_map[q]
                    if var is not None:
                        mult = enc_layer.frequency_multipliers[q]
                        qml.RX(mult * x[var], wires=q)
                    a, b, c = theta[l, q]
                    qml.Rot(a, b, c, wires=q)
                for (i, j, _tau) in self.entangling_layers[l]:
                    qml.CNOT(wires=[i, j])
            return qml.expval(qml.PauliZ(0))

        return circuit

    def predict(self, X: np.ndarray, theta: np.ndarray) -> np.ndarray:
        qnode = self.qnode()
        return np.array([qnode(x, theta) for x in np.atleast_2d(X)])


def random_init_params(n_layers: int, n_qubits: int, rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(-np.pi, np.pi, size=(n_layers, n_qubits, 3))


def mse_loss(circuit: QAlignCircuit, X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> float:
    preds = circuit.predict(X, theta)
    return float(np.mean((preds - y) ** 2))


def train_partial(circuit: QAlignCircuit, X: np.ndarray, y: np.ndarray,
                   theta0: np.ndarray, n_steps: int, lr: float = 0.1) -> dict:
    """Fixed, pre-registered 'partial training' protocol (manuscript
    PHASE0_AUDIT.md decision #6): a FIXED number of gradient-descent steps,
    not 'train until convergence' -- convergence-based stopping would make
    the cost-ratio claim in manuscript Sec 5.6 ill-defined and gameable.
    Uses PennyLane's parameter-shift gradients via a simple manual
    finite-difference-free autodiff loop (qml.GradientDescentOptimizer).
    """
    qnode = circuit.qnode()

    def cost_fn(theta):
        preds = qml.math.stack([qnode(x, theta) for x in X])
        return qml.math.mean((preds - y) ** 2)

    opt = qml.GradientDescentOptimizer(stepsize=lr)
    theta = pnp.array(theta0, requires_grad=True)
    loss_trace = []
    for step in range(n_steps):
        theta, loss = opt.step_and_cost(cost_fn, theta)
        loss_trace.append(float(loss))
    final_loss = mse_loss(circuit, X, y, theta)
    return {
        "final_theta": np.array(theta).tolist(),
        "loss_trace": loss_trace,
        "final_mse": final_loss,
        "n_steps": n_steps,
    }


def gradient_variance_trainability_proxy(circuit: QAlignCircuit, X: np.ndarray, y: np.ndarray,
                                          rng: np.random.Generator, n_inits: int = 20) -> float:
    """Barren-plateau-style trainability proxy (manuscript
    PHASE0_AUDIT.md decision #7): sample-variance of the loss gradient
    (w.r.t. the first-layer, first-qubit parameter, a standard single-
    parameter proxy per McClean et al. 2018) over `n_inits` random
    initializations.
    """
    qnode = circuit.qnode()
    n_layers = len(circuit.encoding.layers)

    def cost_fn(theta):
        preds = qml.math.stack([qnode(x, theta) for x in X])
        return qml.math.mean((preds - y) ** 2)

    grad_fn = qml.grad(cost_fn)
    first_component_grads = []
    for _ in range(n_inits):
        theta0 = random_init_params(n_layers, circuit.n_qubits, rng)
        theta0 = pnp.array(theta0, requires_grad=True)
        g = grad_fn(theta0)
        first_component_grads.append(float(np.array(g).ravel()[0]))
    return float(np.var(first_component_grads))
