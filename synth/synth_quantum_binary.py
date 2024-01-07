import sys
sys.path.append('../')

from bayesian_optimization_quantum import QBO
import pickle
import numpy as np

from multiprocessing.dummy import Pool as ThreadPool

from qiskit import QuantumCircuit
from qiskit.algorithms import IterativeAmplitudeEstimation, EstimationProblem
from qiskit.circuit.library import LinearAmplitudeFunction
from qiskit_aer.primitives import Sampler
from qiskit_finance.circuit.library import NormalDistribution

from qiskit.circuit.library import RYGate

from qiskit_aer.noise import NoiseModel
from qiskit.providers.fake_provider import FakeWashington

device = FakeWashington()
coupling_map = device.configuration().coupling_map
noise_model = NoiseModel.from_backend(device)


quantum_noise = False
binary = True # always set this to True
linear_bandit = False

max_iter = int(1e4)

ls = 0.1

obs_noise = 0.4**2
log_file_name = "saved_synth_funcs/synth_func_ls_" + str(ls) + "_noise_var_" + str(obs_noise) + ".pkl"
all_func_info = pickle.load(open(log_file_name, "rb"))
domain = all_func_info["domain"]
f = all_func_info["f"]

log_file_name = "saved_synth_funcs/random_features_ls_" + str(ls) + "_noise_var_" + str(obs_noise) + ".pkl"
random_features = pickle.load(open(log_file_name, "rb"))

def synth_func(param, eps):
    x = param[0]
    ind = np.argmin(np.abs(domain - x))
    
    num_uncertainty_qubits = 1
    
    p = f[ind, 0]
    low = 0
    high = 1
    
    # Calculate the rotation angle
    theta = 2 * np.arcsin(np.sqrt(p))
    # Create a quantum circuit
    qc = QuantumCircuit(num_uncertainty_qubits)

    # Apply the RY rotation
    qc.append(RYGate(theta), [0])
    uncertainty_model = qc

    monte_carlo = uncertainty_model

    epsilon = eps

    objective_qubits = [0]
    seed = 0

    epsilon = np.clip(epsilon, 1e-6, 0.5)

    alpha = 0.05
    max_shots = 32 * np.log(2/alpha*np.log2(np.pi/(4*epsilon))) 
    
    post_processing = lambda x: x

    problem = EstimationProblem(
        state_preparation=monte_carlo,
        objective_qubits=objective_qubits,
        post_processing=post_processing,)

    if quantum_noise == True:
        ae = IterativeAmplitudeEstimation(
            epsilon_target=epsilon, alpha=alpha, sampler=Sampler(backend_options={
                "method": "density_matrix",
                "coupling_map": coupling_map,
                "noise_model": noise_model,
            },run_options={"shots": int(np.ceil(max_shots)),"seed_simulator":seed},
            transpile_options={"seed_transpiler": seed},)
        )
    else:
        ae = IterativeAmplitudeEstimation(
            epsilon_target=epsilon, alpha=alpha, sampler=Sampler(run_options={"shots": int(np.ceil(max_shots)),"seed_simulator":seed}))

    # Running result
    result = ae.estimate(problem)

    est = result.estimation_processed

    num_oracle_queries = result.num_oracle_queries

    if num_oracle_queries == 0:
        # use the number of oracle calls given by the paper if num_oracle_queries == 0
        num_oracle_queries = int(np.ceil((0.8 / epsilon) * np.log((2 / alpha) * np.log2(np.pi / (4 * epsilon)))))

    return est, p, num_oracle_queries

ts = np.arange(1, max_iter)
beta_t = 1 + np.sqrt(np.log(ts) ** 2)

run_list = np.arange(10)
for itr in run_list:
    np.random.seed(itr)

    log_file_name = "results_quantum/res_iter_" + str(itr) + "_binary.pkl"

    if linear_bandit:
        log_file_name = log_file_name[:-4] + "_linear_bandit.pkl"
    if quantum_noise:
        log_file_name = log_file_name[:-4] + "_quantum_noise.pkl"

    quantum_BO = QBO(f=synth_func, pbounds={'x1':(0, 1)}, log_file=log_file_name, beta_t=beta_t, \
              random_features=random_features, linear_bandit=linear_bandit, domain=domain)
    quantum_BO.maximize(n_iter=max_iter, init_points=1)
