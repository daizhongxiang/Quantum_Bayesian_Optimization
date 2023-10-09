# Code for the submitted paper "Quantum Bayesian Optimization"
This directory contains the code for the paper [Quantum Bayesian Optimization](https://daizhongxiang.github.io/quantum_bo.pdf) accepted to NeurIPS 2023, including both the synthetic experiment and the AutoML experiment (Sec. 6).


## Requirements:
'pip install -r requirements.txt'

## Implementation of Algorithms:
Classical GP-UCB: bayesian_optimization_bo.py, helper_funcs_bo.py
Q-GP-UCB (ours): bayesian_optimization_quantum.py, helper_funcs_quantum.py

## Running our experiments:
The directory "synth" contains the code for running the synthetic experiment, and the directory "automl" contains the code for running the AutoML experiment. Under both directories, 
- The directory "results_bo" saves the results for classical GP-UCB, "results_quantum" saves the results for Q-GP-UCB (ours).
- The notebook "analyze.ipynb" contains the code to analyze the results and plot the figures in the main paper.
- The directory "saved_synth_funcs" contains the generated synthetic function and the random features (because we use random Fourier features approximation in the implementation, see Appendix I for more details). The notebook "generate_synth_func.ipynb" contains the code to generate the synthetic function and the random features. There is no need to run it, since the synthetic function and the random features used in our experiments are already saved in the directory "saved_synth_funcs" as discussed above.

Under "synth":
- "synth_bo.py": runs the classical GP-UCB algorithm, for both the Bernoulli noise and Gaussian noise.
- "synth_quantum.py": runs our Q-GP-UCB algorithm for the Gaussian noise.
- "synth_quantum_binary.py": runs our Q-GP-UCB algorithm for the Bernoulli noise.

Under "automl":
- "svm_bo.py": runs the classical GP-UCB algorithm.
- "svm_quantum.py": runs our Q-GP-UCB algorithm.
- The directory "clinical_data" saves the data used to train the SVM.
