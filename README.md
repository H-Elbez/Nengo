# Nengo STDP

STDP implementation for Nengo and single layer spiking neural network

### Training

Run next command for training.
```bash
python Mnist_STDP.py
```

### Evaluation

After doing training a new folder will be generated with neurons synapses heatmaps and a binar file that contains the network parameters and weights (mnist_params_STDP), to run the evaluation we need to pass that file in the command.
```bash
python Mnist_STDP_evaluation.py {mnist_params_STDP}
```

A work is still needed to improve STDP by running parameters exploration.
