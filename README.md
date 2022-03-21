# sweet-perceptron
![Python](https://img.shields.io/badge/Python-%3E%3D3.8-brightgreen)
![NumPy](https://img.shields.io/badge/NumPy-%3E%3D1.15-blue)

This is a simple neural network library.

The model of NN is configurable multilayer Perceptron (MLP).

It was built only using [NumPy](https://numpy.org/).

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install [sweet-perceptron](https://pypi.org/project/sweet-perceptron).

```bash
pip install sweet-perceptron
```

## Usage

```python
from perceptron import NeuralNetwork
from perceptron.Neuron import functions

# Initialize network
nw = NeuralNetwork((2, 4, 1))

# Change activation functions
nw.activation_funcs = functions.relu, functions.tanh

# Initialize weights
nw.initialize_weights()

# Train netwotk and get list with losses
loss = nw.train(50, ['train inputs set'], ['train outputs set'])

# Get output of forward propagation
nw.forward(['input'])
```

## License

![GitHub](https://img.shields.io/github/license/SweetBubaleXXX/sweet-perceptron)