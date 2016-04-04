Fomoro TensorFlow Starter
===

Starter project for the [getting started](https://fomoro.gitbooks.io/guide/content/getting_started.html) guide. Based on [this TensorFlow tutorial](https://www.tensorflow.org/versions/master/tutorials/mnist/pros/index.html#build-a-multilayer-convolutional-network).

## Training

### Cloud Setup

1. Follow the [installation guide](https://fomoro.gitbooks.io/guide/content/installation.html) for Fomoro.
2. Clone the repo: `git clone https://github.com/fomorians/fomoro-tensorflow.git && cd fomoro-tensorflow`
3. Create a new model: `fomoro model add`
4. Start training: `fomoro session start -f`

### Local Setup

1. [Install TensorFlow](https://www.tensorflow.org/versions/r0.7/get_started/os_setup.html#pip-installation).
2. Clone the repo: `git clone https://github.com/fomorians/fomoro-tensorflow.git && cd fomoro-tensorflow`
3. Run training: `python main.py`

## Evaluation

Evaluate a previously trained model: `python main.py --skip-training --restore`
