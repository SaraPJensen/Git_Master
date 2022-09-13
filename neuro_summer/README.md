# Neuro ml summer project

## Install

You need to install pytorch, preferably with CUDA. In addition, install `requirements.txt`

```console
$ pip install -r requirements.txt
```

## Usage

### Simulation code

```console
$ python -m mikkel_sim
```

```
usage: __main__.py [-h] [-d N] [-s NUM_OF_STEPS] [-n NUM_NEURONS]

Create simulated neuron datasets

options:
  -h, --help            show this help message and exit
  -d N, --num-data N    Generate d datasets (default: 100)
  -s NUM_OF_STEPS, --num-of-steps NUM_OF_STEPS
                        Number of time steps for the simulation (default: 100000)
  -n NUM_NEURONS, --num-neurons NUM_NEURONS
                        Use n neurons in simulation (default: 20)
```

### Train/test model

Make sure to specify the correct arguments in `neuro_ml/__main__.py`, inside `if __name__ == "__main__"`

```console
$ python -m neuro_ml 
```
