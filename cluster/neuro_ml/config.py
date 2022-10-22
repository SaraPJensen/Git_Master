from pathlib import Path
import os

SEED = 42
EPOCHS = 101
BATCH_SIZE = 1
LEARNING_RATE = 0.005 #0.003
TRAIN_VAL_TEST_SIZE = [0.7, 0.15, 0.15]

if os.path.exists("/home/users/sarapje"):
    dataset_path = Path("/home/users/sarapje/snn-glm-simulator/spiking_network/data/")

else:
    dataset_path = Path("/Users/Sara/Desktop/Master/snn-glm-simulator/spiking_network/data/")
