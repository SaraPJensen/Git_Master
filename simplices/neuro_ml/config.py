from pathlib import Path
import os

SEED = 42
EPOCHS = 50
BATCH_SIZE = 1
#LR = 0.001 current best with message passing, need higher for GCNConv
#LR = 0.005 works better for the MP_Outer 
LEARNING_RATE = 0.001 #0.001 #0.005 #0.003    #So far decreasing has made the model better, at least the training loss 
TRAIN_VAL_TEST_SIZE = [0.7, 0.15, 0.15]

# if os.path.exists("/home/users/sarapje"):
#     dataset_path = Path("/home/users/sarapje/snn-glm-simulator/spiking_network/data/")

# else:
#     dataset_path = Path("/Users/Sara/Desktop/Master/snn-glm-simulator/spiking_network/data/")

dataset_path = Path("graph_data/")
