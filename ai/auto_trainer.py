import os

BASE_PATH = "/content/drive/MyDrive/Mini_Project"
CONFIG_PATH = os.path.join(BASE_PATH, "configs")
EXPERIMENT_PATH = os.path.join(BASE_PATH, "outputs")

CONFIG_QUEUE_PATH = os.path.join(CONFIG_PATH, "queue")
CONFIG_ENDS_PATH = os.path.join(CONFIG_PATH, "ends")

configs = [file for file in os.listdir(CONFIG_QUEUE_PATH) if file != ".gitkeep"]

if configs:
    print(f"current config file: {configs[0]}")

    # train.py
    os.system(f"python train.py --config {os.path.join(CONFIG_QUEUE_PATH, configs[0])}")
    os.system(f"mv {os.path.join(CONFIG_QUEUE_PATH, configs[0])} {os.path.join(CONFIG_ENDS_PATH, configs[0])}")