import argparse
import json
import os
import time
import seaborn as sns
import matplotlib.pyplot as plt
from importlib import import_module
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

import constants
from datasets.base_dataset import RSPTestDataset


def load_model(saved_model, model_name, device, weights):
    """Load the trained model with given weights."""
    loc = "model.custom_model"
    model_module = getattr(import_module(loc), model_name)
    model = model_module().to(device)

    model_path = os.path.join(saved_model, weights)
    assert os.path.isfile(model_path), f"Model weights not found at {model_path}"

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    return model


def plot_confusion_matrix(cm, class_names, save_name):
    """Plots the confusion matrix using Seaborn."""
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.savefig(f'{save_name}/cm.png', dpi=300, bbox_inches="tight")  
    print(f"Confusion matrix saved at: {save_name}")


@torch.inference_mode()
def test(model, data_loader, device):
    """Runs inference and computes accuracy & F1-score."""
    print("Calculating inference results....")
    model.eval()

    all_preds = []
    all_labels = []

    with tqdm(total=len(data_loader), desc="Inference Progress", unit="batch") as pbar:
        for idx, (images, labels) in enumerate(data_loader):
            # print(f"Batch [{idx + 1}/{len(data_loader)}] | Image Shape: {tuple(images.shape)}")
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)  # Get raw model outputs
            preds = torch.argmax(outputs, dim=1)  # Convert logits to class predictions

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Convert lists to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Compute Metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    cm = confusion_matrix(all_labels, all_preds)

    # Plot confusion matrix
    class_names = ["Rock", "Scissors", "Paper"]
    plot_confusion_matrix(cm, class_names, args.exp_path)

    return len(all_labels), accuracy, f1, cm


def inference(args):
    """Main inference function."""
    start = time.time()

    # Load model
    model = load_model(args.exp_path, args.model_name, args.device, args.weights)

    # Load dataset
    IMG_ROOT = os.path.join(args.data_dir, "test")
    dataset = RSPTestDataset(root_dir=IMG_ROOT)
    
    IMG_SIZE = (300, 400)
    transform_module = getattr(import_module("datasets.augmentation"), args.augmentation)  # default: TrainAugmentation
    test_transform = transform_module(img_size=IMG_SIZE, is_train=False)
    dataset.set_transform(test_transform)

    test_loader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, drop_last=False
    )

    # Run inference
    total_images, accuracy, f1, cm = test(model, test_loader, args.device)

    print(f"\n🔹 Inference Summary 🔹")
    print(f"Exp. name          : {args.exp}")
    print(f"Test Images Count  : {total_images}")
    print(f"Model Accuracy     : {accuracy*100:.2f}")
    print(f"Model F1-score     : {f1:.4f}")
    print(f"Confusion Matrix   :\n{cm}")
    print(f"Inference Time     : {time.time() - start:.3f}s")


# Run: python inference.py --exp MobileNetV2_CE --data_dir "/content/drive/MyDrive/Mini_Project/DB/images/250307"
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Check CUDA availability
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    assert use_cuda, "CUDA is not available!"

    # Define arguments
    parser.add_argument("--exp", type=str, default="Baseline", help="Experiment directory")
    parser.add_argument("--device", type=str, default=device, help="Device (cuda or cpu)")
    parser.add_argument("--weights", type=str, default="best_epoch.pth", help="Model weights file")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for inference")
    parser.add_argument("--augmentation", type=str, default="TrainAugmentation", help="Augmentation method")

    # Container environment
    parser.add_argument("--data_dir", type=str, default=os.environ.get("SM_CHANNEL_EVAL", "/content/drive/MyDrive/Mini_Project/DB/images/"))

    args = parser.parse_args()
    args.exp_path = os.path.join("./outputs", args.exp)

    # Load model configuration
    json_file = next((file for file in os.listdir(args.exp_path) if file.endswith(".json")), None)
    assert json_file, "No configuration JSON file found!"

    json_path = os.path.join(args.exp_path, json_file)
    with open(json_path, "r") as f:
        config = json.load(f)

    args.model_name = config["model"]

    # Start inference
    inference(args)
