import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from medpy.metric import binary

from vit_model.model import VisionTransformer
from vit_model.config import Config
from dataset import HyperspectralDataset


def evaluate_metrics_medpy(preds, labels, num_classes=2):
    preds = preds.cpu().numpy().astype(np.uint8)
    labels = labels.cpu().numpy().astype(np.uint8)
    metrics = {}

    for cls in range(num_classes):
        pred_bin = (preds == cls).astype(np.uint8)
        label_bin = (labels == cls).astype(np.uint8)

        try:
            dice = binary.dc(pred_bin, label_bin)
        except:
            dice = 0.0
        try:
            iou = binary.jc(pred_bin, label_bin)
        except:
            iou = 0.0
        try:
            sensitivity = binary.sensitivity(pred_bin, label_bin)
        except:
            sensitivity = 0.0
        try:
            specificity = binary.specificity(pred_bin, label_bin)
        except:
            specificity = 0.0
        try:
            precision = binary.precision(pred_bin, label_bin)
        except:
            precision = 0.0
        try:
            acc = binary.accuracy(pred_bin, label_bin)
        except:
            acc = 0.0

        metrics[cls] = {
            "dice": round(dice, 4),
            "iou": round(iou, 4),
            "acc": round(acc, 4),
            "precision": round(precision, 4),
            "recall": round(sensitivity, 4),
            "specificity": round(specificity, 4)
        }

    return metrics


def evaluate_model(model, loader, device, num_classes):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for hsi, rgb, labels in loader:
            hsi, rgb, labels = hsi.to(device), rgb.to(device), labels.to(device)
            outputs, _, _, _ = model(rgb, hsi)
            preds = torch.argmax(outputs, dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    return evaluate_metrics_medpy(all_preds, all_labels, num_classes=num_classes)


def train():
    device = torch.device(Config.device)

    train_dataset = HyperspectralDataset(
        hsi_dir=Config.hyperspectral_dir,
        rgb_dir=Config.rgb_dir,
        label_dir=Config.label_dir,
        list_file=Config.train_list,
        train=True,
        transform=None,
        image_size=Config.image_size
    )

    val_dataset = HyperspectralDataset(
        hsi_dir=Config.hyperspectral_dir,
        rgb_dir=Config.rgb_dir,
        label_dir=Config.label_dir,
        list_file=Config.test_list,
        train=False,
        transform=None,
        image_size=Config.image_size
    )

    train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True,
                              num_workers=Config.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.batch_size, shuffle=False,
                            num_workers=Config.num_workers, pin_memory=True)

    model = VisionTransformer(Config.model_config, img_size=Config.image_size, num_classes=Config.num_classes)
    model.to(device)

    criterion = nn.CrossEntropyLoss(weight=torch.tensor(Config.class_weights).to(device))
    optimizer = optim.AdamW(model.parameters(), lr=Config.learning_rate)

    model.train()
    for epoch in range(Config.num_epochs):
        running_loss = 0.0
        optimizer.zero_grad()

        for step, (hsi, rgb, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.num_epochs}")):
            hsi, rgb, labels = hsi.to(device), rgb.to(device), labels.to(device)

            outputs, reg_loss, _, _ = model(rgb, hsi)
            loss = criterion(outputs, labels)

            if reg_loss is not None:
                if isinstance(reg_loss, torch.Tensor):
                    loss += reg_loss.mean()
                else:
                    loss += torch.as_tensor(reg_loss, device=device)

            loss.backward()

            if (step + 1) % Config.accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"\nEpoch [{epoch+1}/{Config.num_epochs}] Loss: {avg_loss:.4f}")

        train_metrics = evaluate_model(model, train_loader, device, Config.num_classes)
        val_metrics = evaluate_model(model, val_loader, device, Config.num_classes)

        print("Train Metrics:")
        for cls, values in train_metrics.items():
            print(f"  Class {cls}: {values}")
        print("Val Metrics:")
        for cls, values in val_metrics.items():
            print(f"  Class {cls}: {values}")

        if (epoch + 1) % Config.validation_freq == 0:
            os.makedirs(Config.checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(Config.checkpoint_dir, f"model_epoch{epoch+1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")


if __name__ == '__main__':
    train()
