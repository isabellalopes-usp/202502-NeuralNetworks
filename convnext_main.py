# import de pacotes e bibliotecas
import torch.optim as optim
import torch.nn as nn
import torch

import numpy as np
import random

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import models

from convnext_training import train_model, test_model, test_results, training_results
from convnext_dataset import WildfireDataset


if __name__ == "__main__":
    # definicao de semente para a metodo rand, para que a separacao e data augmentation dos datasets seja sempre a mesma
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    generator = torch.Generator()
    generator.manual_seed(seed)

    # Data augmentation para treino
    train_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(10),
            transforms.Resize(224),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Processamento para teste e validação
    processing_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(224),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # carregar dataset
    data_path = "wildfire"

    class_list = ["nowildfire", "wildfire"]
    idx_to_class = {0: "nowildfire", 1: "wildfire"}
    class_to_idx = {v: k for k, v in idx_to_class.items()}

    train_dataset = WildfireDataset(
        root_dir=data_path,
        stage="train",
        class_to_idx=class_to_idx,
        idx_to_class=idx_to_class,
        transform=train_transform,
    )

    test_dataset = WildfireDataset(
        root_dir=data_path,
        stage="test",
        class_to_idx=class_to_idx,
        idx_to_class=idx_to_class,
        transform=processing_transform,
    )

    valid_dataset = WildfireDataset(
        root_dir=data_path,
        stage="valid",
        class_to_idx=class_to_idx,
        idx_to_class=idx_to_class,
        transform=processing_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=8,
        persistent_workers=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        valid_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=0,
    )

    # Definicao do modelo
    model = models.convnext_tiny(weights="IMAGENET1K_V1")
    model.classifier[-1] = nn.Linear(in_features=768, out_features=1, bias=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Congelamento de parâmetros
    for name, param in model.named_parameters():
        param.requires_grad = False
    for name, param in model.classifier.named_parameters():
        param.requires_grad = True

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.002)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.2, patience=4
    )

    train_model(
        model,
        criterion,
        optimizer,
        scheduler,
        train_loader,
        val_loader,
        num_epochs=100,
        model_name="convnext_lessaug",
        early_stop_patience=8,
    )

    test_metrics = test_model(test_loader, model)

    test_results(test_metrics, "convnext_lessaug")
    training_results("convnext_lessaug")
