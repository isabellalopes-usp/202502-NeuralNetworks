from torchmetrics import (
    MetricCollection,
    PrecisionRecallCurve,
    AveragePrecision,
    Accuracy,
    AUROC,
    ROC,
    ConfusionMatrix,
    Precision,
    F1Score,
)
from tqdm import tqdm

import seaborn as sns
import pandas as pd
import torch
import csv
import os

# caminho onde serao salvos os checkpoints de todos os modelos
checkpoint_path = "models/"


def get_acc(
    labels: torch.Tensor, output: torch.Tensor, limiar: float, activation: torch.nn
):

    # função de ativação - sigmoide
    output = activation(output)

    y_pred_binario = torch.where(output >= limiar, 1, 0)

    return labels.eq(y_pred_binario).sum().item()


def get_acc(
    labels: torch.Tensor, output: torch.Tensor, limiar: float, activation: torch.nn
):

    # função de ativação - sigmoide
    output = activation(output)

    y_pred_binario = torch.where(output >= limiar, 1, 0)

    return labels.eq(y_pred_binario).sum().item()


# salva checkpoints
def save_checkpoint(state, filename):
    torch.save(state, filename)


# atualiza checkpoint
def save_model_stats(model, train_losses, val_losses, train_accs, val_accs):
    csv_filename = model + ".csv"
    with open(csv_filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", "train_loss", "val_loss", "train_acc", "val_acc"])
        for epoch in range(len(train_losses)):
            writer.writerow(
                [
                    epoch + 1,
                    train_losses[epoch],
                    val_losses[epoch],
                    train_accs[epoch],
                    val_accs[epoch],
                ]
            )


# funcao geral para treinamento e validacao dos modelos
def train_model(
    model,
    criterion,
    optimizer,
    scheduler,
    train_loader,
    val_loader,
    early_stop_patience,
    num_epochs=10,
    start_epoch=0,
    model_name="model",
    is_inception=False,
):
    os.makedirs(f"models/{model_name}/", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    average_precision = AveragePrecision(task="binary")
    activation = torch.nn.Sigmoid()
    threshold = 0.5

    early_stop_counter = early_stop_patience
    best_avg_prec = 0.0
    train_avg_precs = []
    val_avg_precs = []
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    # treinamento do modelo
    # itera sobre o numero de epocas definido
    for epoch in range(start_epoch, num_epochs):

        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # itera sobre as classes no dataset de treino
        for images, labels in tqdm(train_loader, desc=f"Train Epoch {epoch}"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            # calculo da loss
            if is_inception and model.training:
                outputs, aux_outputs = model(images)

                loss1 = criterion(outputs, labels)
                loss2 = criterion(aux_outputs, labels)
                loss = loss1 + 0.4 * loss2
            else:
                outputs = torch.squeeze(model(images))
                loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            # _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += get_acc(labels, outputs, threshold, activation)
            average_precision(outputs, labels.long())

        # calculo da loss e da accuracy finais
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_avg_prec = average_precision.compute()
        epoch_acc = 100.0 * correct / total
        train_avg_precs.append(epoch_avg_prec)
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)
        print(
            f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%, Average Precision: {epoch_avg_prec:.2f}"
        )
        average_precision.reset()

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            # itera sobre as classes no dataset de validacao
            for images, labels in tqdm(val_loader, desc=f"Val Epoch {epoch}"):
                images, labels = images.to(device), labels.to(device)

                outputs = torch.squeeze(model(images))

                # calculo da loss
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                total += labels.size(0)

                correct += get_acc(labels, outputs, threshold, activation)
                average_precision(outputs, labels.long())
        # calculo da loss e da accuracy finais
        val_avg_prec = average_precision.compute()
        val_loss /= len(val_loader.dataset)
        val_acc = 100.0 * correct / total
        val_avg_precs.append(val_avg_prec)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        print(
            f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%, Validation Average Precision: {val_avg_prec:.2f}"
        )
        average_precision.reset()

        scheduler.step(val_avg_prec)

        early_stop_counter -= 1
        # salva/atualiza checkpoint apenas se a accuracy da epoca atual for maior que a maior accuracy armazenada ate o momento
        if val_avg_prec > best_avg_prec:
            best_avg_prec = val_avg_prec
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "loss": val_loss,
                    "accuracy": val_acc,
                    "average_precision": val_avg_prec,
                },
                filename=f"{checkpoint_path}/{model_name}/model.pth",
            )
            early_stop_counter = early_stop_patience
        save_model_stats(
            f"{checkpoint_path}/{model_name}/training",
            train_losses,
            val_losses,
            train_accs,
            val_accs,
        )

        if early_stop_counter == 0:
            break


def test_model(test_loader, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metrics = MetricCollection(
        {
            "AvgPr": AveragePrecision(task="binary"),
            "AUROC": AUROC(task="binary"),
            "Acc": Accuracy(task="binary"),
            "Prec": Precision(task="binary"),
            "F1": F1Score(task="binary"),
            "PrecisionRecall": PrecisionRecallCurve(task="binary"),
            "ROC": ROC(task="binary"),
            "CF": ConfusionMatrix(task="binary", normalize="true"),
        }
    ).to(device)
    activation = torch.nn.Sigmoid()

    for images, labels in tqdm(test_loader, desc=f"Testing"):
        images, labels = images.to(device), labels.to(device)
        outputs = activation(torch.squeeze(model(images)))

        metrics(outputs, labels.long())

    results = metrics.compute()
    metrics.reset()

    return results


def plot_curve(value, key, x_name, y_name):
    print(f"Plotting {key}...", end="\t")

    data = {x_name: value[0].cpu(), y_name: value[1].cpu()}

    plot = sns.lineplot(data=pd.DataFrame(data), x=x_name, y=y_name)
    plot.set_title(key)
    fig = plot.get_figure()
    fig.savefig(f"models/{model_name}/{key}.png")
    fig.clf()
    print(f"Saved to: models/{model_name}/{key}.png")


def test_results(metrics, model_name):
    print()
    for key, value in metrics.items():
        if key == "CF":
            print(f"Plotting {key}...", end="\t")
            plot = sns.heatmap(
                data=value.cpu(), annot=True, annot_kws={"fontsize": 24}, cmap="Blues"
            )
            plot.set_title("Confusion Matrix")

            fig = plot.get_figure()
            fig.savefig(f"models/{model_name}/{key}.png")
            fig.clf()

            print(f"Saved to: models/{model_name}/{key}.png")
        elif type(value) != tuple:
            print(f"{key}\t{value:.4f}")
        else:
            if key == "PrecisionRecall":
                x_name = "Recall"
                y_name = "Precision"
            elif key == "ROC":
                x_name = "FPR"
                y_name = "TPR"
            plot_curve(value, key, x_name, y_name)


def training_results(model_name):
    df = pd.read_csv(f"models/{model_name}/training.csv")

    dfm = df[["train_loss", "val_loss", "epoch"]].melt(
        "epoch", var_name="Curves", value_name="loss"
    )

    plot = sns.lineplot(data=dfm, x="epoch", y="loss", hue="Curves")
    plot.set_title("Loss Treinamento")
    fig = plot.get_figure()
    fig.savefig(f"models/{model_name}/loss.png")
    fig.clf()


if __name__ == "__main__":
    from torchvision import models, transforms
    from torch.utils.data import DataLoader
    from convnext_dataset import WildfireDataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = "convnext_lessaug"
    data_path = "wildfire"
    idx_to_class = {0: "nowildfire", 1: "wildfire"}
    class_to_idx = {v: k for k, v in idx_to_class.items()}

    processing_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(224),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    test_dataset = WildfireDataset(
        root_dir=data_path,
        stage="test",
        class_to_idx=class_to_idx,
        idx_to_class=idx_to_class,
        transform=processing_transform,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=3,
    )

    checkpoint = torch.load(f"models/{model_name}/model.pth")
    model = models.convnext_tiny(weights="IMAGENET1K_V1")
    model.classifier[-1] = torch.nn.Linear(in_features=768, out_features=1, bias=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    test_metrics = test_model(test_loader, model)
    test_results(test_metrics, model_name)

    training_results(model_name)
