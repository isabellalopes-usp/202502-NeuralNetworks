import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
import numpy as np
import kagglehub
import os

from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from PIL import ImageFile

def create_cnn():

    '''
    camadas convolucionais:
      - camada 1: 32 filtros de 3x3 (detectar features básicas)
      - camada 2: 64 filtros de 3x3 (detectar features mais complexas)
      - camada 3: 128 filtros de 3x3 (features de alto nível, como formas e objetos)
      - camada 4: 128 filtros de 3x3 (refinamento)

    max pooling:
      preserva fortes ativações, que tem mais chance de detectar padrões relevantes.
      também reduz overfitting por que se baseia em "diminuir" a imagem, forçando o
      modelo a focar em features mais robustas, não aprendendo detalhes específicos
      das imagens

    camadas densas:
      - flatten: deixar em uma dimensão
      - dropout: 0.5 para reduzir overfitting após o flatten e na camada densa principal
      - dense 512: camada conectada (combinar features)
      - dense 1 com func de ativação sigmoid: camada de saída para classificação binária

    '''
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(350, 350, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5), # Adicionado Dropout
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5), # Adicionado Dropout
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model

ImageFile.LOAD_TRUNCATED_IMAGES = True

'''
data augmentation: "cria mais dados" mudando as features como
escala, rotação, tamanho e orientação
'''

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_data_dir = os.path.join(path, 'train')
test_data_dir = os.path.join(path, 'test')

#treino
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(350, 350),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

#validação
validation_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(350, 350),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

#teste
test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(350, 350),
    batch_size=32,
    class_mode='binary',
    shuffle=False #mantem a ordem das classes e previsões
)

model = create_cnn()
print(model.summary())

'''
early stopping: para se nao melhorar por 4 épocas
reduceLROnPlatea: reduz o learning rate na estagnação (3 épocas)
'''

callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=3)
]

history = model.fit(
    train_generator,
    epochs=20,
    validation_data=validation_generator,
    callbacks=callbacks
)

sns.set_style("whitegrid")

#avaliação e métricas

print("\n--- AVALIAÇÃO NO CONJUNTO DE TESTE ---\n")
test_loss, test_accuracy = model.evaluate(test_generator, verbose=0)

print(f"test_loss: {test_loss:.4f}")
print(f"test_accuracy: {test_accuracy:.4f}")

test_generator.reset() #reseta o gerador de teste, garantindo que a ordem dos labels seja a mesma das previsões
y_pred_probs = model.predict(test_generator, steps=len(test_generator), verbose=0)
y_pred_classes = (y_pred_probs > 0.5).astype(int)
y_true = test_generator.classes

class_labels = list(test_generator.class_indices.keys())

print("\n--- MATRIZ DE CONFUSÃO ---\n")
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_labels, yticklabels=class_labels)
plt.title('Matriz de confusão')
plt.ylabel('Label verdadeiro')
plt.xlabel('Label predito')
plt.show()

print("\n--- RELATÓRIO DE CLASSIFICAÇÃO ---\n")
print(classification_report(
    y_true,
    y_pred_classes,
    target_names=class_labels,
    digits=4
))

fpr, tpr, thresholds = roc_curve(y_true, y_pred_probs)
roc_auc = auc(fpr, tpr)
print(f"Área sob a curva ROC (AUC): {roc_auc:.4f}")

def plot_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    #gráfico de loss
    ax1.plot(history.history['loss'], label='Train loss')
    ax1.plot(history.history['val_loss'], label='Validation loss')
    ax1.set_title('Train loss and validation loss by epoch')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (binary crossentropy)')
    ax1.legend()

    #gráfico de acurácia
    ax2.plot(history.history['accuracy'], label='Train acc')
    ax2.plot(history.history['val_accuracy'], label='Validation acc')
    ax2.set_title('Train and validation acc by epoch')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    plt.tight_layout()
    plt.show()

plot_history(history)

#gráfico ROC
plt.figure(figsize=(7, 7))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taxa de Falsos Positivos (FPR)')
plt.ylabel('Taxa de Verdadeiros Positivos (TPR) - Recall')
plt.title('Curva ROC')
plt.legend(loc="lower right")
plt.show()
