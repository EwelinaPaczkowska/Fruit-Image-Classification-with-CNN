#!/usr/bin/env python3
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt


# ============================================================
# TODO 1: Wczytanie danych
# ============================================================

def wczytaj_dane(sciezka_danych):
    klasy = ["banana", "orange", "lemon"]
    X = []
    y = []

    rozmiar = (128, 128)

    for idx, klasa in enumerate(klasy):
        sciezka_klasy = os.path.join(sciezka_danych, klasa)

        for plik in os.listdir(sciezka_klasy):
            sciezka = os.path.join(sciezka_klasy, plik)

            img = cv2.imread(sciezka)
            if img is None:
                continue

            img = cv2.resize(img, rozmiar)
            img = img / 255.0

            X.append(img)
            y.append(idx)

    X = np.array(X, dtype="float32")
    y = to_categorical(y, num_classes=3)

    return X, y


# ============================================================
# TODO 2: Augmentacja
# ============================================================

def przygotuj_augmentacje():
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        zoom_range=0.1,
        validation_split=0.2
    )
    return datagen


# ============================================================
# TODO 3: Model CNN
# ============================================================

def zbuduj_model(input_shape, liczba_klas):
    model = models.Sequential()

    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D(2, 2))

    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(2, 2))

    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(2, 2))

    model.add(layers.Flatten())

    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(liczba_klas, activation='softmax'))

    return model


# ============================================================
# TODO 4: Kompilacja
# ============================================================

def skompiluj_model(model):
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )


# ============================================================
# TODO 5: Trening
# ============================================================

def trenuj_model(model, X_train, y_train):

    datagen = przygotuj_augmentacje()

    train_generator = datagen.flow(
        X_train, y_train,
        batch_size=32,
        subset='training'
    )

    val_generator = datagen.flow(
        X_train, y_train,
        batch_size=32,
        subset='validation'
    )

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    historia = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=20,
        callbacks=[early_stop]
    )

    return historia


# ============================================================
# TODO 6: Klasyfikacja
# ============================================================

def klasyfikuj_obraz(model, sciezka_obrazu):
    klasy = ["Banan", "Pomarańcza", "Cytryna"]

    img = cv2.imread(sciezka_obrazu)
    img = cv2.resize(img, (128, 128))
    img = img / 255.0

    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)[0]

    idx = np.argmax(pred)

    print("Rozpoznano:", klasy[idx])
    print("Prawdopodobieństwa:")
    for i in range(len(klasy)):
        print(f"{klasy[i]}: {pred[i]:.2f}")


# ============================================================
# Wykresy
# ============================================================

def rysuj_wykresy(historia):
    plt.plot(historia.history['accuracy'], label='train accuracy')
    plt.plot(historia.history['val_accuracy'], label='val accuracy')
    plt.legend()
    plt.title("Dokładność")
    plt.show()

    plt.plot(historia.history['loss'], label='train loss')
    plt.plot(historia.history['val_loss'], label='val loss')
    plt.legend()
    plt.title("Strata")
    plt.show()


# ============================================================
# MAIN
# ============================================================

def main():
    sciezka_danych = "dataset"

    input_shape = (128, 128, 3)
    liczba_klas = 3

    X, y = wczytaj_dane(sciezka_danych)

    model = zbuduj_model(input_shape, liczba_klas)

    skompiluj_model(model)

    model.summary()

    historia = trenuj_model(model, X, y)

    # zapis modelu
    model.save("model_owoce.keras")

    # wykresy
    rysuj_wykresy(historia)

    # test
    klasyfikuj_obraz(model, "test.jpg")


if __name__ == "__main__":
    main()