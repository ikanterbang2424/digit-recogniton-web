
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt

# 1. KONFIGURASI
SEED = 42
BATCH_SIZE = 128
EPOCHS = 5
MODEL_PATH = "digit_model.keras"

tf.keras.utils.set_random_seed(SEED)

# 2. MEMUAT EMNIST DIGITS
print("Memuat dataset EMNIST Digits...")

(train_raw, val_raw, test_raw), info = tfds.load(
    "emnist/digits",
    split=[
        "train[:90%]",
        "train[90%:]",
        "test"
    ],
    as_supervised=True,
    with_info=True
)

print("Jumlah data training:", info.splits["train"].num_examples * 0.9)
print("Jumlah data validation:", info.splits["train"].num_examples * 0.1)
print("Jumlah data testing:", info.splits["test"].num_examples)

# 3. PREPROCESSING
def preprocess(image, label):
    # Koreksi orientasi gambar EMNIST
    image = tf.transpose(image, perm=[1, 0, 2])

    # Normalisasi piksel dari 0-255 menjadi 0-1
    image = tf.cast(image, tf.float32) / 255.0

    # Label angka 0-9
    label = tf.cast(label, tf.int64)

    return image, label


print("Melakukan preprocessing dataset...")

train_ds = train_raw.map(
    preprocess,
    num_parallel_calls=tf.data.AUTOTUNE
)

val_ds = val_raw.map(
    preprocess,
    num_parallel_calls=tf.data.AUTOTUNE
)

test_ds = test_raw.map(
    preprocess,
    num_parallel_calls=tf.data.AUTOTUNE
)

# Acak data training dan buat batch
train_ds = (
    train_ds
    .shuffle(20000, seed=SEED)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

val_ds = (
    val_ds
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

test_ds = (
    test_ds
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

# 4. MEMERIKSA HASIL PREPROCESSING
images, labels = next(iter(train_ds))

plt.figure(figsize=(10, 4))

for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(images[i, :, :, 0], cmap="gray")
    plt.title(f"Label: {labels[i].numpy()}")
    plt.axis("off")

plt.suptitle("Contoh EMNIST setelah koreksi orientasi")
plt.tight_layout()
plt.show()

# 5. MEMBANGUN MODEL CNN
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)),

    # Convolution layer pertama
    tf.keras.layers.Conv2D(
        32,
        kernel_size=(3, 3),
        activation="relu"
    ),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

    # Convolution layer kedua
    tf.keras.layers.Conv2D(
        64,
        kernel_size=(3, 3),
        activation="relu"
    ),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

    # Flatten feature map
    tf.keras.layers.Flatten(),

    # Fully connected layer
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.3),

    # Output 10 kelas angka
    tf.keras.layers.Dense(10, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# 6. MELATIH MODEL
print("\nMemulai training CNN...")

history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=val_ds
)

# 7. EVALUASI AKHIR PADA DATA TESTING
print("\nEvaluasi model menggunakan data testing...")

test_loss, test_accuracy = model.evaluate(
    test_ds,
    verbose=1
)

print(f"\nTest loss     : {test_loss:.4f}")
print(f"Test accuracy : {test_accuracy * 100:.2f}%")

# 8. GRAFIK AKURASI DAN LOSS
plt.figure(figsize=(12, 4))

# Grafik akurasi
plt.subplot(1, 2, 1)
plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)
plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)
plt.title("Akurasi Model CNN")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

# Grafik loss
plt.subplot(1, 2, 2)
plt.plot(
    history.history["loss"],
    label="Training Loss"
)
plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)
plt.title("Loss Model CNN")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# 9. MENYIMPAN MODEL
model.save(MODEL_PATH)

print(f"\nModel berhasil disimpan: {MODEL_PATH}")
print("Model siap digunakan oleh aplikasi web.")
