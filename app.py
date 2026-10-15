
import streamlit as st
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf

# Konfigurasi halaman
st.set_page_config(
    page_title="Digit Recognition",
    page_icon="🔢",
    layout="centered"
)

# Memuat model yang sudah dilatih
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("digit_model.keras")

model = load_model()

# Tampilan utama
st.title("🔢 Digit Recognition")
st.write(
    "Gambarkan angka dari 0 sampai 9, "
    "lalu biarkan AI mengenalinya!"
)

st.info(
    "Gunakan mouse untuk menggambar angka "
    "berwarna putih di atas kanvas hitam."
)

# Kanvas untuk menggambar
canvas_result = st_canvas(
    fill_color="rgba(255, 255, 255, 1)",
    stroke_width=18,
    stroke_color="#FFFFFF",
    background_color="#000000",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="digit_canvas",
    display_toolbar=True
)

# Tombol prediksi
if st.button("🔍 Prediksi Angka", type="primary"):

    if canvas_result.image_data is None:
        st.warning("Silakan gambar angka terlebih dahulu.")

    elif not np.any(canvas_result.image_data[:, :, :3] > 0):
        st.warning("Kanvas masih kosong. Gambar angka dahulu.")

    else:
        # Mengubah gambar RGBA menjadi grayscale
        image = Image.fromarray(
            canvas_result.image_data.astype("uint8")
        ).convert("L")

        # Mengubah ukuran menjadi 28 x 28 piksel
        image = image.resize(
            (28, 28),
            Image.Resampling.LANCZOS
        )

        # Mengubah gambar menjadi array
        image_array = np.array(
            image,
            dtype="float32"
        ) / 255.0

        # Menyesuaikan bentuk input model
        image_array = image_array.reshape(1, 28, 28, 1)

        # Melakukan prediksi
        predictions = model.predict(
            image_array,
            verbose=0
        )[0]

        predicted_digit = int(np.argmax(predictions))
        confidence = float(np.max(predictions)) * 100

        # Menampilkan hasil
        st.success("Prediksi berhasil!")

        st.metric(
            label="Angka yang dikenali",
            value=str(predicted_digit)
        )

        st.write(
            f"Tingkat keyakinan model: {confidence:.2f}%"
        )

        # Menampilkan probabilitas semua angka
        st.subheader("Probabilitas setiap angka")

        st.bar_chart({
            str(i): float(predictions[i])
            for i in range(10)
        })

st.caption(
    "Model dilatih menggunakan MNIST Dataset."
)
