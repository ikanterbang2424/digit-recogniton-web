# Digit Recognition Web App using CNN and EMNIST

Aplikasi web untuk mengenali angka tulisan tangan (0–9) menggunakan **Convolutional Neural Network (CNN)** dan dataset **EMNIST Digits**. Pengguna dapat menggambar angka secara langsung pada canvas, kemudian model AI akan memprediksi angka tersebut dan menampilkan tingkat probabilitas prediksinya.

## Project Overview

Proyek ini dibuat untuk mempelajari penerapan Deep Learning dalam pengenalan citra digital serta mengimplementasikan model Machine Learning ke dalam aplikasi web interaktif.

Fitur utama:
- ✍️ Menggambar angka secara langsung pada canvas.
- 🤖 Mengenali angka 0 sampai 9 menggunakan model CNN.
- 📊 Menampilkan hasil prediksi dan probabilitas setiap kelas angka.
- 🧠 Melatih model menggunakan dataset EMNIST Digits.
- 🌐 Menjalankan model melalui antarmuka web berbasis Streamlit.

## Technologies Used

- **Python** — bahasa pemrograman utama.
- **TensorFlow / Keras** — membangun dan melatih model CNN.
- **EMNIST Digits** — dataset gambar angka tulisan tangan.
- **Streamlit** — membangun antarmuka web.
- **streamlit-drawable-canvas** — menyediakan area menggambar.
- **NumPy** — memproses data numerik dan gambar.
- **Pillow (PIL)** — memproses gambar sebelum prediksi.
- **TensorFlow Datasets (TFDS)** — memuat dataset EMNIST.

## Project Structure

```text
digit-recogniton-web/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── .gitignore
└── digit_model.keras
```

| File | Description |
|---|---|
| `app.py` | Menjalankan aplikasi web dan menampilkan hasil prediksi angka. |
| `train_model.py` | Memuat dataset EMNIST, melatih model CNN, mengevaluasi model, dan menyimpan hasil pelatihan. |
| `requirements.txt` | Daftar library Python yang dibutuhkan. |
| `README.md` | Dokumentasi proyek. |
| `.gitignore` | Menentukan file dan folder yang tidak perlu diunggah ke GitHub. |
| `digit_model.keras` | Model CNN yang telah dilatih dan digunakan untuk prediksi. |

## Installation

### 1. Clone Repository

Buka terminal, kemudian jalankan:

```bash
git clone https://github.com/ikanterbang2424/digit-recogniton-web.git
cd digit-recogniton-web
```

### 2. Create a Virtual Environment

Disarankan menggunakan virtual environment agar dependensi proyek terpisah dari proyek Python lainnya.

**Windows:**

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Pastikan versi library kompatibel dengan kode proyek yang digunakan.

## How to Run the Application

Pastikan file `digit_model.keras` tersedia di folder proyek. Kemudian jalankan:

```bash
streamlit run app.py
```

Streamlit akan menampilkan alamat lokal di terminal. Buka alamat tersebut di browser untuk menggunakan aplikasi.

## Train the CNN Model

Jika ingin melatih model dari awal, jalankan:

```bash
python train_model.py
```

Script akan memuat dataset EMNIST Digits, melakukan preprocessing, melatih model CNN, mengevaluasi performanya, dan menyimpan model hasil pelatihan sebagai `digit_model.keras`.

**Note:** Pelatihan membutuhkan koneksi internet saat dataset pertama kali diunduh, ruang penyimpanan yang cukup, serta waktu komputasi. Jika model yang sudah dilatih tersedia, kamu tidak perlu melatih ulang untuk menjalankan aplikasi.

## How It Works

1. **Data Loading:** Dataset EMNIST Digits dimuat untuk proses pelatihan.
2. **Preprocessing:** Gambar diubah ke format grayscale berukuran 28 × 28 piksel dan dinormalisasi.
3. **Model Training:** CNN mempelajari pola visual dari gambar angka.
4. **Model Saving:** Model yang telah dilatih disimpan dalam format `.keras`.
5. **User Input:** Pengguna menggambar angka pada canvas aplikasi.
6. **Image Processing:** Gambar canvas diproses agar sesuai dengan format input model.
7. **Prediction:** Model menghasilkan probabilitas untuk setiap angka dari 0 hingga 9.
8. **Result Display:** Aplikasi menampilkan angka dengan probabilitas prediksi tertinggi.

## Model Architecture

Model CNN menggunakan beberapa lapisan berikut:

- `Conv2D` — mengekstraksi fitur visual dari gambar.
- `MaxPooling2D` — mengurangi dimensi fitur.
- `Flatten` — mengubah fitur menjadi vektor.
- `Dense` — mempelajari pola untuk klasifikasi.
- `Dropout` — membantu mengurangi risiko overfitting.
- `Softmax` — menghasilkan probabilitas untuk 10 kelas angka.

## Project Objectives

- Memahami penerapan CNN untuk klasifikasi gambar.
- Menggunakan dataset EMNIST Digits untuk pengenalan angka tulisan tangan.
- Mengintegrasikan model Deep Learning dengan aplikasi web.
- Membuat aplikasi prediksi angka yang interaktif dan mudah digunakan.

## Future Improvements

- Meningkatkan akurasi melalui optimasi model dan preprocessing gambar.
- Menambahkan fitur untuk menyimpan riwayat prediksi.
- Mengembangkan tampilan antarmuka yang lebih responsif.
- Melakukan deployment agar aplikasi dapat diakses secara online.

## Author

**Gofran Muttaqi Likhan**

GitHub: [@ikanterbang2424](https://github.com/ikanterbang2424)

---

*This project is developed for educational purposes to explore image classification, deep learning, and interactive web application development.*
