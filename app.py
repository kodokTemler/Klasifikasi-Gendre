import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Judul Aplikasi
st.title('Prediksi Genre Film')

# Upload File
uploaded_file = "datasetfilm.xlsx"

if uploaded_file is not None:
    # Langkah 1: Memasukkan Dataset
    data = pd.read_excel(uploaded_file)

    # Langkah 2: Pre-Processing
    features = ['Duration', 'IMDB Rating', 'Metascore']  # Fitur yang digunakan
    target = 'Genre'  # Kolom target

    # Menghapus nilai kosong pada data
    data = data.dropna(subset=features + [target])

    # Simpan kategori awal sebelum mengonversi target menjadi numerik
    categories = data[target].astype('category').cat.categories
    data[target] = data[target].astype('category').cat.codes

    # Split data menjadi training dan testing
    X_train, X_test, y_train, y_test = train_test_split(data[features], data[target], test_size=0.2, random_state=42)

    # Langkah 3: Training Model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Simpan model menggunakan pickle
    pickle.dump(model, open('model.pkl', 'wb'))
    pickle.dump(categories, open('categories.pkl', 'wb'))

    # Menampilkan informasi bahwa model sudah dilatih
    st.write("Model telah dilatih dan siap digunakan!")

# Form input untuk prediksi
st.header('Masukkan Fitur Film')

# Form input dari user
duration = st.number_input('Durasi Film (menit)', min_value=0)
imdb_rating = st.number_input('Rating IMDB', min_value=0.0, max_value=10.0)
metascore = st.number_input('Metascore', min_value=0, max_value=100)

# Tombol prediksi
if st.button('Prediksi Genre'):
    # Memuat model dan kategori yang sudah disimpan
    model = pickle.load(open('model.pkl', 'rb'))
    categories = pickle.load(open('categories.pkl', 'rb'))

    # Membuat input untuk prediksi
    input_data = pd.DataFrame({
        'Duration': [duration],
        'IMDB Rating': [imdb_rating],
        'Metascore': [metascore]
    })

    # Prediksi genre film
    genre_code = model.predict(input_data)[0]
    predicted_genre = categories[genre_code]

    # Menampilkan hasil prediksi
    st.write(f"Prediksi Genre Film: {predicted_genre}")
