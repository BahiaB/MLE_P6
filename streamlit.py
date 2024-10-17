import streamlit as st
import requests
from PIL import Image
import io

# URL de l'API FastAPI
API_URL = "https://de67-35-237-118-34.ngrok-free.app/predict/"

def predict(image: Image.Image):
    # Convertir l'image en bytes
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_bytes = buffered.getvalue()

    # Envoyer l'image à l'API pour la prédiction
    response = requests.post(API_URL, files={"file": img_bytes})

    if response.status_code == 200:
        return response.json()["prediction"]
    else:
        # Afficher le message d'erreur complet
        st.error(f"Erreur lors de la prédiction : {response.status_code}, {response.text}")
        return None

# Titre de l'application
st.title("What's your dog's breed?")

# Télécharger une image
uploaded_file = st.file_uploader("Please upload your image here", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Afficher l'image téléchargée
    image = Image.open(uploaded_file)
    st.image(image, caption='Image téléchargée.', use_column_width=True)

    # Bouton de prédiction
    if st.button("Prédire"):
        st.write("Prédiction en cours...")
        prediction = predict(image)
        if prediction:
            st.write(f"Résultat de la prédiction: **{prediction}**")
