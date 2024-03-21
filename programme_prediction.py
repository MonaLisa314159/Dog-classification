import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
import pickle
import numpy as np

model = load_model("Models/EfficientNetV2M_2_lrDecay.h5")

file_path = "Models/class_names.pkl"
with open(file_path, "rb") as f:
    class_names = pickle.load(f)
    
#st.write("Hello")
# Afficher le formulaire de téléchargement de l'image
st.title("Détection de la race d'un chien sur une photo")
uploaded_file = st.file_uploader("Téléchargez une image de chien", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
	
	#Afficher l'image téléchargée
	image = Image.open(uploaded_file)
	st.image(image, caption="Image téléchargée")
	
	# Prétraiter l'image pour la prédiction
	resized_image = image.resize((300, 300))
	image_array = img_to_array(resized_image)
	image_array = np.expand_dims(image_array, axis=0)
	image_preprocess = preprocess_input(image_array)
	
	# Prédire la race de chien
	predictions = model.predict(image_preprocess)
	predicted_label = np.argmax(predictions, axis=1)[0]
	
	# Afficher la race prédite
	st.write("Race prédite : ", class_names[predicted_label])
