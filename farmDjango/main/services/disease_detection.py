import os
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image, ImageEnhance
from rembg import remove
import io
import numpy as np
from django.conf import settings

class DiseaseDetector:
    def __init__(self):
        self.model = self._load_model()
        self.class_names = [
            'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
            'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
            'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
            'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
            'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
            'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
            'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight',
            'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
            'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight',
            'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
            'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
            'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
        ]

    def _load_model(self):
        """Charge le modèle de prédiction"""
        model_path = os.path.join(settings.BASE_DIR, 'main', 'models', 'model_modeleDDDD.keras')
        return tf.keras.models.load_model(model_path)
    
    def remove_background(self, img_path):
        """Enlève l'arrière-plan de l'image à l'aide de 'rembg'"""
        try:
            with open(img_path, "rb") as f:
                input_image = f.read()
            output_image_data = remove(input_image)

            img = Image.open(io.BytesIO(output_image_data)).convert("RGB")
            img.show()  # Afficher l'image pour vérifier
            img = img.resize((224, 224))  # Redimensionner à la taille attendue par le modèle
            return img
        except Exception as e:
            print(f"Erreur lors de la suppression du fond : {e}")
            return None

    def preprocess_image(self, img_path, input_size=(128, 128)):
        """Prétraitement de l'image avant la prédiction"""
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image non trouvée : {img_path}")

        img = self.remove_background(img_path)
        if img is None:
            return None  # Si l'image n'a pas pu être traitée

        img = img.resize(input_size)
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(1.1)

        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        return preprocess_input(img_array)
    



    def predict(self, img_path, plant_name=None):
        """Prédit la classe de l'image et retourne les résultats"""
        try:
            # Prétraiter l'image
            img_array = self.preprocess_image(img_path)
            if img_array is None:
                return None, "Erreur lors du prétraitement de l'image"

            # Prédiction
            prediction = self.model.predict(img_array)[0]
            results = [(name, float(prediction[i])) for i, name in enumerate(self.class_names)]
            results.sort(key=lambda x: x[1], reverse=True)

            # Filtrer par nom de plante si fourni
            if plant_name:
                plant_classes = [(i, name) for i, name in enumerate(self.class_names) 
                                 if plant_name.lower() in name.lower()]
                if not plant_classes:
                    return None, "Aucun nom de classe ne correspond à cette plante"
                
                filtered = [(name, float(prediction[i])) for i, name in plant_classes]
                filtered.sort(key=lambda x: x[1], reverse=True)
                return filtered, None
            else:
                return results, None

        except Exception as e:
            return None, str(e)
