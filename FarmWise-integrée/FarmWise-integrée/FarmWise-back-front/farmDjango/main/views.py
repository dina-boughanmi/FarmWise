# main/views.py
from django.shortcuts import render
from xhtml2pdf import pisa
from io import BytesIO
from django.template.loader import get_template
from django.http import HttpResponse
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
import tensorflow as tf
import os
from django.conf import settings
from PIL import Image
import numpy as np
from io import BytesIO
import pandas as pd 
import joblib
from . import preprocessing_rot_data
from . import volunteer
from sklearn.preprocessing import StandardScaler, LabelEncoder
import json
import torch 
import torch.nn as nn
import torch.nn.functional as F
from .gcn_model import GCNNet
from django.core.files.storage import FileSystemStorage
from django.core.files.base import ContentFile
from tensorflow.keras.preprocessing import image
from django.views.decorators.csrf import csrf_exempt
from .services.disease_detection import DiseaseDetector



def index(request):
    return render(request, 'main/index.html')
def about_view(request):
    return render(request, 'main/about.html')
def login(request):
    return render(request, 'main/login.html')
def signup_view(request):  # Renamed from 'signUp'
    return render(request, 'main/signUp.html')

def contact_view(request):  
    return render(request, 'main/contact.html')
@login_required
def services_view(request):  
    return render(request, 'main/service.html')
@login_required

def fertilizer_view(request):  
    return render(request, 'main/dosage.html')
@login_required

def land_view(request):  
    return render(request, 'main/landPrice.html')
@login_required


#--------------------------------------- DiseaseDetector ------------------------------------------------#
def disease_view(request):  
    detector = DiseaseDetector()
    context = {}
    
    if request.method == 'POST' and request.FILES.get('image'):
        uploaded_file = request.FILES['image']
        fs = FileSystemStorage()
        
        # Sauvegarder temporairement le fichier
        filename = fs.save(uploaded_file.name, uploaded_file)
        filepath = os.path.join(settings.MEDIA_ROOT, filename)
        
        plant_name = request.POST.get('plant_name', '').strip()
        predictions, error = detector.predict(filepath, plant_name if plant_name else None)
        
        # Supprimer le fichier après traitement
        if os.path.exists(filepath):
            os.remove(filepath)
        
        if error:
            context['error'] = error
        else:
            context['predictions'] = predictions
            context['plant_name'] = plant_name
    
    return render(request, 'main/disease.html', context)

#----------------------------------------------- Crop Rotation --------------------------------------------#

# Load model (keep this at module level to load only once)
MODEL_PATH_ROT = os.path.join(settings.BASE_DIR, 'main', 'models', 'hybrid_model.pkl')
model_rot= joblib.load(MODEL_PATH_ROT)

# Standardize 
SCALER_PATH = os.path.join(settings.BASE_DIR, 'main', 'encoders', 'standard_scaler_rot.pkl')
scaler= joblib.load(SCALER_PATH)

#label_encoder
LABEL_ENCODER_PATH = os.path.join(settings.BASE_DIR, 'main', 'encoders', 'seq_label_encoders.pkl')
seq_encoders= joblib.load(LABEL_ENCODER_PATH)

# Load GCN model
MODEL_PATH_GCN = os.path.join(settings.BASE_DIR, 'main', 'models', 'gcn_model.pth')
input_dim=38
gcn_model= GCNNet(in_dim=input_dim)
gcn_model.load_state_dict(torch.load(MODEL_PATH_GCN))
gcn_model.eval()


@login_required
def cropRotation_view(request):  
    #read the csv file 
    ROT_DATA = os.path.join(settings.BASE_DIR, 'main', 'data', 'merged_dataset.csv')
    df= pd.read_csv(ROT_DATA, sep=',', encoding='latin1')
    
    
    if request.method == 'GET':
        context = {
            'crop1_list': df['Crop (t-2)'].dropna().str.strip().unique(),
            'crop2_list': df['Crop (t-1)'].dropna().str.strip().unique(),
            'crop3_list': df['Crop (t)'].dropna().str.strip().unique(),
            'current_crop_list': df['Crop'].dropna().str.strip().unique()
        }
        return render(request, 'main/cropRotation.html', context)
    
    if request.method == 'POST':
        data = json.loads(request.body.decode('utf-8'))
        crop1 = data.get('crop1')
        crop2 = data.get('crop2')
        crop3 = data.get('crop3')
        curent_crop = data.get('crop4')
        
        
        
        # Define your column names
        sequential_cols = ['Crop (t-2)', 'Crop (t-1)', 'Crop (t)']
        static_cat_cols = ['Soil_type', 'soil salinity', 'area', 'Disease', 'Pest']
        numerical_cols = [
            'Etm', 'Water Requirements', 'days_of_water_usage',
            'Fertilizer N', 'Fertilizer K', 'Fertilizer P',
            'Root Depth', 'Residue', 'plant spacing', 'Total_Fertilizer_Index'
        ]

        
        crop_sequence=[crop1,crop2,crop3]
        crop_info=preprocessing_rot_data.adding_static_data(df,crop_sequence,curent_crop)
        if crop_info is None:
            return JsonResponse({'success': False, 'error': 'Current crop not found in dataset.'}, status=400)
        
        # Standardize 
        df_static_cat = pd.get_dummies(df[static_cat_cols], drop_first=True)
        static_input_encoding =preprocessing_rot_data.encode_static(crop_info,scaler,df_static_cat) 
        
        
        #label_encoder
        crop_sequence_encoded =preprocessing_rot_data.encode_sequential(crop_sequence,seq_encoders)


        # Load GCN model
        gcn_input=preprocessing_rot_data.encode_gcn(static_input_encoding,gcn_model)
        
        #Target_encoder
        TARGET_ENCODER_PATH = os.path.join(settings.BASE_DIR, 'main', 'encoders', 'target_label_encoder.pkl')
        target_le= joblib.load(TARGET_ENCODER_PATH)
        preds=preprocessing_rot_data.predict_next(crop_sequence_encoded,gcn_input,target_le,model_rot)


        return JsonResponse(
                {
                'success': True,
                'crop_sequence': preds,
                'current_crop':curent_crop
                } 
                )
    return JsonResponse({'success': False, 'error': 'Invalid request'}, status=400)


#------------------------------------------------Volunteer Plant ----------------------------------------------------#   

# Load model (keep this at module level to load only once)
MODEL_PATH_VOL = os.path.join(settings.BASE_DIR, 'main', 'models', 'xception_model_50.keras')
model_vol = tf.keras.models.load_model(MODEL_PATH_VOL)

class_labels=['Allium', 'Borage', 'Calendula', 'Cattail', 'Chickweed', 'Coltsfoot', 
'Common_Mallow', 'Common_Yarrow', 'Dendelion', 'Dodder', 'Field Bindweed', 
'Giant Reed', 'Henbit', 'Johnson Grass', 'Nutsedge', 'Spiny Amaranth', 
'Wild Mustard', 'cow parsley']

@login_required
@csrf_exempt 
def voluntary_view(request):
    
    #read the csv file 
    ROT_DATA = os.path.join(settings.BASE_DIR, 'main', 'data', 'VolunteerPlant.csv')
    df= pd.read_csv(ROT_DATA, sep=',')
    
    if request.method == 'GET':
        return render(request, 'main/voluntary.html')
    
    if request.method == 'POST' and request.FILES.get('volunteerImage'):
        try:
            image_file = request.FILES['volunteerImage']
            
            # heck if the file is a JPG
            if not image_file.name.lower().endswith('.jpg'):
                return JsonResponse({
                    'success': False,
                    'error': 'Only .jpg images are allowed.'
                }, status=400)

            # Resize the image
            img = Image.open(image_file).convert('RGB')
            img = img.resize((299, 299))

            # Save resized image to memory
            img_io = BytesIO()
            img.save(img_io, format='JPEG')
            img_content = ContentFile(img_io.getvalue(), name=image_file.name)

            # Save image to media folder
            fs = FileSystemStorage(location=os.path.join(settings.MEDIA_ROOT, 'volunteers'))
            filename = fs.save(image_file.name, img_content)
            file_url = fs.url('volunteers/' + filename)

            # Preprocess for model (e.g., Xception)
            img_array = image.img_to_array(img)  
            img_array = np.expand_dims(img_array, axis=0) 
            img_array = img_array / 255.0  
            
            #prediction
            predictions = model_vol.predict(img_array)
            predicted_class = np.argmax(predictions)
            plant_name=class_labels[predicted_class] 
            
            #metadata
            column=volunteer.get_matching_column(df,plant_name)
            metadata=volunteer.metadata(df,plant_name,column)
            solution_text=metadata['Solution ']
            solution_steps = [step.strip() for step in solution_text.split('/') if step.strip()]
            print(solution_steps)
            
            return JsonResponse({
                'success': True,
                'image_url': file_url,
                'scientific_name' : metadata['ScientificName'],
                'Common_name': metadata['Common name'],
                'Type': metadata['Type'],
                'Description': metadata['Description '],
                'Solution': solution_steps       
                })
        except Exception as e:
            return JsonResponse({
                'success': False,
                'error': str(e)
            }, status=400)
    
    return JsonResponse({'success': False, 'error': 'Invalid request'}, status=400)



#---------------------------------------------------------------------------------------------------------#

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import pandas as pd
import joblib
import os
import json
from django.conf import settings
from django.shortcuts import render

# Load the trained stacking model
model2 = joblib.load(os.path.join(settings.BASE_DIR, 'main', 'stacking_cat_hgb_gb_model (1).joblib'))
model_landprice = joblib.load(os.path.join(settings.BASE_DIR, 'main', 'catboost_model.pkl'))
# Load LabelEncoders
encoder_dir = os.path.join(settings.BASE_DIR, 'main')

encoders = {
    'State': joblib.load(os.path.join(encoder_dir, 'State_encoder.joblib')),
    'Specific Crop': joblib.load(os.path.join(encoder_dir, 'Specific Crop_encoder.joblib')),
    'Fertilizer Type': joblib.load(os.path.join(encoder_dir, 'Fertilizer Type_encoder.joblib')),
    'Fertilizer Name': joblib.load(os.path.join(encoder_dir, 'Fertilizer Name_encoder.joblib')),
    'Application Method': joblib.load(os.path.join(encoder_dir, 'Application Method_encoder.joblib')),
    'Region of Use in Tunisia': joblib.load(os.path.join(encoder_dir, 'Region of Use in Tunisia_encoder.joblib')),
}

# Load JSON mappings
fertilizers_data = json.load(open(os.path.join(settings.BASE_DIR, 'main', 'data', 'fertilizers.json')))
crops_data = json.load(open(os.path.join(settings.BASE_DIR, 'main', 'data', 'crop.json')))
states_data = json.load(open(os.path.join(settings.BASE_DIR, 'main', 'data', 'state.json')))



# List of columns to encode
label_cols = list(encoders.keys())

# Expected column order based on the training feature order
expected_order = [
    'N (%)', 'P (%)', 'K (%)', 'Fertilizer Type', 'Absorption Rate (%)',
    'Price per kg (TND)', 'State', 'Region of Use in Tunisia', 'Specific Crop',
    'Crop N', 'Crop P', 'Crop K', 'Application Method', 'Root Depth (cm)',
    'Soil_P', 'Soil_K', 'Soil_N', 'pH', 'NPK_ratio', 'N_gap', 'P_gap', 'K_gap',
    'Soil_P_suff', 'Soil_K_suff', 'Soil_N_suff', 'Effective_N', 'Effective_P', 'Effective_K'
]

def autofill_fields(input_data):
    # Autofill from Fertilizer Name
    fertilizer_info = next((item for item in fertilizers_data if item["Fertilizer Name"] == input_data["Fertilizer Name"]), None)
    if fertilizer_info:
        input_data["N (%)"] = fertilizer_info.get("N (%)")
        input_data["P (%)"] = fertilizer_info.get("P (%)")
        input_data["K (%)"] = fertilizer_info.get("K (%)")
        input_data["Fertilizer Type"] = fertilizer_info.get("Fertilizer Type")
        input_data["Application Method"] = fertilizer_info.get("Application Method")
    else:
        raise ValueError(f"Fertilizer Name '{input_data['Fertilizer Name']}' not found.")

    # Autofill from State
    state_info = next((item for item in states_data if item["State"] == input_data["State"]), None)
    if state_info:
        input_data["Region of Use in Tunisia"] = state_info.get("Region of Use in Tunisia")
    else:
        raise ValueError(f"State '{input_data['State']}' not found.")

    # Autofill from Specific Crop
    crop_info = next((item for item in crops_data if item["Specific Crop"] == input_data["Specific Crop"]), None)
    if crop_info:
        input_data["Crop N"] = crop_info.get("Crop N")
        input_data["Crop P"] = crop_info.get("Crop P")
        input_data["Crop K"] = crop_info.get("Crop K")
        input_data["Root Depth (cm)"] = crop_info.get("Root Depth (cm)")
    else:
        raise ValueError(f"Specific Crop '{input_data['Specific Crop']}' not found.")

    return input_data

def preprocess_input(samples):
    df = pd.DataFrame(samples)

    # Encode label columns
    for col in label_cols:
        if col in df.columns:
            encoder = encoders[col]
            try:
                df[col] = encoder.transform(df[col])
            except ValueError as e:
                raise ValueError(f"Invalid value for column '{col}': {e}")

    # Feature engineering
    df["NPK_ratio"] = df["N (%)"] / (df["P (%)"] + df["K (%)"] + 1e-3)
    df["N_gap"] = df["Crop N"] - df["N (%)"]
    df["P_gap"] = df["Crop P"] - df["P (%)"]
    df["K_gap"] = df["Crop K"] - df["K (%)"]
    df["Soil_P_suff"] = df["Soil_P"] / (df["Crop P"] + 1e-3)
    df["Soil_K_suff"] = df["Soil_K"] / (df["Crop K"] + 1e-3)
    df["Soil_N_suff"] = df["Soil_N"] / (df["Crop N"] + 1e-3)
    df["Effective_N"] = df["N (%)"] * df["Absorption Rate (%)"] / 100
    df["Effective_P"] = df["P (%)"] * df["Absorption Rate (%)"] / 100
    df["Effective_K"] = df["K (%)"] * df["Absorption Rate (%)"] / 100

    # Drop unnecessary columns
    df = df.drop(columns=["Fertilizer Name", "Usage (Dosage)"], errors="ignore")

    # Reorder the columns
    df = df[expected_order]

    return df

# Your original API endpoint
class Salma(APIView):
    def post(self, request):
        try:
            input_data = request.data.get('features')
            if not input_data:
                return Response({'error': 'Missing input features'}, status=status.HTTP_400_BAD_REQUEST)

            # Auto-fill missing fields
            input_data = autofill_fields(input_data)

            # Preprocess the input
            input_df = preprocess_input([input_data])

            # Make prediction
            y_pred = model2.predict(input_df).flatten()[0]

            return Response({
                'prediction': round(float(y_pred), 2)
            })

        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)

# New view for HTML form
def load_json_file(filename):
    path = os.path.join(settings.BASE_DIR, 'main', 'static', 'json', filename)  # adjust if needed
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)
@login_required
def predict_view(request):
    prediction = None
    error = None
    form_data = {}

    fertilizers_data = json.load(open(os.path.join(settings.BASE_DIR, 'main', 'data', 'fertilizers.json')))
    crops_data = json.load(open(os.path.join(settings.BASE_DIR, 'main', 'data', 'crop.json')))
    states_data = json.load(open(os.path.join(settings.BASE_DIR, 'main', 'data', 'state.json')))

    fertilizer_list = [item['Fertilizer Name'] for item in fertilizers_data]
    state_list = [item['State'] for item in states_data]
    crop_list = [item['Specific Crop'] for item in crops_data]

    if request.method == 'POST':
        try:
            form_data = {
                "Fertilizer_Name": request.POST.get('Fertilizer Name'),
                "State": request.POST.get('State'),
                "Specific_Crop": request.POST.get('Specific Crop'),
                "Soil_N": request.POST.get('Soil_N'),
                "Soil_P": request.POST.get('Soil_P'),
                "Soil_K": request.POST.get('Soil_K'),
                "pH": request.POST.get('pH'),
                "Price_per_kg_TND": request.POST.get('Price per kg (TND)'),
                "Absorption_Rate": request.POST.get('Absorption Rate (%)'),
            }

            input_data = {
                "Fertilizer Name": form_data["Fertilizer_Name"],
                "State": form_data["State"],
                "Specific Crop": form_data["Specific_Crop"],
                "Soil_N": float(form_data["Soil_N"]),
                "Soil_P": float(form_data["Soil_P"]),
                "Soil_K": float(form_data["Soil_K"]),
                "pH": float(form_data["pH"]),
                "Price per kg (TND)": float(form_data["Price_per_kg_TND"]),
                "Absorption Rate (%)": float(form_data["Absorption_Rate"]),
            }

            input_data = autofill_fields(input_data)
            input_df = preprocess_input([input_data])
            y_pred = model2.predict(input_df).flatten()[0]
            prediction = round(float(y_pred), 2)

        except Exception as e:
            error = str(e)

        request.session['last_full_input'] = input_data  # Add this
        request.session['last_form_data'] = form_data
        request.session['last_prediction'] = prediction



    return render(request, 'main/dosage.html', {
        'prediction': prediction,
        'error': error,
        'fertilizer_list': fertilizer_list,
        'state_list': state_list,
        'crop_list': crop_list,
        'form_data': form_data,
    })


def download_pdf(request):
    if request.method == 'POST':
        context = {
            'form_data': request.session.get('last_form_data'),
            'full_input': request.session.get('last_full_input'),
            'prediction': request.session.get('last_prediction'),
        }

        template = get_template('main/pdf_template.html')
        html = template.render(context)

        response = HttpResponse(content_type='application/pdf')
        response['Content-Disposition'] = 'attachment; filename="fertilizer_report.pdf"'
        pisa.CreatePDF(BytesIO(html.encode('utf-8')), dest=response)
        return response

    return HttpResponse("Invalid request.")


import openai
from django.shortcuts import render
from django.http import JsonResponse
import json
import os

openai.api_key = "your_openai_api_key"  # Store securely in environment variables

def agri_chatbot(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        user_message = data.get('message', '')

        # GPT prompt limited to agriculture domain
        prompt = f"You are an expert in agriculture. Answer the following question in a helpful way:\n\nUser: {user_message}\nBot:"

        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an agriculture expert who provides helpful information."},
                    {"role": "user", "content": user_message}
                ]
            )
            reply = response.choices[0].message.content.strip()
        except Exception as e:
            reply = "Sorry, I'm having trouble processing your request."

        return JsonResponse({'reply': reply})
    else:
        return render(request, 'main/chat.html')
    
    # Charger les fichiers d'encodage
encoding_files = {
    "Emplacement": pd.read_excel("main/data/Emplacements.xlsx"),
    "CLAF": pd.read_excel("main/data/CLAF.xlsx"),
    "State": pd.read_excel("main/data/states.xlsx"),
    "PSCL": pd.read_excel("main/data/PSCL.xlsx"),
    "LNDF": pd.read_excel("main/data/LNDF.xlsx"),
    "Drain": pd.read_excel("main/data/Drain.xlsx"),  # <-- Ajout
}

import logging

logger = logging.getLogger(__name__)


# Vue API pour la prédiction
class PredictPrice(APIView):
    def post(self, request):
        data = request.data
        try:
            # Affichage des données reçues pour le débogage
            logger.info(f"Données reçues : {data}")
            
            # Extraire les variables numériques simples
            surface = float(data["Surface"])
            tawc = float(data["TAWC"])
            phaq = float(data["PHAQ"])

            # Fonction d'encodage avec log de débogage
            def encode_value(col_name, raw_value):
                df = encoding_files[col_name]
                encoded_col = f"{col_name}_encoded"
                encoded = df[df[col_name] == raw_value][encoded_col]
                if not encoded.empty:
                    logger.info(f"{col_name} encodé pour {raw_value} : {encoded.values[0]}")
                    return encoded.values[0]
                logger.error(f"Valeur d'encodage non trouvée pour {col_name} : {raw_value}")
                return None

            # Encodage des variables catégorielles
            emplacement_encoded = encode_value("Emplacement", data["Emplacement"])
            claf_encoded = encode_value("CLAF", data["CLAF"])
            state_encoded = encode_value("State", data["State"])
            pscl_encoded = encode_value("PSCL", data["PSCL"])
            lndf_encoded = encode_value("LNDF", data["LNDF"])
            drain_encoded = encode_value("Drain", data["Drain"])

            if None in [emplacement_encoded, claf_encoded, state_encoded, pscl_encoded, lndf_encoded, drain_encoded]:
                logger.error("Encodage invalide pour au moins une variable.")
                return Response({"error": "Encodage invalide pour au moins une variable."}, status=400)

            # Créer le DataFrame d'entrée
            input_df = pd.DataFrame([{
                "Surface": surface,
                "Emplacement_encoded": emplacement_encoded,
                "State_encoded": state_encoded,
                "LNDF": lndf_encoded,
                "CLAF_encoded": claf_encoded,
                "Drain": drain_encoded,
                "PSCL_encoded": pscl_encoded,
                "TAWC": tawc,
                "PHAQ": phaq,
            }])

            # Prédire
            prediction = model_landprice.predict(input_df)[0]
            logger.info(f"Prédiction effectuée : {prediction}")
            return Response({"predicted_price": round(prediction, 2)})

        except Exception as e:
            logger.error(f"Erreur lors du traitement de la requête : {str(e)}")
            return Response({"error": str(e)}, status=500)




#lyndaa
from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
import numpy as np
import json
import os
import joblib
from tensorflow.keras.models import load_model

# Chargement des modèles

# Obtenir le chemin du répertoire courant (celui de views.py)
APP_DIR = os.path.dirname(os.path.abspath(__file__))

# Construire les chemins vers les fichiers
path_lstm = os.path.join(APP_DIR, 'modele_lstm.h5')
path_xgb = os.path.join(APP_DIR, 'xgboost_model_days.pkl')
path_preprocessor = os.path.join(APP_DIR, 'preprocessor.pkl')

# Chargement des modèles
model_lstm = load_model(path_lstm, custom_objects={'mse': 'mse'})
model_xgb = joblib.load(path_xgb)
preprocessor = joblib.load(path_preprocessor)

@api_view(['GET', 'POST'])
def yield_view(request):
    if request.method == 'POST':
        try:
            # Récupération des données
            input_data = request.data.get('features')
            model_type = request.data.get('model_type')

            if not input_data:
                return Response({'error': 'Données manquantes'}, status=status.HTTP_400_BAD_REQUEST)

            input_df = pd.DataFrame([input_data])

            if model_type == 'lstm':
                # Prétraitement LSTM
                X_processed = preprocessor.transform(input_df).toarray()
                X_lstm = X_processed.reshape((X_processed.shape[0], 1, X_processed.shape[1]))
                y_pred = model_lstm.predict(X_lstm).flatten()[0]
            elif model_type == 'xgb':
                # Prédiction XGBoost
                y_pred = model_xgb.predict(input_df).flatten()[0]
            else:
                return Response({'error': 'Type de modèle invalide'}, status=status.HTTP_400_BAD_REQUEST)

            return Response({'prediction': round(float(y_pred), 2)})

        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)

    # GET request - affichage du formulaire
    return render(request, 'main/yield.html')

