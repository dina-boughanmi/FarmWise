# main/views.py
from django.shortcuts import render

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
def services_view(request):  
    return render(request, 'main/service.html')
def fertilizer_view(request):  
    return render(request, 'main/dosage.html')
def land_view(request):  
    return render(request, 'main/landPrice.html')
def disease_view(request):  
    return render(request, 'main/disease.html')
def voluntary_view(request):  
    return render(request, 'main/voluntary.html')



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
