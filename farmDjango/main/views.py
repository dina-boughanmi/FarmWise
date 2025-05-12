# main/views.py
from django.shortcuts import render
from xhtml2pdf import pisa
from io import BytesIO
from django.template.loader import get_template
from django.http import HttpResponse
from django.contrib.auth.decorators import login_required


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

def yield_view(request):  
    return render(request, 'main/yield.html')
@login_required

def disease_view(request):  
    return render(request, 'main/disease.html')
@login_required

def voluntary_view(request):  
    return render(request, 'main/voluntary.html')





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
