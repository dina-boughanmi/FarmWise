# main/views.py
from django.shortcuts import render

from django.core.files.storage import FileSystemStorage
from django.conf import settings
from .services.disease_detection import DiseaseDetector
import os

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
def yield_view(request):  
    return render(request, 'main/yield.html')

def voluntary_view(request):  
    return render(request, 'main/voluntary.html')


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

