# main/urls.py
from django.urls import path
from . import views
from . import auth_views  # import your new auth file
from .views import PredictPrice,land_view  # Importer correctement la vue


urlpatterns = [
    path('', views.index, name='index'),
    path('about/', views.about_view, name='about'),
    #path('login/', views.login, name='login'),
    #path('signup/', views.signup_view, name='signup'),  
    path('contact/', views.contact_view, name='contact'), 
    path('service/', views.services_view, name='service'), 
    # path('fertilizer/', views.fertilizer_view, name='fertilizer'), 
    path('landPrice/', views.land_view, name='landPrice'),         # Affichage HTML
    path('api/landPrice/', PredictPrice.as_view(), name='predict_land_price'),  # API de p
    path('yield/', views.yield_view, name='yield'), 
    path('disease/', views.disease_view, name='disease'),
    path('cropRotation/', views.cropRotation_view, name='cropRotation'), 
    path('voluntary/', views.voluntary_view, name='voluntary'), 
    path('predict/', views.predict_view, name='fertilizer'),    # new form URL
    path('download-pdf/', views.download_pdf, name='download_pdf'),
    path('signup/', auth_views.signup_view, name='signup'),
    path('login/', auth_views.login_view, name='login'),
    path('logout/', auth_views.logout_view, name='logout'),
    path('chatbot/', views.agri_chatbot, name='agri_chatbot'),
    path('profile/', auth_views.profile_view, name='profile'),  # optional, see below
    path('payment/', auth_views.payment_view, name='payment'),  # ✅ Add this line








]
