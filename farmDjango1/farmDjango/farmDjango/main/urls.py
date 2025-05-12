# main/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('about/', views.about_view, name='about'),
    path('login/', views.login, name='login'),
    path('signup/', views.signup_view, name='signup'),  
    path('contact/', views.contact_view, name='contact'), 
    path('service/', views.services_view, name='service'), 
    path('fertilizer/', views.fertilizer_view, name='fertilizer'), 
    path('landPrice/', views.land_view, name='landPrice'), 
    path('yield/', views.yield_view, name='yield'), 
    path('disease/', views.disease_view, name='disease'), 
    path('voluntary/', views.voluntary_view, name='voluntary'), 




]
