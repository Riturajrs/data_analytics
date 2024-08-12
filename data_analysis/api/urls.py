from django.urls import path
from . import views

urlpatterns = [path('ping/', views.Ping), path('get-report/', views.calculate_time)]