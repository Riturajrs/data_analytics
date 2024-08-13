from django.urls import path
from . import views

urlpatterns = [
    path("ping/", views.Ping),
    path("trigger-report/", views.trigger_report, name="trigger-report"),
]
