from django.urls import path
from . import views

urlpatterns = [
    path("trigger_report/", views.trigger_report, name="trigger_report"),
    path("get_report/<report_id>", views.get_report, name="get_report"),
    path("download_report/<report_id>", views.download_report, name="download_report"),
]
