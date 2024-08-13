from django.urls import path
from . import views

urlpatterns = [
    path("trigger-report/", views.trigger_report, name="trigger-report"),
    path("get-report/<report_id>", views.get_report, name="get-report"),
]
