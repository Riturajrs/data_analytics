from django.http import HttpResponse
from rest_framework import status
from rest_framework.response import Response
from rest_framework.decorators import (
    api_view,
)
from api.models import *
from .tasks import generate_report_task
from django.urls import reverse

@api_view(["GET"])
def Ping(request):
    return Response(
        {"status": "online"},
        status=status.HTTP_200_OK,
    )


@api_view(["GET"])
def get_report(request, report_id):
    try:
        report = Report.objects.get(id=report_id)
    except Report.DoesNotExist:
        return Response({"error": "Report not found"}, status=status.HTTP_404_NOT_FOUND)

    if report.status == "Complete":
        download_url = request.build_absolute_uri(reverse('download_report', args=[report_id]))
        return Response({
            "status": "Complete",
            "download_url": download_url
        })
    else:
        return Response({"status": "Running"})

@api_view(["GET"])
def download_report(request, report_id):
    try:
        report = Report.objects.get(id=report_id)
    except Report.DoesNotExist:
        return HttpResponse("Report not found", status=status.HTTP_404_NOT_FOUND)

    if report.status == "Complete" and report.csv_file:
        response = HttpResponse(report.csv_file.read(), content_type="text/csv")
        response["Content-Disposition"] = f'attachment; filename="report_{report_id}.csv"'
        return response
    else:
        return HttpResponse("File not available", status=status.HTTP_404_NOT_FOUND)

@api_view(["GET"])
def trigger_report(request):
    report = Report.objects.create()
    generate_report_task.delay(str(report.id))
    return Response({"report_id": str(report.id)}, status=status.HTTP_200_OK)
