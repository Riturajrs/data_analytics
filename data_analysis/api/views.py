from django.http import HttpResponse
from rest_framework import status
from rest_framework.response import Response
from rest_framework.decorators import (
    api_view,
)
from api.models import *
from .tasks import generate_report_task


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
        response = HttpResponse(report.csv_file, content_type="text/csv")
        response["Content-Disposition"] = (
            f'attachment; filename="report_{report_id}.csv"'
        )
        return response
    else:
        return Response({"status": "Running"})


@api_view(["GET"])
def trigger_report(request):
    report = Report.objects.create()
    generate_report_task.delay(str(report.id))
    return Response({"report_id": str(report.id)}, status=status.HTTP_200_OK)
