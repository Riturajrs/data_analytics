import csv
import io
from .utils import get_all_stores_info
from .models import *
from django.core.files.base import ContentFile
from celery import shared_task

@shared_task
def generate_report_task(report_id):
    try:
        results = get_all_stores_info()
        output = io.StringIO()
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(
            [
                "store_id",
                "uptime_last_hour(in minutes)",
                "uptime_last_day(in hours)",
                "update_last_week(in hours)",
                "downtime_last_hour(in minutes)",
                "downtime_last_day(in hours)",
                "downtime_last_week(in hours)",
            ]
        )

        for row in results:
            writer.writerow(
                [
                    row["store_id"],
                    row["uptime_last_hour(in minutes)"],
                    row["uptime_last_day(in hours)"],
                    row["update_last_week(in hours)"],
                    row["downtime_last_hour(in minutes)"],
                    row["downtime_last_day(in hours)"],
                    row["downtime_last_week(in hours)"],
                ]
            )

        csv_content = output.getvalue().strip()
        output.close()

        report = Report.objects.get(id=report_id)
        report.status = "Complete"
        report.csv_file.save(f"report_{report_id}.csv", ContentFile(csv_content))
        report.save()

    except Exception as e:
        print("Error in report generation: %s", str(e))


