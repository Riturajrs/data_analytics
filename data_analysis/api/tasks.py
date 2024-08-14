import csv
import io
import pytz

from .ml import get_model, predict_state
from .utils import convert_to_datetime, convert_to_utc_tz, get_business_hours, get_observations, get_all_stores_info
from .models import *
from django.core.files.base import ContentFile
from celery import shared_task
from datetime import datetime, timedelta

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


def calculate_uptime_and_downtime_in_minutes(
    state_predictor, start_time, end_time, current_time, observations_dict
):
    uptime = 0
    downtime = 0
    while current_time < end_time:
        state = observations_dict.get(current_time, None)
        if state is None:
            minutes_since_start = int((current_time - start_time).total_seconds() / 60)
            state = predict_state(state_predictor, minutes_since_start)

        if state == "active":
            uptime += 1
        else:
            downtime += 1
        current_time += timedelta(minutes=1)
    return uptime, downtime


def get_store_info(store_id):
    last_hour = None
    business_hours = get_business_hours(store_id)
    observations = get_observations(store_id)
    business_hour_dict = dict()
    for business_hour in business_hours:
        business_hour["start_time_local"] = convert_to_utc_tz(
            business_hour["start_time_local"], store_id
        )
        business_hour["end_time_local"] = convert_to_utc_tz(
            business_hour["end_time_local"], store_id
        )
        business_hour_dict[int(business_hour["day"])] = [
            business_hour["start_time_local"],
            business_hour["end_time_local"],
        ]
    status_array = list()
    time_from_start_array = list()
    observations_dict = dict()
    last_day_start_time = None
    last_day_end_time = None
    for observation in observations:
        observation["timestamp_utc"] = convert_to_datetime(observation["timestamp_utc"])
        if observation["timestamp_utc"].weekday() not in business_hour_dict:
            business_hour_dict[observation["timestamp_utc"].weekday()] = [
                convert_to_utc_tz("00:00:00", store_id),
                convert_to_utc_tz("23:59:59", store_id),
            ]
        business_hour_timings = business_hour_dict[
            observation["timestamp_utc"].weekday()
        ]
        observation_datetime = observation["timestamp_utc"]
        start_time = datetime.combine(
            observation_datetime.date(), business_hour_timings[0].time()
        )
        end_time = datetime.combine(
            observation_datetime.date(), business_hour_timings[1].time()
        )
        start_time = start_time.astimezone(pytz.UTC)
        end_time = end_time.astimezone(pytz.UTC)
        if end_time < start_time:
            end_time = end_time + timedelta(days=1)
        business_hour_dict[observation["timestamp_utc"].weekday()] = [
            start_time,
            end_time,
        ]
        if last_hour is None or last_hour < end_time:
            last_hour = end_time
        if last_day_end_time is None or last_day_end_time < end_time:
            last_day_end_time = end_time
            last_day_start_time = start_time
        if start_time <= observation_datetime <= end_time:
            status_array.append(observation["status"])
            time_from_start_array.append(
                int((observation_datetime - start_time).total_seconds() / 60)
            )
            minute_start = observation_datetime.replace(second=0, microsecond=0)
            observations_dict[minute_start] = observation["status"]
    state_predictor = get_model(status_array, time_from_start_array)
    last_hour = last_hour.replace(second=0, microsecond=0)
    end_time = last_hour
    last_hour = last_hour - timedelta(hours=1)
    uptime_last_hour, downtime_last_hour = calculate_uptime_and_downtime_in_minutes(
        state_predictor, start_time, end_time, last_hour, observations_dict
    )
    last_day_start_time = last_day_start_time.replace(second=0, microsecond=0)
    last_day_end_time = last_day_end_time.replace(second=0, microsecond=0)
    uptime_last_day, downtime_last_day = calculate_uptime_and_downtime_in_minutes(
        state_predictor,
        last_day_start_time,
        last_day_end_time,
        last_day_start_time,
        observations_dict,
    )
    uptime_last_day = int(uptime_last_day / 60)
    downtime_last_day = int(downtime_last_day / 60)
    uptime_last_week = 0
    downtime_last_week = 0
    for (
        start_time_business_hour,
        end_time_business_hour,
    ) in business_hour_dict.values():
        start_time_business_hour = start_time_business_hour.replace(
            second=0, microsecond=0
        )
        end_time_business_hour = end_time_business_hour.replace(second=0, microsecond=0)
        uptime_day, downtime_day = calculate_uptime_and_downtime_in_minutes(
            state_predictor,
            start_time_business_hour,
            end_time_business_hour,
            start_time_business_hour,
            observations_dict,
        )
        uptime_day = int(uptime_day / 60)
        downtime_day = int(downtime_day / 60)
        uptime_last_week += uptime_day
        downtime_last_week += downtime_day

    return {
        "store_id": store_id,
        "uptime_last_hour(in minutes)": uptime_last_hour,
        "uptime_last_day(in hours)": uptime_last_day,
        "update_last_week(in hours)": uptime_last_week,
        "downtime_last_hour(in minutes)": downtime_last_hour,
        "downtime_last_day(in hours)": downtime_last_day,
        "downtime_last_week(in hours)": downtime_last_week,
    }


