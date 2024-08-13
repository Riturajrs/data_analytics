import csv
import io
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pytz
from datetime import datetime, timedelta
from rest_framework import status
from rest_framework.response import Response
from rest_framework.decorators import (
    api_view,
)
from django.core.files.base import ContentFile
from api.models import *


def convert_to_utc_tz(local_dt, store_id):
    store_tz_str = StoreTimezones.objects.filter(store_id=store_id).values_list(
        "timezone_str", flat=True
    )

    if store_tz_str is None or len(store_tz_str) == 0:
        store_tz_str = "America/Chicago"
    else:
        store_tz_str = store_tz_str[0]
    format_str = "%H:%M:%S"

    local_naive_dt = datetime.strptime(local_dt, format_str)

    local_tz = pytz.timezone(store_tz_str)

    local_aware_dt = local_tz.localize(local_naive_dt)

    utc_dt = local_aware_dt.astimezone(pytz.utc)

    return utc_dt


def convert_to_datetime(timestamp_str):
    timestamp_str = timestamp_str.replace(" UTC", "")
    format_str = "%Y-%m-%d %H:%M:%S.%f"
    naive_datetime = datetime.strptime(timestamp_str, format_str)
    utc_datetime = pytz.utc.localize(naive_datetime)

    return utc_datetime


def parse_time(time_str):
    return datetime.strptime(time_str, "%H:%M").time()


def get_business_hours(store_id):
    return MenuHours.objects.filter(store_id=store_id).values(
        "day", "start_time_local", "end_time_local"
    )


def get_observations(store_id):
    return StoreStatus.objects.filter(store_id=store_id).values(
        "timestamp_utc", "status"
    )


@api_view(["GET"])
def Ping(request):
    return Response(
        {"status": "online"},
        status=status.HTTP_200_OK,
    )


def predict_state(classifier, value):
    prediction = classifier.predict(np.array([[value]]))
    return "active" if prediction[0] == 1 else "inactive"


def get_model(states, values):

    state_to_num = {"active": 1, "inactive": 0}
    y = np.array([state_to_num[state] for state in states])

    X = np.array(values).reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    classifier = RandomForestClassifier(n_estimators=100, random_state=45)
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Model accuracy: {accuracy:.2f}")

    return classifier


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


def get_all_stores_info():
    stores = StoreStatus.objects.values("store_id").distinct()[:10]
    store_data = []

    for store in stores:
        store_info = get_store_info(store["store_id"])
        store_data.append(store_info)

    return store_data


def trigger_report(request):
    report = Report.objects.create()
    generate_report_task(report.id)
    return Response({"report_id": str(report.id)})


def generate_report_task(report_id):
    results = get_all_stores_info()

    output = io.StringIO()
    writer = csv.writer(output)
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

    csv_content = output.getvalue()
    output.close()

    report = Report.objects.get(id=report_id)
    report.status = "Complete"
    report.csv_file.save(f"report_{report_id}.csv", ContentFile(csv_content))
    report.save()


@api_view(["GET"])
def trigger_report(request):
    report = Report.objects.create()
    generate_report_task(report.id)
    return Response({"report_id": str(report.id)})
