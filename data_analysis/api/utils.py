from .ml import get_model, predict_state
from .models import *
import pytz
from datetime import datetime, timedelta


def convert_to_utc_tz(local_dt, store_id):
    store_tz_str = StoreTimezones.objects.filter(store_id=store_id).values_list(
        "timezone_str", flat=True
    )

    # Default timezone
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


def calculate_uptime_and_downtime_in_minutes(
    state_predictor, start_time, end_time, current_time, observations_dict
):
    uptime = 0
    downtime = 0
    # current time at the start is start of business hours time
    while current_time < end_time:
        state = observations_dict.get(current_time, None)
        if state is None:
            # Since the model has been trained on minutes' data the time from start needs to be in minutes
            minutes_since_start = int((current_time - start_time).total_seconds() / 60)
            state = predict_state(state_predictor, minutes_since_start)

        if state == "active":
            uptime += 1
        else:
            downtime += 1
        # Here we are calculating status every minute
        current_time += timedelta(minutes=1)
    return uptime, downtime

# This function is for efficiently calculating status every hour
def calculate_uptime_and_downtime_in_hours(
    state_predictor, start_time, end_time, current_time, observations_dict
):
    uptime = 0
    downtime = 0
    # current time at the start is start of business hours time
    while current_time < end_time:
        state = observations_dict.get(current_time, None)
        if state is None:
            # Since the model has been trained on minutes' data the time from start needs to be in minutes
            minutes_since_start = int((current_time - start_time).total_seconds() / 60)
            state = predict_state(state_predictor, minutes_since_start)

        if state == "active":
            uptime += 1
        else:
            downtime += 1
        # Here we are calculating status every hour
        current_time += timedelta(hours=1)
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
        # Business hour for each day
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
            # If business hours are not given, it means it is always open
            business_hour_dict[observation["timestamp_utc"].weekday()] = [
                convert_to_utc_tz("00:00:00", store_id),
                convert_to_utc_tz("23:59:59", store_id),
            ]
        business_hour_timings = business_hour_dict[
            observation["timestamp_utc"].weekday()
        ]
        observation_datetime = observation["timestamp_utc"]

        # Combining start time and end time with the day of the observation
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
            
        # Updating business hour dictionary with UTC start and end times
        business_hour_dict[observation["timestamp_utc"].weekday()] = [
            start_time,
            end_time,
        ]
        
        # Recording latest observation for calculating status of last hour
        if last_hour is None or last_hour < end_time:
            last_hour = end_time
            
        # Recording latest observation for calculating status of last day
        if last_day_end_time is None or last_day_end_time < end_time:
            last_day_end_time = end_time
            last_day_start_time = start_time
        
        # If the given observation is within the business hours, then store it for training the ML model
        if start_time <= observation_datetime <= end_time:
            status_array.append(observation["status"])
            time_from_start_array.append(
                int((observation_datetime - start_time).total_seconds() / 60)
            )
            # Before storing the observations in dictionary remove unncessary seconds and miliseconds data
            minute_start = observation_datetime.replace(second=0, microsecond=0)
            observations_dict[minute_start] = observation["status"]
            
    # Train ML model for extrapolation of data through predictions
    state_predictor = get_model(status_array, time_from_start_array)
    
    # Preparaing variables for calculation of last hour
    last_hour = last_hour.replace(second=0, microsecond=0)
    end_time = last_hour
    last_hour = last_hour - timedelta(hours=1)
    
    # Recording status every minute
    uptime_last_hour, downtime_last_hour = calculate_uptime_and_downtime_in_minutes(
        state_predictor, start_time, end_time, last_hour, observations_dict
    )
    
    # Preparaing variables for calculation of last day
    last_day_start_time = last_day_start_time.replace(second=0, microsecond=0)
    last_day_end_time = last_day_end_time.replace(second=0, microsecond=0)
    
    # Recording status every hour
    uptime_last_day, downtime_last_day = calculate_uptime_and_downtime_in_hours(
        state_predictor,
        last_day_start_time,
        last_day_end_time,
        last_day_start_time,
        observations_dict,
    )
    
    uptime_last_week = 0
    downtime_last_week = 0
    
    # Running loop for calculating status of each day for the past week
    for (
        start_time_business_hour,
        end_time_business_hour,
    ) in business_hour_dict.values():
        
        # Preparing data for calculation of status for a particular day
        start_time_business_hour = start_time_business_hour.replace(
            second=0, microsecond=0
        )
        end_time_business_hour = end_time_business_hour.replace(second=0, microsecond=0)
        
        # Recording status for a particular day on per hour basis
        uptime_day, downtime_day = calculate_uptime_and_downtime_in_hours(
            state_predictor,
            start_time_business_hour,
            end_time_business_hour,
            start_time_business_hour,
            observations_dict,
        )
        
        # Adding up status for each day, for status of last week
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
    stores = StoreStatus.objects.values("store_id").distinct()
    # Sampling top 100 stores
    stores = stores[:100]
    store_data = []

    for store in stores:
        store_info = get_store_info(store["store_id"])
        store_data.append(store_info)

    return store_data

