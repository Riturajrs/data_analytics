import pytz
from datetime import datetime, timedelta
from rest_framework import status
from rest_framework.response import Response
from rest_framework.decorators import (
    api_view,
)
from django.db.models import Max
from api.models import *

def convert_to_utc_tz(local_dt, store_id):
    store_tz_str = StoreTimezones.objects.filter(store_id=store_id).values_list('timezone_str', flat=True)[0]
    
    if store_tz_str is None:
        store_tz_str = 'America/Chicago'

    format_str = '%H:%M:%S'
    
    local_naive_dt = datetime.strptime(local_dt, format_str)
    
    local_tz = pytz.timezone(store_tz_str)
    
    
    local_aware_dt = local_tz.localize(local_naive_dt)
    
    utc_dt = local_aware_dt.astimezone(pytz.utc)
    
    return utc_dt

def convert_to_datetime(timestamp_str):
    timestamp_str = timestamp_str.replace(' UTC', '')
    format_str = '%Y-%m-%d %H:%M:%S.%f'
    naive_datetime = datetime.strptime(timestamp_str, format_str)
    utc_datetime = pytz.utc.localize(naive_datetime)
    
    return utc_datetime

def parse_time(time_str):
    return datetime.strptime(time_str, '%H:%M').time()

def get_business_hours(store_id):
    return MenuHours.objects.filter(store_id=store_id).values('day', 'start_time_local', 'end_time_local')

def get_observations(store_id):
    return StoreStatus.objects.filter(store_id=store_id).values('timestamp_utc', 'status')


@api_view(["GET"])
def Ping(request):
    return Response(
        {"status": "online"},
        status=status.HTTP_200_OK,
    )


def get_store_info():
    stores = StoreStatus.objects.values('store_id').distinct()
    
    current_timestamp = StoreStatus.objects.aggregate(max_timestamp=Max('timestamp_utc'))['max_timestamp']
    current_timestamp = convert_to_datetime(current_timestamp)
    last_hour = current_timestamp - timedelta(hours=1)
    
    for store in stores:
        store_id = store['store_id']
        business_hours = get_business_hours(store_id)
        observations = get_observations(store_id) 
        
        business_hour_dict = dict()
        for business_hour in business_hours:
            business_hour['start_time_local'] = convert_to_utc_tz(business_hour['start_time_local'],store_id)
            business_hour['end_time_local'] = convert_to_utc_tz(business_hour['end_time_local'],store_id)
            business_hour_dict[int(business_hour['day'])] = [business_hour['start_time_local'],business_hour['end_time_local']]
        
        status_array = list()
        time_from_start_array = list()
        
        for observation in observations:
            observation['timestamp_utc'] = convert_to_datetime(observation['timestamp_utc'])
            
            if observation['timestamp_utc'].weekday() not in business_hour_dict:
                business_hour_dict[observation['timestamp_utc'].weekday()] = [
                    convert_to_utc_tz('00:00:00', store_id),
                    convert_to_utc_tz('23:59:59', store_id),
                ]
            
            business_hour_timings = business_hour_dict[observation['timestamp_utc'].weekday()]
            
            observation_datetime = observation['timestamp_utc']
            start_time = datetime.combine(observation_datetime.date(), business_hour_timings[0].time())
            end_time = datetime.combine(observation_datetime.date(), business_hour_timings[1].time())
            
            
            start_time = start_time.astimezone(pytz.UTC)
            end_time = end_time.astimezone(pytz.UTC)
            
            if end_time < start_time:
                end_time = end_time + timedelta(days=1)
            
            if start_time <= observation_datetime <= end_time:
                status_array.append(observation['status'])
                time_from_start_array.append(int((observation_datetime-start_time).total_seconds()))
                # return [business_hour_timings, observation_datetime-start_time, observation['status']]
        return [status_array, time_from_start_array]
    return stores
    
    
@api_view(["GET"])
def calculate_time(request):
    uptime_last_week = 0
    downtime_last_week = 0
    

    results = get_store_info()
    return Response(results)