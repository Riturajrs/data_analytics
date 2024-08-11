import pytz
from django.utils.dateparse import parse_datetime
from datetime import datetime, timedelta
from models import MenuHours, StoreStatus, StoreTimezones

def convert_to_local_tz(utc_dt, store_id):
    store_tz_str = StoreTimezones.objects.filter(store_id=store_id).values('timezone_str')
    utc_dt = utc_dt.replace(tzinfo=pytz.utc)
    local_tz = pytz.timezone(store_tz_str)
    return utc_dt.astimezone(local_tz)

def parse_time(time_str):
    return datetime.strptime(time_str, '%H:%M').time()

def get_business_hours(store_id):
    return MenuHours.objects.filter(store_id=store_id).values('day', 'start_time_local', 'end_time_local')

def get_observations(store_id):
    return StoreStatus.objects.filter(store_id=store_id).values('timestamp_utc', 'status')
