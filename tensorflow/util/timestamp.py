import time
import datetime

def current_time_in_seconds():
    return time.time()

def current_timestamp():
    return timestamp(current_time_in_seconds())

def timestamp(ts):
    return datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%dT%H-%M-%S')