from itertools import cycle, count
import datetime
from collections import namedtuple

Coord = namedtuple("Coord", ["lat","lon"])

CARRIERS = 600
AVG_SPEED = 13.8 # m/s
START_TIME = datetime.datetime.now()



CITIES = [
        {
            "name": "LONDON",
            "code": "LON",
            "initial_point": Coord(-0.1202201, 51.517235)
        },
        {
            "name": "MADRID",
            "code": "MAD",
            "initial_point": Coord(-3.707429, 40.415369)
        }
]

states = cycle(['free','to_provider','to_customer'])

worker_sequence = count(start=1, step=1)

def next_state():
    return next(states)

def next_worker_id():
    return next(worker_sequence)

def get_current_day():
    today = datetime.date.today()
    return "{}{:02d}{:02d}".format(today.year,today.month,today.day)
