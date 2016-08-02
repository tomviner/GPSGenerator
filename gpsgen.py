from itertools import cycle, count
import datetime
from collections import namedtuple, deque
import random, time, json


import numpy as np
import pyproj


CARRIERS = 30
AVG_SPEED = 13.8 # m/s +- 50 km/h
AVG_SPEED_TUPLE = tuple([-13.8,13.8]) 
START_TIME = datetime.datetime.now()
GPS_TRANSMIT_RATE = 15 #seconds
HOURS_PER_SHIFT = 8

Coord = namedtuple("Coord", ["lat","lon"])

"""[To improve] this is just a helper data structure"""
CITIES = {
        "LON": {
            "name": "LONDON",
            "initial_point": Coord(-0.1202201, 51.517235),
            "proj_map": "epsg:27700"
        },
        "MAD": {
            "name": "MADRID",
            "initial_point": Coord(-3.707429, 40.415369),
            "proj_map": "epsg:2062"
        }
}

"""[To improve] this is just a helper data structure"""
MapProj = {
        "LON": pyproj.Proj(init=CITIES["LON"]["proj_map"]),
        "MAD": pyproj.Proj(init=CITIES["MAD"]["proj_map"])
}

worker_sequence = count(start=1, step=1)

task_sequence = count(start=1, step=1)


def next_worker_id():
    return next(worker_sequence)

def next_task_id():
    return next(task_sequence)

def get_current_day():
    today = datetime.date.today()
    return "{}{:02d}{:02d}".format(today.year,today.month,today.day)



"""
Not sure about this
could be interesting, using it as
>>> task = Task(next_task_id())
Task = namedtuple("Task", ['task_id'])
"""

class Task():
    def __init__(self):
        self.task_id = next_task_id()


class Worker():
    states = ['free','to_provider','to_customer']

    def __init__(self, worker_id, city, coord, task_id, state='free'):
        """
        In this proof of concept every Worker has a ACTION linked even if it
        is <Free>, as well as the a last coord, gps coordinates for displaying 
        the last valid position for the worker.
        Speed is a vector np array it allow us to multiply matrixes
        """
        self.worker_id = worker_id
        self.coord = coord
        self.city = city
        self.state_machine = cycle(self.states)
        self.current_state = state
        self.task_id = task_id
        self.speed = np.array((0,0))
        self.mapConvert = MapProj[self.city]
        self.trigger_state_generator()


    def _get_id(self):
        return self.worker_id

    def trigger_state_generator(self):
        """
        Just semantic alias to start the generator(cycle)
        """
        self.current_state = next(self.state_machine)
        return

    def switch_state(self):
        self.current_state = next(self.state_machine)
        return 

    def get_position(self):
        return self.coord

    def __str__(self):
        return "{city} -> {worker_id} {task_id} {coord}".format(**vars(self))


class Step():
    def __init__(self, worker,start_date=None):
        self.worker = worker
        if start_date is not None:
            self.time = start_date
        else:
            self.time = datetime.datetime.now()
        self.rate = datetime.timedelta(seconds=GPS_TRANSMIT_RATE)


    def to_dict(self):
        return vars(self.worker)

    def full_qualified_id(self):
        """
        This is the full qualified id, that wwe will use as key on
        redis(GEOADD).
        """
        return "{city}:{worker_id}:{task_id}".format(**self.to_dict())

    def __str__(self):
        step_data  =  (self.full_qualified_id(),
                       self.worker.current_state,
                       self.worker.coord,
                       self.time)
        return "[<STEP> for {}] [<ACTION>:{}] [<COORD>:{}] [<TIME>: {}]".format(*step_data)



class Simulation():
    """
    This wants to be a base class for different simulation modes
    like ToJsonFileSimulation or RealtimeSimulation (ok real time is always an
    illusion)
    We also add some metadata to the simulation like start and end datetime
    """
    def __init__(self):
        self.starts = datetime.datetime.now()
        self.ends = None
        self.steps = []

class ToJsonFileSimulation(Simulation):

    def add_step(self, step):
        self.steps.append(step)


class RealtimeSimulation(Simulation):

    def gen_stream(self):
        pass


city_switcher = cycle(CITIES.keys())

#considering moving this two next functions to main to handle the simulation
def next_step(step, timeIncrement):
    worker = step.worker
    worker.speed = np.array((random.uniform(-13.8,13.8),
                             random.uniform(-13.8,13.8)))
    lon, lat = worker.coord
    position = np.array(worker.mapConvert(lon, lat))
    position += worker.speed * (timeIncrement.total_seconds())
    lon, lat = worker.mapConvert(position[0], position[1], inverse = True)
    worker.coord = Coord(lon, lat)
    step.time += timeIncrement


def process_step_timeline(step, city):
    for tik in range(0, int(HOURS_PER_SHIFT * 60 * 60 / GPS_TRANSMIT_RATE)):
        """
        # if <store to file> flag
        # send to store
        # or send to a stream
        # or both
        * Think also in a way to freeze the next execution, only in case of
        simulating "real time streaming"
        """
        print (step)
        yield
        step.worker.city = city
        next_step(step, step.rate)


class StepScheduler():
    def __init__(self):
        self._step_queue = deque()

    def new_step(self, step):
        self._step_queue.append(step)

    def run(self):
        while self._step_queue:
            step = self._step_queue.popleft()
            try:
                next(step)
                self._step_queue.append(step)
            except StopIteration:
                pass
# ok this is a mess, but well,, the interface for this craziness took only
# 10 lines of code
# [remenber] you need to think a way to schedule the <ACTIONS> change (odds)
# see you tomorrow
sched = StepScheduler()
simulation_start_date= datetime.datetime(2016,4,25,8,0,0)
for carrier in range(0 , (CARRIERS * len(CITIES))):
        city = next(city_switcher)
        initial_w = Worker(next_worker_id(),
             city,
             CITIES[city]['initial_point'],
             next_task_id())
        s = Step(initial_w, start_date= simulation_start_date)
        # s = Step(initial_w)
        sched.new_step(process_step_timeline(s, city))


sched.run()
