from itertools import cycle, count
import datetime
from collections import namedtuple
import numpy as np


CARRIERS = 600
AVG_SPEED = 13.8 # m/s +- 50 km/h
START_TIME = datetime.datetime.now()

Coord = namedtuple("Coord", ["lat","lon"])

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
        """In this proof of concept every Worker has a ACTION linked even if it is
        <Free>, as well as the a last coord, gps coordinates for displaying 
        the last valid position for the worker.
        Speed is a vector np array it allow us to multiply matrixes"""
        self.worker_id = worker_id
        self.coord = coord
        self.city = city
        self.state_machine = cycle(self.states)
        self.current_state = state
        self.task_id = task_id
        self.speed = np.array((0,0))
        self.trigger_state_generator()


    def _get_id(self):
        return self.worker_id

    def full_qualified_id():
        """This is the full qualified id, that wwe will use as key on
        redis(GEOADD)."""
        return "{city}:{worker_id}:{task_id}"

    def trigger_state_generator(self):
        """Just semantic alias to start the generator(cycle)"""
        self.current_state = next(self.state_machine)
        return

    def switch_state(self):
        self.current_state = next(self.state_machine)
        return 

    def get_position(self):
        return self.coord


class Step():
    def __init__(self, worker):
        self.worker = worker

    def to_dict(self):
        return vars(self.worker)


class Simulation():
    """This wants to be a base class for different simulation modes
    like ToJsonFileSimulation or RealtimeSimulation (ok real time is always an
    illusion)
    We also add some metadata to the simulation like start and end datetime"""
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


#considering moving this two next functions to main as an interface to the simulation
def gen_result(step):
    """Returns a list with the complete simulation"""
    pass

def gen_next_step(step):
    """Return one step, for streaming or whatever"""
    pass

