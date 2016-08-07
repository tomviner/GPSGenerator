import sys
from itertools import cycle, count
import datetime
from collections import namedtuple, deque
import random, time, json
import argparse
from argparse import RawTextHelpFormatter
import time
from functools import wraps
from distutils.version import StrictVersion

import numpy as np
import pyproj
import redis
from redis.exceptions import ConnectionError


Coord = namedtuple("Coord", ["lat","lon"])

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

MapProj = {
        "LON": pyproj.Proj(init=CITIES["LON"]["proj_map"]),
        "MAD": pyproj.Proj(init=CITIES["MAD"]["proj_map"])
}

worker_sequence = count(start=1, step=1)

task_sequence = count(start=1, step=1)

city_switcher = cycle(CITIES.keys())


def next_worker_id():
    return next(worker_sequence)

def next_task_id():
    return next(task_sequence)

def get_current_day():
    today = datetime.date.today()
    return "{}{:02d}{:02d}".format(today.year,today.month,today.day)



"""
Not sure about this could be interesting, using it as
Task = namedtuple("Task", ['task_id'])
>>> task = Task(next_task_id())
**Note currently not using this, if we dont need no sync to a real database,
lets keep this, unused.
"""

class Task():
    def __init__(self):
        self.task_id = next_task_id()

"""
I changed the state machine to iterate over this namedtuples, in which I
will store 1/100 to change odds to change next state. Quick dirty solution.
I also stored the code and a name for a posible representation in a redis key
"""
Free = namedtuple("Free", ['odds', 'code', 'name'])
ToProvider = namedtuple("ToProvider", ['odds', 'code', 'name'])
ToCustomer = namedtuple("ToCustomer", ['odds', 'code', 'name'])
"""
## Instantiation for the class variable, lets start thinking the odds will be
inmutable, thats why I stored them in a namedtuple,  want to keep this data
structure tight, quick and dirty again.
"""
free = Free(6,0,'free') # this means 6/100 chances to change to --> ToProvider state
to_provider = ToProvider(2,1,'to_provider') # 2/100 chances to change to --> ToCustomer state
to_customer = ToCustomer(2,2,'to_customer') # 2/100 chances to change to --> Free state


class Worker():
    states = [free, to_provider, to_customer]

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
    """
    1 to 1 wrapper to a Worker, not only for semantics, a Silumation
    has Steps not Workers, but also for adding the initial step data like
    start_date.
    The start_date is the reference date from which next step is processed
    Worker instance is manipulated from Step too.
    """
    def __init__(self, worker,start_date=None):
        self.worker = worker
        if start_date is not None:
            self.time = start_date
        else:
            self.time = datetime.datetime.now()


    def to_dict(self):
        return vars(self.worker)

    def full_qualified_id(self):
        """
        Full qualified id, that we will use as key on redis(GEOADD).
        **Note
        Maybe is a good idea to add a day reference to this, This way web
        can manage Courier data in redis like a session. Lets see which is
        the best format for this get_current_day() first class function is
        for this. Consider also adding this function as a class member.
        """
        return "{city}:{worker_id}:{task_id}".format(**self.to_dict())

    def to_json_line(self):
        return {'id': self.full_qualified_id(),
                'state': self.worker.current_state.name,
                'coord': [self.worker.coord.lat, self.worker.coord.lon],
                'timestamp': float(time.mktime(self.time.timetuple()))
                }

    def __str__(self):
        step_data  =  (self.full_qualified_id(),
                       self.worker.current_state,
                       self.worker.coord,
                       self.time)
        f_string = "[<STEP> for {}] [<ACTION>:{}] [<COORD>:{}] [<TIME>: {}]"
        return f_string.format(*step_data)


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

def io_coroutine(gen):
    """
    This decorator, pumps the coroutine to the next step.
    It is also a Syntactic sugar for identifying coroutines.
    Which are much more visible now.
    Registers the initiliced coroutine for closing them when
    the data flow ends, this is specially important, when
    opened file descriptors are involved, in this case we are
    sending data to a "with open" block, registering coroutines
    allow us to kill them easyly with Simulator.end_io_coroutines()
    """
    def pumped(self, *args, **kwargs):
        g = gen(self, *args, **kwargs)
        self.io_coroutines.append(g)
        next(g)
        return g
    return pumped

class Simulation():
    """
    This is the main class of this script. It configures the simulation, starts
    the necesary coroutines, and starts the scheduler. In the schedulers queue
    are stored the main generator(freezed), generator that triggers each
    coroutine when required. Add More ...
    """
    def __init__(self, sim_type, nworkers, hours_shift, speed, transmit_rate,
                     is_json, is_pretty, start_date=None, database=None):
        self.sim_type = sim_type
        self.nworkers = nworkers
        self.hours_shift = hours_shift
        self.speed = speed
        self.transmit_rate = transmit_rate
        self.is_json = is_json
        self.is_pretty = is_pretty
        self.start_date = start_date
        self.ends = None
        self.database = database
        self.scheduler = StepScheduler()
        self.io_coroutines = []
        self.instantiate_coroutines()

    def instantiate_coroutines(self):
        if self.is_json_file:
            self.filewriter = self.file_writer_to_json()
        else:
            self.filewriter = self.file_writer()
        self.dbwriter = self.send_to_db()

    def end_io_coroutines(self):
        for coro in self.io_coroutines:
            try:
                coro.send(None)
            except StopIteration:
                pass

    @property
    def rate(self):
        return datetime.timedelta(seconds=self.transmit_rate)

    @property
    def starts(self):
        if self.start_date is None:
            return datetime.datetime.now()
        else:
            return self.start_date

    @property
    def is_json_file(self):
        return any([self.is_json, self.is_pretty])

    @property
    def metadata(self):
        return {'start_date': str(self.starts),
                'end_date': None,
                'simulation_type': self.sim_type}

    def produce_json_structure(self):
        return {'metadata': self.metadata,
                'data': []}

    @io_coroutine
    def send_to_db(self):
        while True:
            data = yield
            if data is None:
                break
            key = data.full_qualified_id()
            lon, lat = data.worker.coord
            t = data.time.timestamp()
            values = (lon, lat, t)
            self.database.execute_command("GEOADD", key, *values)

    @io_coroutine
    def file_writer(self):
        with open('data.dat', 'wt') as f:
            while True:
                line = yield
                if line is None:
                    break
                f.write(str(line) + "\n")

    @io_coroutine
    def file_writer_to_json(self):
        with open('data.json', 'w+') as f:
            json_object = self.produce_json_structure()
            while True:
                line = yield
                if line is None:
                    break
                json_object['data'].append(line.to_json_line())
            if self.is_pretty:
                f.write(json.dumps(json_object, indent=4))
            else:
                f.write(json.dumps(json_object))

    def next_step(self, step, timeIncrement):
        worker = step.worker
        worker.speed = np.array((random.uniform(-(self.speed),self.speed),
                                random.uniform(-(self.speed),self.speed)))
        lon, lat = worker.coord
        position = np.array(worker.mapConvert(lon, lat))
        position += worker.speed * (timeIncrement.total_seconds())
        lon, lat = worker.mapConvert(position[0], position[1], inverse = True)
        worker.coord = Coord(lon, lat)
        step.time += timeIncrement
        if worker.current_state.odds >= random.randint(1,100):
            if isinstance(worker.current_state, Free):
                worker.task_id = next_task_id()
            if isinstance(worker.current_state, ToCustomer):
                """
                This means next state is free
                task id zero will be the representation of a free of duty
                courier
                Maybe its better to put a string like "FREE for that"
                """
                worker.task_id = 0
            worker.switch_state()

    def process_step_timeline(self, step, city):
        for _ in range(0, int(self.hours_shift * 60*60 / self.transmit_rate)):
            """
            Explain this in a propper way...
            """
            if self.sim_type == "live":
                self.dbwriter.send(step)
            if self.sim_type == "file":
                self.filewriter.send(step)
            if self.sim_type == "both":
                self.dbwriter.send(step)
                self.filewriter.send(step)
            yield
            step.worker.city = city
            self.next_step(step, self.rate)

    def start(self):
        try:
            for carrier in range(0 , (self.nworkers * len(CITIES))):
                city = next(city_switcher)
                initial_w = Worker(next_worker_id(),
                                city,
                                CITIES[city]['initial_point'],
                                0)
                s = Step(initial_w, start_date=self.starts)
                self.scheduler.new_step(self.process_step_timeline(s, city))
            self.scheduler.run()
            self.end_io_coroutines()
        except (KeyboardInterrupt, SystemExit):
            sys.stdout.write("\nClosing all files properly....")
            self.end_io_coroutines()
            sys.stdout.write("\nFiles closed.\n")

#helperS
def default_date():
    return  datetime.datetime.now()

# Validators
def valid_date(s):
    try:
        date = datetime.datetime.strptime(s, "%Y-%m-%d")
        return date.replace(hour=8,minute=0)
    except ValueError:
        msg = "Not a valid date: '{0}'.".format(s)
        raise argparse.ArgumentTypeError(msg)

def valid_hours(s):
    hours = int(s)
    if not (1 <= hours <= 24):
        msg = "Hours must be between 1 and 24"
        raise argparse.ArgumentTypeError(msg)
    return hours

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="gpsgen",
                                     description="GPS Stream generator",
                                     formatter_class=RawTextHelpFormatter)
    parser.add_argument("sim_type",
                        choices=['live', 'file', 'both'],
                        help="Select a type of simulation to execute\n")

    parser.add_argument("-nw", "--nworkers",
                        type=int,
                        default=600,
                        action="store",
                        dest="nworkers",
                        help="Number of Workers per city.\n"
                             "Unit: %(type)s\n"
                             "Default: %(default)s")

    parser.add_argument("-s", "--speed",
                        type=float,
                        default=13.8,
                        action="store",
                        dest="speed",
                        help="Speed workers are continously moving\n"
                             "Unit: m/s\n"
                             "Default: %(default)s m/s")

    parser.add_argument("-hs", "--hours-shift",
                        type=valid_hours,
                        default=24,
                        action="store",
                        dest="hours_shift",
                        help="Hours per shift, the worker drives\n"
                             "Unit: int\n"
                             "Default: %(default)s")

    parser.add_argument("-tr", "--transmit-rate",
                        type=int,
                        default=15,
                        action="store",
                        dest="transmit_rate",
                        help="GPS tracking transmit rate in seconds\n"
                             "Unit: seconds\n"
                             "Default: %(default)s s")

    parser.add_argument("-d", "--start-date",
                        type=valid_date,
                        default=default_date(),
                        action="store",
                        dest="start_date",
                        help="Start date for the simulation\n"
                             "Format: YYYY-MM-DD\n"
                             "Default: NOW")

    parser.add_argument("-j", "--json",
                        default=False,
                        action="store_true",
                        help="Write the file in json format compressed\n")

    parser.add_argument("-pj", "--pretty-json",
                        dest="pretty",
                        default=False,
                        action="store_true",
                        help="Write the file in json format\n"
                             "Indented 4 spaces\n")


    parser.add_argument("--version",
                        action="version",
                        version='%(prog)s 0.1.0')


    args = parser.parse_args()

    if args.sim_type == "live" and (args.json or args.pretty):
        msg = "Live simulation and json flag is not compatible"
        raise argparse.ArgumentTypeError(msg)

    if args.json and args.pretty:
        msg = "Choose between --json or --pretty-json, not both"
        raise argparse.ArgumentTypeError(msg)

    try:
        r = redis.StrictRedis(host='localhost', port=6379, db=0)
        info = r.execute_command("INFO")
    except Exception:
        raise Exception("Be sure redis server is on, on port 6379")

    if StrictVersion(info['redis_version']) < StrictVersion("3.2.0"):
        raise Exception("GEO redis is available only from version '3.2.0'")

    simulation = Simulation(args.sim_type,
                            args.nworkers,
                            args.hours_shift,
                            args.speed,
                            args.transmit_rate,
                            args.json,
                            args.pretty,
                            start_date=args.start_date,
                            database=r)
    simulation.start()
