from itertools import cycle

states = cycle(['free','to_provider','to_customer'])


def next_state():
    print(next(states))


