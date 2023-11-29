import random


def random_pause_length():
    """Creates a random number in the given interval.
    This way the pauses between the flipping of Amazon pages are randomized to simulate human behavior."""
    return str(random.uniform(2.0324, 3.4824))


print(random_pause_length())
