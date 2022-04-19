"""
This file is the implementation of ID-assignment coordinator.
"""

"""
in detector initialization, include a coordinator.
in tracker initialization, include a coordinator (pointer), points to the detector.coordinator
"""


class Coordinator:
    def __init__(self, num_trackers):
        self.__counter = 0
        self.__num_trackers = num_trackers

    def assign_id(self):
        return

