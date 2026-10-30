"""Bounded display work; full-rate samples remain in acquisition/recording."""
from collections import deque
from threading import Lock

import numpy as np


def peak_envelope(times, values, maximum_points=4096):
    """Keep each bucket's actual min/max samples in chronological order."""
    if maximum_points < 2:
        raise ValueError('Peak envelopes require at least two display points')
    if len(values) <= maximum_points:
        return times, values
    bucket = int(np.ceil(len(values) / max(1, maximum_points // 2)))
    complete = len(values) // bucket
    grouped = values[:complete * bucket].reshape(complete, bucket)
    offsets = np.arange(complete) * bucket
    indices = np.column_stack((offsets + grouped.argmin(axis=1), offsets + grouped.argmax(axis=1)))
    indices.sort(axis=1)
    indices = indices.ravel()
    tail = values[complete * bucket:]
    if len(tail):
        indices = np.r_[indices, complete * bucket + np.sort([tail.argmin(), tail.argmax()])]
    return times[indices], values[indices]


class ChannelMailbox:
    """Worker callbacks only store references; the GUI drains on its timer.

    Publishers transfer immutable block buffers. Drop obsolete display blocks,
    never acquisition/recording samples, and retain sample-count continuity.
    """
    def __init__(self, capacity_samples, channels):
        self.lock = Lock()
        self.capacity = capacity_samples
        self.channels = channels
        self.blocks = deque()
        self.stored_samples = 0
        self.seen_samples = 0
        self.seen_blocks = 0
        self.dropped = False
        self.psd = None
        self.error = None

    def submit(self, block):
        values = np.atleast_2d(np.asarray(block))
        with self.lock:
            if values.ndim != 2 or values.shape[0] != self.channels:
                self.error = 'Channel display count does not match the captured run setup.'
                return
            self.seen_samples += values.shape[1]
            self.seen_blocks += 1
            if values.shape[1] > self.capacity:
                values = values[:, -self.capacity:]
                self.dropped = True
            self.blocks.append(values)
            self.stored_samples += values.shape[1]
            while len(self.blocks) > 256 or (self.stored_samples > self.capacity and len(self.blocks) > 1):
                self.stored_samples -= self.blocks.popleft().shape[1]
                self.dropped = True

    def submit_psd(self, packet):
        with self.lock:
            self.psd = packet

    def drain(self):
        with self.lock:
            batch = (list(self.blocks), self.seen_samples, self.seen_blocks, self.dropped, self.psd, self.error)
            self.blocks.clear()
            self.stored_samples = self.seen_samples = self.seen_blocks = 0
            self.dropped = False
            self.psd = self.error = None
        return batch
