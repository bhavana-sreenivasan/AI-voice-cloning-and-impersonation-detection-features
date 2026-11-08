"""Run a capture source through a LiveAnalyzer on background threads.

    capture thread  ->  bounded queue  ->  analysis thread  ->  events (callback + .events deque)

The queue is bounded on purpose: if scoring is slower than real time (e.g. SSL-AASIST on CPU),
the OLDEST audio is dropped and counted in .dropped_chunks, so the monitor stays current instead
of falling further and further behind the call.
"""
import queue
import threading
from collections import deque

import numpy as np


class LiveSession:
    def __init__(self, analyzer, source, on_event=None, max_queue=50, keep_events=600, drop_oldest=True):
        self.analyzer, self.source, self.on_event = analyzer, source, on_event
        self.drop_oldest = drop_oldest   # False for offline files: wait instead of dropping audio
        self.q = queue.Queue(maxsize=max_queue)
        self.events = deque(maxlen=keep_events)
        self.dropped_chunks = 0
        self.error = None
        self._stop = threading.Event()
        self._cap = threading.Thread(target=self._capture, daemon=True)
        self._ana = threading.Thread(target=self._analyse, daemon=True)

    def start(self):
        self._cap.start()
        self._ana.start()
        return self

    def stop(self):
        self._stop.set()
        try:
            self.source.close()
        except Exception:      # noqa: BLE001
            pass

    def join(self, timeout=None):
        self._cap.join(timeout)
        self._ana.join(timeout)

    @property
    def running(self):
        return self._ana.is_alive()

    def _capture(self):
        try:
            for chunk in self.source:
                if self._stop.is_set():
                    break
                if not self.drop_oldest:
                    while not self._stop.is_set():
                        try:
                            self.q.put(chunk, timeout=0.2)
                            break
                        except queue.Full:
                            pass
                    continue
                while True:
                    try:
                        self.q.put_nowait(chunk)
                        break
                    except queue.Full:
                        try:
                            self.q.get_nowait()
                            self.dropped_chunks += 1
                        except queue.Empty:
                            pass
        except Exception as e:      # noqa: BLE001
            self.error = e
        finally:
            self.q.put(None)        # end-of-stream marker (always fits: consumer is draining)

    def _analyse(self):
        try:
            while True:
                try:
                    chunk = self.q.get(timeout=0.2)
                except queue.Empty:
                    if self._stop.is_set():
                        break
                    continue
                if chunk is None:
                    break
                for ev in self.analyzer.push(np.asarray(chunk, dtype=np.float32)):
                    self.events.append(ev)
                    if self.on_event:
                        self.on_event(ev)
        except Exception as e:      # noqa: BLE001
            self.error = e
