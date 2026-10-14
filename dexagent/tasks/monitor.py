"""Stop an episode at the first measured success or failure, within an action."""


class EpisodeEnded(Exception):
    def __init__(self, verdict, state):
        super().__init__(verdict.reason)
        self.verdict = verdict
        self.state = state


class TaskMonitor:
    def __init__(self, task, measure=None):
        self.task = task
        self.measure = measure or (lambda state: state)
        self.verdict = None
        self.samples = 0
        self.last_time = -1.0

    def __call__(self, state):
        if self.verdict is not None and self.verdict.terminal:
            raise EpisodeEnded(self.verdict, state)
        if state.time_s <= self.last_time:
            return
        self.last_time = state.time_s
        self.samples += 1
        self.verdict = self.task.update(self.measure(state))
        if self.verdict.terminal:
            raise EpisodeEnded(self.verdict, state)
