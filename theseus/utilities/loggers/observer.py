from typing import Dict, List
import logging
import torch
import matplotlib as mpl
from .subscriber import LoggerSubscriber

def get_type(value):
    if isinstance(value, torch.nn.Module):
        return LoggerObserver.TORCH_MODULE
    if isinstance(value, mpl.figure.Figure):
        return LoggerObserver.FIGURE
    if isinstance(value, str):
        return LoggerObserver.TEXT
    return LoggerObserver.SCALAR

class LoggerObserver(object):
    """Logger Oberserver Degisn Pattern
    notifies every subscribers when .log() is called
    """
    SCALAR = 'scalar'
    FIGURE = 'figure'
    TORCH_MODULE = 'torch_module'
    TEXT = 'text'

    WARN = logging.WARN
    ERROR = logging.ERROR
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    CRITICAL = logging.CRITICAL

    instances = {}

    def __new__(cls, name, *args, **kwargs):
        if name in LoggerObserver.instances.keys():
            return LoggerObserver.instances[name]
        else:
            return object.__new__(cls, *args, **kwargs)

    def __init__(self, name) -> None:
        self.subscriber = []
        self.name = name
        LoggerObserver.instances[name] = self

    @classmethod
    def getLogger(cls, name):
        if name in LoggerObserver.instances.keys():
            return LoggerObserver.instances[name]
        else:
            return cls(name)

    def subscribe(self, subscriber: LoggerSubscriber):
        self.subscriber.append(subscriber)

    def log(self, logs: List[Dict]):
        for subscriber in self.subscriber:
            for log in logs:
                tag = log['tag']
                value = log['value']
                type = log['type'] if 'type' in log.keys() else get_type(value)
                kwargs = log['kwargs'] if 'kwargs' in log.keys() else {}

                if type == LoggerObserver.SCALAR:
                    subscriber.log_scalar(
                        tag=tag,
                        value=value,
                        **kwargs
                    )

                if type == LoggerObserver.FIGURE:
                    subscriber.log_figure(
                        tag=tag,
                        value=value,
                        **kwargs
                    )

                if type == LoggerObserver.TORCH_MODULE:
                    subscriber.log_torch_module(
                        tag=tag,
                        value=value,
                        **kwargs
                    )

                if type == LoggerObserver.TEXT:
                    subscriber.log_text(
                        tag=tag,
                        value=value,
                        **kwargs
                    )

    def text(self, value, level):
        self.log([{
            'tag': 'stdout',
            'value': value,
            'type': LoggerObserver.TEXT,
            'kwargs': {
                'level': level
            }
        }])