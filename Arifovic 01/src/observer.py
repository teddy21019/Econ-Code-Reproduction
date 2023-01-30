from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict

from pandas import DataFrame


class DataCollector:
    agent_data: DataFrame   = DataFrame()
    model_data: DataFrame   = DataFrame()
    _agent_data_dict : Dict[str, Any] = {}
    _model_data_dict : Dict[str, Any] = {}
    time      : int         =   field(default=0)

    def add_agent_data(self, id, key, value):
        pass

    def add_model_data(self, *args, **kwargs):
        pass 

    def next_tick(self):
        self.time += 1
        
    
    @property
    def result(self):
        return self.agent_data, self.model_data

