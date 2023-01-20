from collections import deque
from typing import Deque, Dict 
from mesa import Agent, Model
from mesa.time import BaseScheduler

Generation = Deque[Dict[Agent]]

class OGActivation(BaseScheduler):
    """
    This scheduler is custom for an overlapping generation model. 
    Each agent is one of the n-th generation, and as the model steps, 
    it steps into the next generation. Each generation has different duties, 
    thus different class methods to appoint. 

    The activation order for this custom model is the order of the generation, 
    that is, younger generation moves first. This might not be the case 
    for other models.
    """

    def __init__(self, model: Model, generation_num : int = 2) -> 'OGActivation':
        super().__init__(model)
        self._max_gen_num = generation_num
        self.generation_list : Generation = deque(maxlen=generation_num)
        """
        For a deque, pop() takes the last item, while popleft() takes the first item
        append and append left follows the same rule
        """
    
    @property
    def _deque_num(self):
        return len(self.generation_list)
    
    def new_generation(self):
        self.generation_list.appendleft(
            {}
        )
    

    def add(self, agent: Agent, generation: int = 0):
        """
        Adds an agent to a certain generation 
        """
        if agent.unique_id in self._agents:
            raise Exception(
                f"Agent with unique id {repr(agent.unique_id)} already added to scheduler"
            )

        self._agents[agent.unique_id] = agent

        if generation >= self._deque_num:
            raise Exception(
                f"Unable to add agents to generation index {generation}. Current max generation index at {self._deque_num}."
            )
        self.generation_list[generation][agent.unique_id] = agent

    def remove(self, agent: Agent) -> None:
        """
        Has to find its location in generation.
        No need to implement in this example.
        """
        gen_index = 0

        while gen_index < self._deque_num :
            if agent.unique_id in self.generation_list[gen_index]:
                del self.generation_list[gen_index][agent.unique_id]
                return 
            gen_index += 1
        
        return

    def step(self, shuffle_agents:bool = True):
        for gen in self.generation_list:

        