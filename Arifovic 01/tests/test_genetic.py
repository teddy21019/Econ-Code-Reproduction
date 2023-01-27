import random
import numpy as np
import pytest
from src.base.GA import BaseGene
from src.genetic import AGene, AGeneticAlgorithm, EvaluableGene, family_sort_with_generation

AGENT_N = 1000


def eval_func(gene:BaseGene):
    try:
        return np.sum(gene.string)
    except Exception as e:
        print(gene)
        raise e

class WrongGA_Gene(AGene):

    def __init__(self, string: np.ndarray = None):
        super().__init__(string)
    
    @classmethod
    def validate(cls, str:np.ndarray) :
        return False

@pytest.fixture
def ga_sample():

    agent_to_register = [EvaluableGene(AGene(), random.randint(1,10)) for _ in range(AGENT_N)]
    ga = AGeneticAlgorithm() 
    ga.set_evaluation_function(eval_func)


    ga.register_agents(agent_to_register)

    return ga

def test_encode():
    gene = AGene()
    gene_consumption = gene.consumption_1()
    gene_portfolio = gene.portfolio()

    a, b = gene.encode()

    assert a == gene_consumption
    assert b == gene_portfolio

def test_create_gene_list():
    agent_to_register = [EvaluableGene(AGene()) for _ in range(10)]
    ga = AGeneticAlgorithm() 


    ga.register_agents(agent_to_register)



def test_add_new_agent():

    ga = AGeneticAlgorithm() 


    agents_to_register = [EvaluableGene(AGene()) for _ in range(20)]

    ga.register_agents(agents_to_register)

    new_agents_to_register = [EvaluableGene(AGene()) for _ in range(10)]
    ga.register_agents(new_agents_to_register)

    assert len(ga.agents) == 30

def test_add_wrong_agent():
    ga = AGeneticAlgorithm()

    agents_to_register = None
    with pytest.raises(TypeError):
        agents_to_register = [EvaluableGene(WrongGA_Gene('111')) for _ in range(20)]

    ga.register_agents(agents_to_register)


def test_create_gene():
    valid_string = np.random.rand(30) > 0.8 
    try :
        AGene(valid_string)
    except TypeError as tper:
        assert False


def test_crossover():
    for _ in range(10):
        gene_all_true = AGene(np.array([True]*30))
        gene_all_false = AGene(np.array([False]*30))


        g3, g4 = gene_all_true * gene_all_false 

        assert (g3.string | g4.string == np.array([True]*30)).all()
        assert g3.string.sum() + g4.string.sum() == 30
    

def test_remove_agent():
    
    ga = AGeneticAlgorithm() 


    agents_to_register = [EvaluableGene(AGene()) for _ in range(20)]

    ga.register_agents(agents_to_register)


    ga.remove_agents(agents_to_register[-3:])

    assert len(ga.agents) == 17
    assert agents_to_register[-1] not in ga.agents
    assert agents_to_register[0] in ga.agents

def test_ga_same_gene_diff_obj():

    ga = AGeneticAlgorithm() 


    agents_to_register = [EvaluableGene(AGene()) for _ in range(20)]

    ga.register_agents(agents_to_register)

    same_gene_ga = AGene(agents_to_register[0].gene.string)

    assert same_gene_ga not in ga.agents
    assert agents_to_register[0] in ga.agents


def test_reproduction_stage(ga_sample: AGeneticAlgorithm):
    def get_fitness_sum(l):
        return sum([agent.fitness for agent in l])
    pre_fitness_sum = get_fitness_sum(ga_sample.agents)

    ga_sample.reproduction_stage()

    post_fitness_sum = get_fitness_sum(ga_sample.winner_agents)
    assert pre_fitness_sum < post_fitness_sum

def test_crossover_stage(ga_sample: AGeneticAlgorithm):

    ga_sample.reproduction_stage()
    ga_sample.crossover_stage()

    with pytest.raises(AttributeError) : 
        ga_sample.winner_agents
    
    assert len(ga_sample.families) - len(ga_sample.agents) <= 1
    assert len(random.sample(ga_sample.families, len(ga_sample.families))[0]) == 4
    assert len([offspring for offspring in ga_sample.families[-1] if offspring.fitness == 0])


def test_mutate(ga_sample: AGeneticAlgorithm):
    ga_sample.p_mut = 1

    ga_sample.reproduction_stage()
    ga_sample.crossover_stage()

    first_family = ga_sample.families[0]

    ga_sample.mutation_stage()

    mutated_first_family = ga_sample.families[0]
 
    assert not np.all(first_family[2].gene.string == mutated_first_family[2].gene.string)
    assert np.sum(first_family[3].gene.string != mutated_first_family[3].gene.string) == 1

def test_family_sort_with_generation():
    mom = EvaluableGene(AGene(), 20)
    dad = EvaluableGene(AGene(), 15)
    son = EvaluableGene(AGene(), 20)
    dau = EvaluableGene(AGene(), 17)

    remaining = family_sort_with_generation([mom, dad, son, dau])
    

    assert (mom.gene.string == remaining[1].gene.string).all()
    assert (son.gene.string == remaining[0].gene.string).all()

def test_whole_process_num(ga_sample: AGeneticAlgorithm):

    ga_sample.reproduction_stage()
    ga_sample.crossover_stage()
    ga_sample.mutation_stage()
    ga_sample.election_stage()

    assert len(ga_sample.gene_pool) == AGENT_N

