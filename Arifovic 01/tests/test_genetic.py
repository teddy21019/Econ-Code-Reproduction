from src.genetic import AGene, AGeneticAlgorithm, EvaluableGene

def test_encode():
    gene = AGene()
    gene_consumption = gene.consumption_1()
    gene_portfolio = gene.portfolio()

    a, b = gene.encode()

    assert a == gene_consumption
    assert b == gene_portfolio