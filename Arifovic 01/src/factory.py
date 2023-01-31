from typing import List, Tuple, Type, Dict, Callable
from src.base.GA import BaseGene
from functools import reduce 

DecodeFunction = Callable[[BaseGene, str], Tuple[float, int]]

def gene_factory(
        decoding_list:List[Tuple[int, str]], 
        gene_class:Type[BaseGene]) -> Tuple[Type[BaseGene], DecodeFunction]:
    
    """A gene factory that returns a gene class generator 
    corresponding to the given decoding list and a decoder 
    function.

    Parameters
    -----
    decoding_list:List[Tuple[int,str]] 
    >>> [
        (20, "consumption_1"),
        (10, "portfolio_1")
    ]

    gene_class : Type[BaseGene] 
    A subclass of BaseGene that can be instantiated. 
    This class should be ready to extend to different number of genes.

    Returns
    -----
    CustomGene class : subclass of `gene_class`. 
    All implementations inherit from `gene_class` except for 
    the length of the gene.

    decode :  Callable[[BaseGene, str], Tuple[float, int]]
    A customized decoding function that takes in the specific 
    gene class returned along in the tuple, and a string of 
    the corresponding segment. 
    The string should be in the set of the decoding list, 
    or else a ValueError will be raised.

    """

    from inspect import isabstract
    if isabstract(gene_class):
        raise TypeError("gene_class must be a subclass of BaseGene that implements methods")

    total_gene_len: int = reduce(lambda dl1, dl2 : dl1 + dl2[0], decoding_list, 0)

    segment_dict : Dict[str, Tuple[int, int]]= {}
    start = 0
    for item in decoding_list:
        end = start + item[0]
        segment_dict[item[1]] = (start, end)
        start = end

    
    class CustomGene(gene_class):
        """
            Custom gene that is created by gene_factory. It is paired with a `decode` function.
        """
        N:int = total_gene_len

    def decode(gene: BaseGene, segment_name: str) -> Tuple[float, int]:
        """
        A decoder function that takes a BaseGene and a segment name and returns the decoded number and the number of string it has.
        """
        import numpy as np

        if not isinstance(gene, CustomGene):
            raise TypeError("gene must be a CustomGene defined by the gene_factory function!")

        if segment_name not in segment_dict:
            raise ValueError(f"{segment_name} is not a valid segment")
        start_pos, end_pos = segment_dict[segment_name]

        segment_string: np.ndarray = gene.string[start_pos:end_pos]
        n = end_pos - start_pos
        return segment_string.dot(1 << np.arange(n)), n


    return CustomGene, decode

