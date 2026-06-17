"""A small binary-chromosome genetic algorithm for the P2 WSN problem.

The GA is the *data source* for Method 1 (information-gain / diminishing
returns over inter-sample differences) and Method 4 (solution-quality gap vs
MILP optimum).  It optimises the same binary chromosome ``B ∈ {0,1}^N`` the
MILP uses, scored by the SimLab surrogate fitness.
"""
