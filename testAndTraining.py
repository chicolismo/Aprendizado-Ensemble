import random
from collections import namedtuple

# TODO: é preciso verificar se os cojnutos são diferentes?
def bootstrap(D, r=100):
    """
    Separa o conjunto D em r conjuntos de teste e treino com reposição
    """
    n = len(D)
    bootstrapSets = []
    # Cria r bootstraps
    for i in range(r):
        trainingSet = []
        # Seleciona n instâncias aleatórias (com repetição) para o conjunto de treino
        for j in range(n):
           index = random.randint(0, n-1)
           trainingSet.append(D[index])
        testSet = {inst for inst in D if inst not in trainingSet}
        Bootstrap = namedtuple('Bootstrap', ['training', 'test'])
        bootstrapSets.append(Bootstrap(training=trainingSet, test=testSet))
    return bootstrapSets


