import random
from collections import namedtuple
from collections import defaultdict

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

# TODO: checar se pode ser feito com essa função
def mRandomFeatures(L, m):
    """
    Seleciona m dos L atributos
    """
    features = random.sample(L, m)
    return features

def stratifiedKFold(D, k=10):
    '''
    Separa o conjunto de dados em k partições que mantém a proporção de instâncias por classe em cada fold
    '''
    instancesPerClass = defaultdict(list)
    for instance in D:
        instancesPerClass[instance[-1]].append(instance)
        amountPerClass = defaultdict(list)
    for c in instancesPerClass:
        # Guarda o número (inteiro) de instâncias por fold e o resto da divisão por k
        amountPerClass[c] = [len(instancesPerClass[c])//k,len(instancesPerClass[c])%k]
    folds = [[] for l in range(k)]
    for f in range(k):
        for c in instancesPerClass:
            # Entrega um número igual de instâncias de cada classe para cada fold
            for i in range(amountPerClass[c][0]):
                folds[f].append(instancesPerClass[c].pop())
            # E mais uma instância (dentre as que ficaram de fora da divisão inteira)
            # aos "resto da divisão" primeiros folds
            if amountPerClass[c][1] >= 1:
                folds[f].append(instancesPerClass[c].pop())
                amountPerClass[c][1] = amountPerClass[c][1] - 1
    return folds




