import random
from collections import namedtuple
from collections import defaultdict
import tree as tr

Bootstrap = namedtuple('Bootstrap', ['training', 'test'])

# TODO: é preciso verificar se os conjutos são diferentes?
def bootstrap(data, r=100):
    """
    Separa o conjunto D em r conjuntos de teste e treino com reposição
    """
    bootstrap_sets = []
    # Cria r bootstraps
    for _ in range(r):
        training_set = []
        # Seleciona n instâncias aleatórias (com repetição) para o conjunto de treino
        for _ in range(len(data)):
            training_set.append(random.choice(data))
        test_set = [inst for inst in data if inst not in training_set]
        bootstrap_sets.append(Bootstrap(training=training_set, test=test_set))
    return bootstrap_sets


def stratifiedKFold(data, k=10):
    '''
    Separa o conjunto de dados em k partições que mantém a proporção de instâncias por classe em cada fold
    '''
    instancesPerClass = defaultdict(list)
    for instance in data:
        instancesPerClass[instance[-1]].append(instance)
    amountPerClass = defaultdict(list)
    for c in instancesPerClass:
        # Guarda o número (inteiro) de instâncias por fold e o resto da divisão por k
        amountPerClass[c] = [len(instancesPerClass[c])//k, len(instancesPerClass[c])%k]
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


def crossValidation(D, L, numeric_indices=None, k=10):
    folds = stratifiedKFold(D, k)
    print("Número de folds: ", len(folds))
    for current_fold in folds:
        training_data = []
        for fold in folds:
            if fold != current_fold:
                for item in fold:
                    training_data.append(item)

        print('Training', training_data)
        trees = tr.random_forest(training_data, L, numeric_indices)

        for element in current_fold:
            print(tr.majority_voting(trees, element))
