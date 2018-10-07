import random
from collections import namedtuple
from collections import defaultdict
import tree as tr
import math

Bootstrap = namedtuple('Bootstrap', ['training', 'test'])

# TODO: é preciso verificar se os conjutos são diferentes?
def bootstrap(data, r=10):
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


def crossValidation(D, L, numeric_indices=None, k=10, r=10):
    Data = namedtuple('Data', L)
    folds = stratifiedKFold(D, k)
    # print("Número de folds: ", len(folds))
    numOfClasses = tr.count_classes(D)
    # confusionMatrix = [[0 for x in range(numOfClasses)] for y in range(numOfClasses)]
    confusionMatrix = nested_dict(numOfClasses)
    classes = list(set([e[-1] for e in D]))
    # for c in classes:
    #     for c2 in classes:
    #         confusionMatrix[c][c2] = 0
    fmeasures = []
    for current_fold in folds:
        for c in classes:
            for c2 in classes:
                confusionMatrix[c][c2] = 0
        training_data = []
        for fold in folds:
            if fold != current_fold:
                for item in fold:
                    training_data.append(item)

        # print('Training', training_data)
        trees = tr.random_forest(training_data, L, numeric_indices, r)

        for element in current_fold:
            decision = tr.majority_voting(trees, Data(*element[0:-1]))
            confusionMatrix[element[-1]][decision] += 1
            print("DECISAO: " + decision)
        tp, fp, fn = sum_tp_fp_fn(confusionMatrix)
        fmeasures.append(f_measure(1,tp,fp,fn))
    aritmMean = mean(fmeasures)
    stdDeviation = std_deviation(fmeasures)
    return aritmMean, stdDeviation

def nested_dict(n):
    if n == 1:
        return defaultdict(int)
    else:
        return defaultdict(lambda: nested_dict(n-1))

def sum_tp_fp_fn(confusionMatrix, c=None):
    '''Retorna a soma dos verdadeiros positivos, falsos positivos e falsos negativos
    c = classe a ter a soma retornada: se None, faz a soma de todas as classes'''
    tp = 0
    fp = 0
    fn = 0
    if c != None:
        tp = confusionMatrix[c][c]
        for l in confusionMatrix:
            if c != l:
                fp += confusionMatrix[l][c]
                fn += confusionMatrix[c][l]
        return tp, fp, fn
    else:
        for l in confusionMatrix:
            for l2 in confusionMatrix:
                if l != l2:
                    fp += confusionMatrix[l2][l]
                    fn += confusionMatrix[l][l2]
                else:
                    tp += confusionMatrix[l][l2]
        return tp, fp, fn

def precision(truePositives, falsePositives):
    return truePositives/(truePositives+falsePositives)

def recall(truePositives, falseNegatives):
    return truePositives/(truePositives+falseNegatives)

def f_measure(beta, tp, fp, fn):
    b2 = beta*beta
    prec = precision(tp, fp)
    rev = recall(tp, fn)
    return (1 + (b2))*((prec*rev)/((b2*prec)+rev))

def mean(list):
    sum = 0
    for e in list:
        sum += e
    return sum/len(list)

def std_deviation(list):
    m = mean(list)
    sSum = 0
    for e in list:
        sSum += (e - m)*(e - m)
    sSum = sSum/len(list)
    return math.sqrt(sSum)