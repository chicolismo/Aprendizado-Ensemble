import random
import math
import statistics
import tree as tr
from collections import namedtuple, defaultdict

Bootstrap = namedtuple('Bootsrap', ('training', 'test'))

RANDOM = random.Random(123)

def bootstrap(data, r=10):
    '''
    Separa o conjunto 'data' em r conjuntos de teste e treino com reposição
    '''
    bootstrap_sets = []
    for _ in range(r):
        training_set = []

        # Seleciona n instâncias aleatórias (com repetição)
        # para o conjunto de treinamento
        for _ in range(len(data)):
            training_set.append(RANDOM.choice(data))

        test_set = [row for row in data if row not in training_set]

        bootstrap_sets.append(
            Bootstrap(training=training_set, test=test_set))

    return bootstrap_sets


def stratified_k_fold(data, k=10):
    '''
    Separa o conjunto de dados em k partições que mantém a proporção
    de instâncias por classe em cada fold
    '''
    instances_per_class = defaultdict(list)
    for instance in data:
        instances_per_class[instance[-1]].append(instance)

    amount_per_class = {}
    for class_name, instances in instances_per_class.items():
        # Guarda o número (inteiro) de instâncias for fold
        # e o resto da divisão por k
        size = len(instances)
        amount_per_class[class_name] = [size // k, size % k]

    folds = [[] for _ in range(k)]
    for fold in folds:
        for class_name, instances in instances_per_class.items():
            # Entrega um número igual de instâncias de cada class para cada
            # fold
            for _ in range(amount_per_class[class_name][0]):
                fold.append(instances.pop())

            # E mais uma instância (dentre as que ficaram de fora da
            # divisão inteira)
            if amount_per_class[class_name][1] >= 1:
                fold.append(instances.pop())
                amount_per_class[class_name][1] -= 1
    return folds


def cross_validation(data, attributes, numeric_fields=None, k=10, r=10):
    Data = namedtuple('Data', attributes)
    folds = stratified_k_fold(data, k)
    n_classes = tr.count_classes(data)
    confusion_matrix = nested_dict(n_classes)
    classes = list(set(row[-1] for row in data))
    fmeasures = []
    for current_fold in folds:
        for c1 in classes:
            for c2 in classes:
                confusion_matrix[c1][c2] = 0
        training_data = []
        for fold in folds:
            if fold != current_fold:
                for item in fold:
                    training_data.append(item)
        trees = tr.random_forest(training_data, attributes, numeric_fields, r)

        for element in current_fold:
            decision = tr.majority_voting(trees, Data(*element[0:-1]))
            print("Decisão: ", decision)
            confusion_matrix[element[-1]][decision] += 1
        tp, fp, fn = sum_tp_fp_fn(confusion_matrix)
        fmeasures.append(f_measure(1, tp, fp, fn))
    return statistics.mean(fmeasures), statistics.stdev(fmeasures)


def nested_dict(n):
    if n is 1:
        return defaultdict(int)
    else:
        return defaultdict(lambda: nested_dict(n - 1))


def sum_tp_fp_fn(confusion_matrix, target_class=None):
    '''
    Retorna a soma dos verdadeirs positivos, falsos positivos, e falsos
    negativos.

    target_class = classe que se quer a soma. Se for None, faz a soma
    de todas as classes.
    '''
    tp = 0
    fp = 0
    fn = 0
    if not target_class is None:
        tp = confusion_matrix[target_class][target_class]
        for class_name in confusion_matrix:
            if target_class != class_name:
                fp += confusion_matrix[class_name][target_name]
                fn += confusion_matrix[target_name][class_name]
    else:
        for c1 in confusion_matrix:
            for c2 in confusion_matrix:
                if c1 != c2:
                    fp += confusion_matrix[c2][c1]
                    fn += confusion_matrix[c1][c2]
                else:
                    tp += confusion_matrix[c1][c2]
    return tp, fp, fn


def precision(tp, fp):
    return tp / (tp + fp)


def recall(tp, fn):
    return tp / (tp + fn)


def f_measure(beta, tp, fp, fn):
    try:
        b2 = beta**2
        prec = precision(tp, fp)
        rec = recall(tp, fp)
        return (1 + b2) * ((prec * rec) / (b2 * prec + rec))

    except ZeroDivisionError:
        return 0
