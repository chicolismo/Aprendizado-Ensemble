# encoding: utf-8

from collections import defaultdict
import math
import numbers
from operator import itemgetter
from copy import deepcopy

class Node:
    def __init__(self, label=None):
        self.label = label
        self.terminal = True
        self.numeric = None
        self.children = []
        self.parentValue = None

    def add_child(self, node, parentValue, numeric):
        self.terminal = False
        self.children.append(node)
        node.parentValue = parentValue
        self.numeric = numeric

    def predict(self, instance):
        print(self.label)
        if self.terminal:
            return self.label
        current = instance.index(getattr(instance, self.label))
        for child in self.children:
            if self.numeric == True:
                if eval(str(instance[current]) + child.parentValue):
                    print(child.parentValue)
                    return child.predict(instance)
            else:
                if child.parentValue == instance[current]:
                    print(child.parentValue)
                    return child.predict(instance)

def all_same_class(D):
    length = len(D)
    if length == 1:
        return True
    elif length > 1:
        first = D[0]
        for i, element in enumerate(D[1:]):
            # print(f"Classe do primeiro: {first[-1]}, Classe do elemento {i + 1}: {element[-1]}")
            if element[-1] != first[-1]:
                return False
        return True
    else:
        raise Exception('"D" não pode ser uma lista vazia.')
        # return False


def most_frequent_class(D):
    '''
    Retorna a classe mais frequente de um conjunto de dados D.
    '''
    counter = defaultdict(int)
    for element in D:
        counter[element[-1]] += 1
    return max(counter, key=counter.get)

def isNumeric(D, attr, numericIndexes):
    attrIndex = D[0].index(getattr(D[0], attr))
    if numericIndexes != None and (attrIndex in numericIndexes):  # Se for contínuo
        return True
    else:
        return False

def divideNumericalAttr(D, attr):
    attrIndex = D[0].index(getattr(D[0], attr))
    D2 = sorted(D, key=lambda x: x[attrIndex])  # Ordena pelo atributo
    values = []
    # Adiciona a média de dois valores seguidos com classes diferentes como possíveis ponto de corte
    for i in range(len(D2) - 1):
        currentClass = D2[i][-1]
        nextClass = D2[i + 1][-1]
        if currentClass != nextClass:
            values.append((float(D2[i][attrIndex]) + float(D2[i + 1][attrIndex])) / 2)

    return values


def getCutPoint(D, attr, numericIndexes, values):
    entropy = {}
    for value in values:
        entropy[value] = info(D, attr, numericIndexes, value)
    return min(entropy, key=entropy.get)





# TODO: Verificar se que vai haver algum caso que não se passa nenhuma lista de atributos???

def info(D, attr=None, numericIndexes = None, cutpoint=None):
    """
    Calcula a entropia de D_v.
    Menor = melhor.
    """
    n = len(D)
    entropy = 0

    if attr is not None:
        # Cria um conjunto com todos os valores possíveis para o attributo
        # escolhido
        if isNumeric(D, attr, numericIndexes) and cutpoint != None:

            # Valores menores ou iguais ao ponto de corte
            value_occurrences = 0
            classes_count = defaultdict(int)
            for row in D:
                if float(getattr(row, attr)) <= cutpoint:
                    value_occurrences += 1
                    classes_count[row[-1]] += 1

            probalibility_sum = 0.0
            for class_occurrences in classes_count.values():
                p = (class_occurrences / value_occurrences)
                probalibility_sum -= p * math.log(p, 2)

            # Soma ponderada das probalidades de cada atributo
            value_weight = (value_occurrences / n)
            entropy += value_weight * probalibility_sum

            # valores maiores que o ponto de corte
            value_occurrences = 0
            classes_count = defaultdict(int)
            for row in D:
                if float(getattr(row, attr)) > cutpoint:
                    value_occurrences += 1
                    classes_count[row[-1]] += 1

            probalibility_sum = 0.0
            for class_occurrences in classes_count.values():
                p = (class_occurrences / value_occurrences)
                probalibility_sum -= p * math.log(p, 2)

            # Soma ponderada das probalidades de cada atributo
            value_weight = (value_occurrences / n)
            entropy += value_weight * probalibility_sum
        else:
            values = {getattr(row, attr) for row in D}

            # Para cada valor do atributo, determina a proporção das classes
            for value in values:
                value_occurrences = 0
                classes_count = defaultdict(int)
                for row in D:
                    if getattr(row, attr) == value:
                        value_occurrences += 1
                        classes_count[row[-1]] += 1

                probalibility_sum = 0.0
                for class_occurrences in classes_count.values():
                    p = (class_occurrences / value_occurrences)
                    probalibility_sum -= p * math.log(p, 2)

                # Soma ponderada das probalidades de cada atributo
                value_weight = (value_occurrences / n)
                entropy += value_weight * probalibility_sum

    else:
        class_counter = defaultdict(int) # Quantidade de cada classe em D
        for element in D:
            class_counter[element[-1]] += 1

        for class_occurrences in class_counter.values():
            p = class_occurrences / n  # Probabilidade de um elemento pertencer a classe p_i
            entropy -= p * math.log(p, 2)

    return entropy


def generate_decision_tree(D, L, numericIndexes=None):
    """
    Entrada:
        D: Conjunto de dados de treinamento.
        L: Lista de d atributos (rótulos) preditivos em D.
        numericIndexes: Lista de índices das features de valor contínuo
    Retorna: Árvore de decisão
    """

    # Cria um nodo N
    N = Node()

    # Se todos os exemplos em D possuem a mesma classe y_i,
    # então retorne N como um nó folha rotulado com y_i
    if all_same_class(D):
        N.label = D[0][-1]
        return N

    # Se L é vazia então retorne N como um nó folha rotulado
    # com a classe y_i mais frequente em D.
    if L == None or len(L) == 0:
        N.label = most_frequent_class(D)
        return N

    # Senão

    # Calcula a entropia para cada atributo restante em L.
    # O de menor entropia é escolhido.
    originalEntropy = info(D)

    entropies = {}
    gains = {}
    for attr in L:
        if isNumeric(D, attr, numericIndexes):
            values = divideNumericalAttr(D, attr)
            cutpoint = getCutPoint(D, attr, numericIndexes, values)
            entropies[attr] = info(D, attr, numericIndexes, cutpoint)
            gains[attr] = originalEntropy - entropies[attr]
        else:
            entropies[attr] = info(D, attr, numericIndexes)
            gains[attr] = originalEntropy - entropies[attr]

    # A = Atributo preditivo em L que apresenta "melhor" critério de divisão.
    # A = min(entropies, key=entropies.get)
    A = max(gains, key=gains.get)

    print("Entropia:" + str(entropies))
    print("Ganhos:" + str(gains))

    # Associe A ao nó N
    N.label = A

    # Remove o atributo escolhido da lista de atributos
    # L = L - A
    L = tuple(attr for attr in L if attr != A)

    if isNumeric(D, A, numericIndexes):
        attrIndex = D[0].index(getattr(D[0], A))
        subset = [row for row in D if float(row[attrIndex]) <= cutpoint]
        # Se o subconjunto for vazio, associa a classe mais frequente e retorna
        if not len(subset):
            N.label = most_frequent_class(D)
            return N
        # Senão, associa N a uma subárvore gerad por recursão com o subconjunto como entrada
        N.add_child(generate_decision_tree(subset, L, numericIndexes), "<=" + str(cutpoint), True)

        subset = [row for row in D if float(row[attrIndex]) > cutpoint]
        # Se o subconjunto for vazio, associa a classe mais frequente e retorna
        if not len(subset):
            N.label = most_frequent_class(D)
            return N
        # Senão, associa N a uma subárvore gerad por recursão com o subconjunto como entrada
        N.add_child(generate_decision_tree(subset, L, numericIndexes), ">" + str(cutpoint), True)

    else:
        values = { getattr(row, A) for row in D }

        # Para cada valor em A
        for value in values:
            # Cria um subconjunto D contendo apenas as instâncias com A=valor
            subset = [row for row in D if value in row]

            # Se o subconjunto for vazio, associa a classe mais frequente e retorna
            if not len(subset):
                N.label = most_frequent_class(D)
                return N
            # Senão, associa N a uma subárvore gerad por recursão com o subconjunto como entrada
            N.add_child(generate_decision_tree(subset, L, numericIndexes), value, False)

    # Retorna N
    return N




