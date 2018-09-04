# encoding: utf-8

from collections import defaultdict
import math

class Node:
    def __init__(self, label=None):
        self.label = label
        self.terminal = True
        self.children = []

    def add_child(self, node):
        self.terminal = False
        self.children.append(node)


def gain():
    """
    Retorna o ganho do atributo
    """
    pass


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


def info(D):
    """
    Calcula a entropia de D
    """

    counter = defaultdict(int) # Quantidade de cada classe em D
    for element in D:
        counter[element[-1]] += 1

    # Entropy = - Σ(p_i * log2(pi))
    n = len(D)
    entropy = 0
    for n_class in counter.values():
        p = n_class / n  # Probabilidade de um elemento pertencer a classe p_i
        entropy -= p * math.log(p, 2)

    return entropy


# TODO: Continuar aqui...
def best_prediction_attribute(D, L):
    counter = defaultdict(int)
    for label in L:
        pass


def generate_decision_tree(D, L=None):
    """
    Entrada:
        D: Conjunto de dados de treinamento.
        L: Lista de d atributos (rótulos) preditivos em D.
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
    # A = Atributo preditivo em L que apresenta "melhor" critério de divisão.

    # A = best_prediction_attribute(D, L)

    # Associe A ao nó N

    # L = L - A

    # Para cada valor v distinto do atributo A, considrendo os exemplos em D, faça:
        # D_v = subconjunto dos dados de treinamento em que A = v
        # Se D_v vazio, então retorne N como um nó folha rotulado com a class y_i mais frequente em D_v
        # Senão, associe N a uma subárvore retornada por "generate_decision_tree(D_v, L)"

    # Retorne N




