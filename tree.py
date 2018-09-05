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


def info(D, attr=None):
    """
    Calcula a entropia de D_v.
    Menor = melhor.
    """
    n = len(D)
    entropy = 0

    if attr is not None:
        # Cria um conjunto com todos os valores possíveis para o attributo
        # escolhido
        values = set()
        for element in D:
            values.add(getattr(element, attr))

        # Para cada valor do atributo, determina a proporção das classes
        for value in values:
            value_count = 0
            classes_count = defaultdict(int)
            for element in D:
                if getattr(element, attr) == value:
                    value_count += 1
                    classes_count[element[-1]] += 1
            value_sum = 0.0
            for class_count in classes_count.values():
                p = (class_count / value_count)
                value_sum -= p * math.log(p, 2)

            # Soma ponderada das probalidades de cada atributo
            entropy += (value_count / n) * value_sum

    else:
        counter = defaultdict(int) # Quantidade de cada classe em D
        for element in D:
            counter[element[-1]] += 1
        for n_class in counter.values():
            p = n_class / n  # Probabilidade de um elemento pertencer a classe p_i
            entropy -= p * math.log(p, 2)

    return entropy



def generate_decision_tree(D, L):
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

    # Calcula a entropia para cada atributo restante em L.
    # O de menor entropia é escolhido.
    entropies = {}
    for attr in L:
        entropies[attr] = info(D, attr)

    # A = Atributo preditivo em L que apresenta "melhor" critério de divisão.
    A = min(entropies, key=entropies.get)

    print(entropies)

    # Associe A ao nó N
    N.label = A

    # Remove o atributo escolhido da lista de atributos
    # L = L - A
    L = tuple(attr for attr in L if attr != A)


    # TODO: Continuar aqui...

    # Para cada valor v distinto do atributo A, considrendo os exemplos em D, faça:
        # D_v = subconjunto dos dados de treinamento em que A = v
        # Se D_v vazio, então retorne N como um nó folha rotulado com a class y_i mais frequente em D_v
        # Senão, associe N a uma subárvore retornada por "generate_decision_tree(D_v, L)"

    # Retorne N




