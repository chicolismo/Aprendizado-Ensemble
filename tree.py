# encoding: utf-8

from collections import defaultdict
import math
import random
import test_and_training

RANDOM = random.Random(123)

class BadPredictionException(Exception):
    '''
    Representa um erro de predição
    '''
    pass


class Node:
    def __init__(self, label=None):
        self.label = label
        self.terminal = True
        self.numeric = False
        self.children = []
        self.parent_value = None

    def add_child(self, child, parent_value):
        self.terminal = False
        child.parent_value = parent_value
        self.children.append(child)

    def predict(self, instance):
        if self.terminal:
            return self.label
        
        for child in self.children:
            try:
                if self.numeric:
                    if eval(getattr(instance, self.label) + child.parent_value):
                        return child.predict(instance)
                else:
                    if child.parent_value == getattr(instance, self.label):
                        return child.predict(instance)
            except AttributeError:
                print(child)

        raise BadPredictionException('Não é possível fazer a predição da instância fornecida')

    def print(self, level=0):
        sep = '    '
        if self.parent_value:
            print(sep * level + self.parent_value)
            level += 1
            print(sep * level + self.label)
        else:
            print(sep * level + self.label)

        for child in self.children:
            child.print(level + 1)

def all_same_class(data):
    length = len(data)

    if length is 1:
        return True

    if length > 1:
        first_class = data[0][-1]
        for element in data[1:]:
            if element[-1] != first_class:
                return False
        return True

    raise Exception('"data" não pode ser uma lista vazia')


def count_classes(data):
    counter = {}
    for row in data:
        counter[row[-1]] = True
    return len(counter)


def most_frequent_class(data):
    '''
    Retorna a class mais frequente de um conjunto de dados.
    '''
    counter = defaultdict(int)
    for row in data:
        counter[row[-1]] += 1
    return max(counter, key=counter.get)


def divide_numeric_attr(data, attr):
    '''
    Lista os possíveis pontos de corte (média de duas instâncias seguidas,
    em ordem, com classes diferentes)

    Retorna: lista de candidatos a ponto de corte
    '''
    # Ordena pelo atributo
    sorted_data = list(sorted(data, key=lambda row: getattr(row, attr)))

    values = []
    # Adiciona média de dois valores seguidos com classes diferentes como
    # possíveis pontos de corte
    for a, b in zip(sorted_data[0:], sorted_data[1:]):
        if a[-1] != b[-1]:
            values.append((float(getattr(a, attr)) + float(getattr(b, attr))) / 2)
    return values


def get_cut_point(data, attr, values):
    '''
    Escolhe o ponto de corte que resulta na menor entropia
    Retorna: valor numérico que melhor divide o atributo
    '''
    entropy = {}
    for value in values:
        entropy[value] = info(data, attr, True, value)
    return min(entropy, key=entropy.get)


def m_random_features(attributes, m):
    '''
    Seleciona m atributos
    '''
    return RANDOM.sample(attributes, m)


def info(data, attr=None, numeric=False, cut_point=None):
    '''
    Calcula a entropia de 'attr'
    Menor = melhor
    '''

    n = len(data)
    entropy = 0

    if not attr is None:
        # Cria um conjunto com todos os valores possíveis para o
        # atributo escolhido

        # Se for atributo numérico
        if numeric:
            # Valores menores ou iguais ao ponto de corte
            value_occurrences = 0
            classes_count = defaultdict(int)
            for row in data:
                if float(getattr(row, attr)) <= cut_point:
                    value_occurrences += 1
                    classes_count[row[-1]] += 1

            probability_sum = 0.0
            for class_occurrences in classes_count.values():
                p = (class_occurrences / value_occurrences)
                probability_sum -= p * math.log(p, 2)

            # Soma ponderada das probabilidades de cada atributo
            value_weight = (value_occurrences / n)
            entropy += value_weight * probability_sum

            # Valores maiores ou iguais ao ponto de corte
            value_occurrences = 0
            classes_count = defaultdict(int)
            for row in data:
                if float(getattr(row, attr)) > cut_point:
                    value_occurrences += 1
                    classes_count[row[-1]] += 1

            probability_sum = 0.0
            for class_occurrences in classes_count.values():
                p = (class_occurrences / value_occurrences)
                probability_sum -= p * math.log(p, 2)

            # Soma ponderada das probabilidades de cada atributo
            value_weight = (value_occurrences / n)
            entropy += value_weight * probability_sum

        else: # Não é numérico
            values = {getattr(row, attr) for row in data}
            for value in values:
                value_occurrences = 0
                classes_count = defaultdict(int)
                for row in data:
                    if getattr(row, attr) == value:
                        value_occurrences += 1
                        classes_count[row[-1]] += 1

                probability_sum = 0.0
                for class_occurrences in classes_count.values():
                    p = (class_occurrences / value_occurrences)
                    probability_sum -= p * math.log(p, 2)

                value_weight = (value_occurrences / n)
                entropy += value_weight * probability_sum

    else: # Nenhum atributo informado
        class_counter = defaultdict(int)
        for row in data:
            class_counter[row[-1]] += 1
        for class_occurrences in class_counter.values():
            p = class_occurrences / n
            entropy -= p * math.log(p, 2)

    return entropy


def generate_decision_tree(data, attributes, numeric_fields=None, m = -1):
    """
    Entrada:
        data: Conjunto de dados de treinamento.
        attributes: Lista de d atributos (rótulos) preditivos em D.
        numeric_fields: Lista de nomes das features de valor contínuo
        m = número de atributos a serem selecionados

    Retorna: Árvore de decisão
    """
    node = Node()

    if numeric_fields is None:
        numeric_fields = set()

    # Se todas as classes forem iguais, retorna o nodo com o rótulo igual ao
    # nome da classe
    if all_same_class(data):
        node.label = data[0][-1]
        return node

    # Se attributes for vazio, retorna o nodo como um nó folha rotulado
    # com a classe mais frequente em data.
    if attributes is None or len(attributes) is 0:
        node.label = most_frequent_class(data)
        return node

    # Senão

    # Se foi decidido selecionar um subconjunto de m atributos
    if not m is -1:
        sub_attributes = m_random_features(attributes, m)
    else:
        sub_attributes = attributes

    original_entropy = info(data)

    entropies = {}
    gains = {}
    for attr in sub_attributes:
        if attr in numeric_fields:
            values = divide_numeric_attr(data, attr)
            cut_point = get_cut_point(data, attr, values)
            entropies[attr] = info(data, attr, True, cut_point)
        else:
            entropies[attr] = info(data, attr, False)

        gains[attr] = original_entropy - entropies[attr]
        # gains[attr] = -entropies[attr]


    # Atributo preditivo que representa "melhor" critério de divisão.
    predictive_attr = max(gains, key=gains.get)

    print(f'Ganho do atributo {predictive_attr}: ', gains[predictive_attr])

    node.label = predictive_attr

    # TODO: Tem que ver se na escolha do atributo devemos levar em consideração todos
    # eles ou apenas os subatributos

    # Remove o atributo preditivo da lista de atributos
    attributes = tuple(attr for attr in attributes if attr != predictive_attr)

    if predictive_attr in numeric_fields: # É numérico
        node.numeric = True

        values = divide_numeric_attr(data, predictive_attr)
        cut_point = get_cut_point(data, predictive_attr, values)

        subset = [row for row in data
            if float(getattr(row, predictive_attr)) <= cut_point]

        # Se o subconjunto for vazio, associa a classe mais frequente e retorna
        if not len(subset):
            node.label = most_frequent_class(data)
            return node

        # Senão associa n a uma subárvore gerada por recursão com o subconjunto
        # como entrada
        if not m is -1:
            node.add_child(
                generate_decision_tree(
                    subset,
                    attributes,
                    numeric_fields,
                    math.floor(math.sqrt(len(attributes)))),
                '<=' + str(cut_point))
        else:
            node.add_child(
                generate_decision_tree(
                    subset,
                    attributes,
                    numeric_fields),
                '<=' + str(cut_point))

        subset = [row for row in data
            if float(getattr(row, predictive_attr)) > cut_point]

        # Se o conjunto for vazio, associa a classe mais frequente e retorna
        if not len(subset):
            node.label = most_frequent_class(data)
            return node

        if not m is -1:
            node.add_child(
                generate_decision_tree(
                    subset,
                    attributes,
                    numeric_fields,
                    math.floor(math.sqrt(len(attributes)))),
                '>' + str(cut_point))
        else:
            node.add_child(
                generate_decision_tree(
                    subset,
                    attributes,
                    numeric_fields),
                '>' + str(cut_point))

    else: # Não é numérico
        values = {getattr(row, predictive_attr) for row in data}

        # Para cada valor do atributo preditivo
        for value in values:
            # Cria um subconjunto "data" contendo apenas as instâncias com
            # predictive_attr = value
            subset = [row for row in data if getattr(row, predictive_attr) == value]

            # TODO: Verificar se isso pode acontecer!
            if not len(subset):
                node.label = most_frequent_class(data)
                return node

            if not m is -1:
                node.add_child(
                    generate_decision_tree(
                        subset,
                        attributes,
                        numeric_fields,
                        math.floor(math.sqrt(len(attributes)))),
                    value)
            else:
                node.add_child(
                    generate_decision_tree(
                        subset,
                        attributes,
                        numeric_fields),
                    value)
    return node


def random_forest(data, attributes, numeric_fields=None, r=10):
    bootstrap_sets = test_and_training.bootstrap(data, r)
    trees = []
    sqrt = math.floor(math.sqrt(len(attributes)))
    for bootstrap in bootstrap_sets:
        trees.append(
            generate_decision_tree(
                bootstrap.training,
                attributes,
                numeric_fields,
                sqrt))
    return trees


def most_frequent_element(array):
    counter = defaultdict(int)
    for element in array:
        counter[element] += 1
    return max(counter, key=counter.get)


def majority_voting(trees, instance):
    predictions = []
    for tree in trees:
        try:
            predictions.append(tree.predict(instance))
        except BadPredictionException:
            print('Predição não foi possível. Elemento foi ignorado')
    return most_frequent_element(predictions)


