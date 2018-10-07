# encoding: utf-8

from collections import defaultdict
import math
import random
import test_and_training as tat


class BadPredictionException(Exception):
    pass


class Node:
    def __init__(self, label=None):
        self.label = label
        self.terminal = True
        self.numeric = None
        self.children = []
        self.parent_value = None
        # print(self.label)
        # print(self.parent_value)

    def add_child(self, node, parent_value, numeric):
        # print("LABEL " + str(self.label) + " " + str(self.parent_value) + " " + str(parent_value))
        self.terminal = False
        self.children.append(node)
        node.parent_value = parent_value
        self.numeric = numeric

    # Recebe um Data(tempo='Chuvoso' ... )
    def predict(self, instance):
        # print(self.label)

        if self.terminal:
            return self.label

        for child in self.children:
            if self.numeric:
                # NOTE: Se for numérico mas falhar o if debaixo???
                # print(instance)
                # print(self.label)
                # print(child.parent_value)
                # print(self.terminal)
                # print(self.parent_value)
                if eval(str(getattr(instance, self.label)) + child.parent_value):
                    # print(child.parent_value)
                    return child.predict(instance)
            else:
                if child.parent_value == getattr(instance, self.label):
                    # print(child.parent_value)
                    return child.predict(instance)

        raise BadPredictionException('Não é possível fazer a predição da instância fornecida')



def all_same_class(D):
    length = len(D)

    if length == 1:
        return True

    if length > 1:
        first = D[0]
        for element in D[1:]:
            if element[-1] != first[-1]:
                return False
        return True

    raise Exception('"D" não pode ser uma lista vazia.')

def count_classes(D):
    counter = defaultdict(int)
    for element in D:
        counter[element[-1]] += 1
    return len(counter)

def most_frequent_class(D):
    '''
    Retorna a classe mais frequente de um conjunto de dados D.
    '''
    counter = defaultdict(int)
    for element in D:
        counter[element[-1]] += 1
    return max(counter, key=counter.get)


def is_numeric(data, attr, numeric_indexes):
    '''
    Confere se o atributo está na lista (passada por parâmetro ao programa) de
    atributos de valor numérico
    '''
    attr_index = data[0].index(getattr(data[0], attr))
    return (numeric_indexes is not None) and (attr_index in numeric_indexes)  # Se for contínuo


def divide_numerical_attr(data, attr):
    '''
    Lista os possíveis pontos de corte (média de duas instâncias seguidas, em
    ordem, com classes diferentes)

    Retorna: lista de candidatos a ponto de corte
    '''
    sorted_data = list(sorted(data, key=lambda x: getattr(x, attr)))  # Ordena pelo atributo
    values = []
    # Adiciona a média de dois valores seguidos com classes diferentes como possíveis ponto de corte
    for i in range(len(sorted_data) - 1):
        current_class = sorted_data[i][-1]
        next_class = sorted_data[i + 1][-1]
        if current_class != next_class:
            values.append((float(getattr(sorted_data[i], attr)) +
                           float(getattr(sorted_data[i + 1], attr))) / 2)
    return values


def get_cut_point(D, attr, numeric_indices, values):
    '''
    Escolhe o ponto de corte que resulta na menor entropia
    Retorna: valor numérico que melhor divide o atributo
    '''
    entropy = {}
    for value in values:
        entropy[value] = info(D, attr, numeric_indices, value)
    return min(entropy, key=entropy.get)


def m_random_features(L, m):
    """
    Seleciona m dos L atributos
    """
    features = random.sample(L, m)
    return features


def info(D, attr=None, numeric_indices=None, cutpoint=None):
    """
    Calcula a entropia de D_v.
    Menor = melhor.
    """
    n = len(D)
    entropy = 0

    if attr is not None:
        # Cria um conjunto com todos os valores possíveis para o attributo
        # escolhido
        if is_numeric(D, attr, numeric_indices) and cutpoint != None:

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


def generate_decision_tree(D, L, numeric_indices=None, m=-1):
    """
    Entrada:
        D: Conjunto de dados de treinamento.
        L: Lista de d atributos (rótulos) preditivos em D.
        numeric_indices: Lista de índices das features de valor contínuo
        m = número de atributos a serem selecionados

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
    if L == None or len(L) is 0:
        N.label = most_frequent_class(D)
        return N

    # Senão

    # Se foi decidido selecionar um subconjunto de m atributos
    if m != -1:
        sub_attributes = m_random_features(L, m)
        # print(sub_attributes)
    else:
        sub_attributes = L

    # Calcula a entropia para cada atributo restante em L.
    # O de menor entropia é escolhido.
    original_entropy = info(D)

    entropies = {}
    gains = {}
    for attr in sub_attributes:
        if is_numeric(D, attr, numeric_indices):
            values = divide_numerical_attr(D, attr)
            cutpoint = get_cut_point(D, attr, numeric_indices, values)
            entropies[attr] = info(D, attr, numeric_indices, cutpoint)
            gains[attr] = original_entropy - entropies[attr]
        else:
            entropies[attr] = info(D, attr, numeric_indices)
            gains[attr] = original_entropy - entropies[attr]

    # A = Atributo preditivo em L que apresenta "melhor" critério de divisão.
    # A = min(entropies, key=entropies.get)
    A = max(gains, key=gains.get)

    # print("Entropia: ", entropies)
    # print("Ganhos: ", gains)

    # Associe A ao nó N
    N.label = A

    # Remove o atributo escolhido da lista de atributos
    # L = L - A
    L = tuple(attr for attr in L if attr != A)

    if is_numeric(D, A, numeric_indices):
        values = divide_numerical_attr(D, A)
        cutpoint = get_cut_point(D, A, numeric_indices, values)
        # print("Cutpoint: ", cutpoint)
        subset = [row for row in D if float(getattr(row, A)) <= cutpoint]

        # Se o subconjunto for vazio, associa a classe mais frequente e retorna
        if len(subset) is 0:
            N.label = most_frequent_class(D)
            return N
        # Senão, associa N a uma subárvore gerad por recursão com o subconjunto como entrada
        if m != -1:
            N.add_child(generate_decision_tree(subset, L, numeric_indices,
                                               math.floor(math.sqrt(len(L)))), "<=" + str(cutpoint), True)
        else:
            N.add_child(generate_decision_tree(subset, L, numeric_indices), "<=" + str(cutpoint), True)

        subset = [row for row in D if float(getattr(row, A)) > cutpoint]
        # Se o subconjunto for vazio, associa a classe mais frequente e retorna
        if len(subset) is 0:
            N.label = most_frequent_class(D)
            return N
        # Senão, associa N a uma subárvore gerada por recursão com o subconjunto como entrada
        if m != -1:
            N.add_child(
                generate_decision_tree(subset, L, numeric_indices,
                                       math.floor(math.sqrt(len(L)))), ">" + str(cutpoint), True)
        else:
            N.add_child(generate_decision_tree(subset, L, numeric_indices), ">" + str(cutpoint), True)
    else:
        values = {getattr(row, A) for row in D}

        # Para cada valor em A
        for value in values:
            # Cria um subconjunto D contendo apenas as instâncias com A=valor
            subset = [row for row in D if value in row]

            # Se o subconjunto for vazio, associa a classe mais frequente e retorna
            if len(subset) is 0:
                N.label = most_frequent_class(D)
                return N

            # Senão, associa N a uma subárvore gerada por recursão com o subconjunto como entrada
            if m != -1:
                N.add_child(generate_decision_tree(subset, L, numeric_indices,
                                                   math.floor(math.sqrt(len(L)))), value, False)
            else:
                N.add_child(generate_decision_tree(subset, L, numeric_indices), value, False)
    # Retorna N
    return N


def random_forest(data, attributes, numeric_indices=None, r=10):
    bootstrap_sets = tat.bootstrap(data,r)
    trees = []
    for bootstrap in bootstrap_sets:
        trees.append(
            generate_decision_tree(bootstrap.training,
                                   attributes,
                                   numeric_indices,
                                   math.floor(math.sqrt(len(attributes)))))
    return trees


def majority_voting(trees, element):
    predictions = []

    for tree in trees:
        try:
            prediction = tree.predict(element)
            predictions.append(prediction)
        except BadPredictionException:
            print("Predição não foi possível. Elemento foi ignorado")

    return most_frequent_class(predictions)
