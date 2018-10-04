from collections import namedtuple
import sys
import csv
import tree as tr
import test_and_training

def read_data(filename):
    """
    Converte um arquivo CVS numa lista de tuplas nomeadas contendo as colunas do CSV
    Retorna uma tupla com os nomes dos atributos e a lista contendo as linhas da tabela.
    """
    data = []
    fields = None
    with open(filename) as csvfile:
        reader = csv.reader(csvfile, delimiter=';')

        # Os nomes das colunas serão atributos com letras minúsculas
        fields = tuple(map(lambda s: s.lower(), next(reader)))
        Data = namedtuple('Data', fields)
        for csv_row in reader:
            data.append(Data(*csv_row))
    return (fields, data)


if __name__ == '__main__':

    if len(sys.argv) > 1:
        csv_filename = sys.argv[1]
    else:
        raise Exception('Forneça o nome do arquivo CSV.\nExemplo: python3 main.py data.csv')

    fieldnames, rows = read_data(csv_filename)

    if len(sys.argv) > 2: # Array com indices das features contínuas passado como parâmetro
        numeric_indices = []
        for number in sys.argv[2:]:
            numeric_indices.append(int(number))
    else:
        numeric_indices = None

    # Monta uma árvore de decisão
    # print(rows)
    attributes = fieldnames[:-1]
    tree = tr.generate_decision_tree(rows, attributes, numeric_indices)
    Data = namedtuple('Data', attributes)

    # Faz a predição de um novo valor
    try:
        # result = tree.predict(Data(tempo='Nublado', temperatura='Amena', umidade='Alta', ventoso='Falso'))
        result = tree.predict(Data(tempo='Ensolarado', temperatura=20, umidade=80, ventoso='Falso'))
        print("Novo valor:")
        print(result)
    except tr.BadPredictionException:
        print("Não foi possível predizer o elemento.")

    # Demonstrações de funcionamento:

    # Bootstrap: printa os r (default 100) conjuntos de teste e treino gerados
    print(test_and_training.bootstrap(rows, 5))

    # Escolha de m features: printa uma escolha aleatória de duas features dentre as existentes
    print(tr.m_random_features(attributes, 2))

    # Divisão em K folds estratificados: printa os k (default 10) folds estratificados gerados
    print(test_and_training.stratifiedKFold(rows, 3))

    # tr.randomForest(rows, fieldnames[:-1], numeric_indices)

    test_rows = []
    for row in rows:
        test_rows.append(Data(*row[0:-1]))

    test_and_training.crossValidation(test_rows, attributes, numeric_indices, 4)
