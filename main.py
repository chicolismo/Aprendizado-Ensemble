from collections import namedtuple
import sys
import csv
import random
import tree as tr
import test_and_training as tat


def read_data(filename):
    '''
    Lê um arquivo csv e retorna os nomes dos atributos e uma lista
    com tuplas de registros.
    '''
    data = []
    fields = None
    with open(filename) as csv_file:
        reader = csv.reader(csv_file, delimiter=';')
        fields = tuple(map(lambda s: s.lower(), next(reader)))
        Data = namedtuple('Data', fields)
        for csv_row in reader:
            print(len(csv_row))
            data.append(Data(*csv_row))
    return (fields, data)


if __name__ == '__main__':

    if len(sys.argv) > 1:
        csv_filename = sys.argv[1]
    else:
        raise Exception('Forneça o nome do arquivo CSV.\nExemplo: python3 main.py data.csv')

    fieldnames, rows = read_data(csv_filename)

    numeric_fields = set()  # Conjunto de campos de valor numérico
    for name in sys.argv[2:]:
        numeric_fields.add(name)

    attributes = fieldnames[:-1]
    # print(fieldnames, rows, numeric_fields)

    # TestData = namedtuple('TestData', attributes)
    # test = TestData(*(rows[0][0:-1]))
    # forest = tr.random_forest(rows, attributes, numeric_fields)
    # voting = tr.majority_voting(forest, test)

    # print(tat.cross_validation(rows, attributes, numeric_fields, 10, 10))
