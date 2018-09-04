from collections import namedtuple
import sys
import csv


def read_data(filename):
    """
    Converte um arquivo CVS numa lista de tuplas nomeadas contendo as colunas do CSV
    """
    data = []
    with open(filename) as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        # Os nomes das colunas serão atributos com letras minúsculas
        fieldnames = map(lambda s: s.lower(), next(reader))
        Data = namedtuple('Data', fieldnames)
        for row in reader:
            data.append(Data(*row));
    return data


if __name__ == '__main__':

    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        raise Exception('Forneça o nome do arquivo CSV.\nExemplo: python3 main.py data.csv')


    rows = read_data(filename)
    for row in rows:
        print(row)
