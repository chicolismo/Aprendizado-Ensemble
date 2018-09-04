from collections import namedtuple
import sys
import csv


def read_csv(filename):
    """
    Converte um arquivo CVS numa lista de tuplas cujos campos t
    """
    data = []
    with open(filename) as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        # Os nomes das colunas serão atributos com letras minúsculas
        fieldnames = map(lambda s: s.lower(), next(reader))
        Kls = namedtuple(name, fieldnames)
        for row in reader:
            data.append(Kls(*row));
    return data


if __name__ == '__main__':

    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        raise Exception('Forneça o nome do arquivo CSV')


    rows = read_data('Condition', filename)
    for row in rows:
        print(row[-1])
