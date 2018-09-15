from collections import namedtuple
import sys
import csv
import tree

def read_data(filename):
    """
    Converte um arquivo CVS numa lista de tuplas nomeadas contendo as colunas do CSV
    Retorna uma tupla com os nomes dos atributos e a lista contendo as linhas da tabela.
    """
    data = []
    fieldnames = None
    with open(filename) as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        # Os nomes das colunas serão atributos com letras minúsculas
        fieldnames = tuple(map(lambda s: s.lower(), next(reader)))
        Data = namedtuple('Data', fieldnames)
        for row in reader:
            data.append(Data(*row));
    return (fieldnames, data)


if __name__ == '__main__':

    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        raise Exception('Forneça o nome do arquivo CSV.\nExemplo: python3 main.py data.csv')

    fieldnames, rows = read_data(filename)
    for row in rows:
        print(row)

    if len(sys.argv) > 2: # Array com indices das features contínuas passado como parâmetro
        numericIndexes = []
        for i in range(2,len(sys.argv)):
            numericIndexes.append(int(sys.argv[i]))
    else:
        numericIndexes = None

    tree = tree.generate_decision_tree(rows, fieldnames[:-1], numericIndexes)
    attr = fieldnames[:-1]
    Data = namedtuple('Data', fieldnames[:-1])
    # result = tree.predict(Data(tempo='Nublado', temperatura='Alta', umidade='Alta', ventoso='FALSO'))
    result = tree.predict(Data(tempo='Ensolarado', temperatura=20, umidade=80, ventoso='FALSO'))
    print("Novo valor:")
    print(result)



