from collections import namedtuple
import sys
import csv
import tree as tr
import testAndTraining

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
    # for row in rows:
        # print(row)

    if len(sys.argv) > 2: # Array com indices das features contínuas passado como parâmetro
        numericIndexes = []
        for i in range(2,len(sys.argv)):
            numericIndexes.append(int(sys.argv[i]))
    else:
        numericIndexes = None

    #Monta uma árvore de decisão
    print(rows)
    attr = fieldnames[:-1]
    tree = tr.generate_decision_tree(rows, attr, numericIndexes)
    Data = namedtuple('Data', attr)

    #Faz a predição de um novo valor
    result = tree.predict(Data(tempo='Nublado', temperatura='Amena', umidade='Alta', ventoso='Falso'))
    # result = tree.predict(Data(tempo='Ensolarado', temperatura=20, umidade=80, ventoso='FALSO'))
    print("Novo valor:")
    print(result)

    #Demosntrações de funcionamento:

    #Bootstrap: printa os r (default 100) conjuntos de teste e treino gerados
    print(testAndTraining.bootstrap(rows, 5))
    #Escolha de m features: printa uma escolha aleatória de duas features dentre as existentes
    print(tr.mRandomFeatures(attr, 2))
    #Divisão em K folds estratificados: printa os k (default 10) folds estratificados gerados
    print(testAndTraining.stratifiedKFold(rows, 3))

    # tr.randomForest(rows, fieldnames[:-1], numericIndexes)

    test_rows = []
    for row in rows:
        test_rows.append(Data(*row[0:-1]))
    testAndTraining.crossValidation(test_rows, attr, numericIndexes, 4)



