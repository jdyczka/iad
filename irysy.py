trainFilePath = 'assets/data_train.csv'
testFilePath = 'assets/data_test.csv'

import csv
import math
import operator
# from heapq import nsmallest
from collections import Counter 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import table

def main():
    trainData = list(csv.reader(open(trainFilePath), quoting=csv.QUOTE_NONNUMERIC))
    testData = list(csv.reader(open(testFilePath), quoting=csv.QUOTE_NONNUMERIC))
    # trainDataSmall = [[5.1,3.5,1.4,0.2,0]]
    # testDataSmall = [[4.9,3.0,1.4,0.2,0]]

    kValues = [1, 3, 5, 7, 9, 11]
    # kValues = [3]

    handleCase(trainData, testData, kValues, [0,1,2,3])

    handleCase(trainData, testData, kValues, [0,1,2])
    handleCase(trainData, testData, kValues, [1,2,3])
    handleCase(trainData, testData, kValues, [0,1])
    handleCase(trainData, testData, kValues, [1,3])


def handleCase(trainData, testData, kValues, properties):
    errors = processAndReturnErrors(trainData, testData, kValues, properties)
    accuracy = []
    
    testDataCount = len(testData)
    for e in errors:
        accuracy.append(getAccuracy(len(e), testDataCount))

    worst = accuracy.index(min(accuracy))
    counts = Counter(list(map(lambda d: int(d[4]), testData)))
    propStr = str(list(map(lambda p: p + 1, properties)))
    
    plotBarChart(kValues, accuracy, propStr)
    plotConfusionMatrix(errors[worst], counts, propStr)


def plotBarChart(x, y, propStr):
    
    indX = np.arange(len(x))

    fig, ax = plt.subplots()
    rects1 = ax.bar(indX, y, 0.35, color='SkyBlue')

    ax.set_title('Wykres dla zestawu cech ' + propStr)

    ax.set_xlabel('k')
    ax.set_xticks(indX)
    ax.set_xticklabels(x)

    ax.set_ylabel('Dokładność (%)')
    ax.set_ylim([min(y) - 1, max(y) + 1])
    ax.yaxis.grid(True)

    plt.savefig('charts/barchart' + propStr + '.png')


def getAccuracy(errors, count):
    return round((count - errors) / count * 100, 0)


def getConfusionMatrix(errors, counts):
    confusionMatrix = [[0,0,0],[0,0,0],[0,0,0]]
    for e in errors:
        confusionMatrix[int(e[0])][int(e[1])] += 1
    
    for i,real in enumerate(confusionMatrix):
        for j,calculated in enumerate(real):
            confusionMatrix[i][j] = round(calculated / counts[i] * 100, 0)
        confusionMatrix[i][i] = 100
        for j,calculated in enumerate(real):
            if j != i:
                confusionMatrix[i][i] -= calculated
    return confusionMatrix


def plotConfusionMatrix(errors, counts, propStr):
    species = ['setosa', 'versicolor', 'virginica']
    title = 'Macierz pomyłek dla zestawu cech ' + propStr
    matrix = list(map(
        lambda row: list(map(lambda el: str(int(el)) + '%', row)), getConfusionMatrix(errors, counts)))
 
    # [real value][predicted value]
    df = pd.DataFrame(matrix)
    df.rename(
        index = lambda i: species[i], 
        columns = lambda i: species[i], 
        inplace=True
    )
    print('\n----------------------------------------\n' + title + '\n')
    print(df)

    # fig, ax = plt.subplots()

    # # hide axes
    # # fig.patch.set_visible(False)
    # # ax.axis('off')
    # # ax.axis('tight')

    # ax.set_xlabel('wartość przewidywana')
    # ax.set_ylim(-1,0)
    # # ax.yaxis.ticks(False)

    # ax.table(cellText=df.values, rowLabels=df.index, colLabels=df.columns, loc='center')

    # fig.tight_layout()

    # plt.savefig('charts/matrix' + propStr + '.png')


def processAndReturnErrors(trainData, testData, kValues, properties):
    
    errors = []
    for k in kValues:
        currentErrors = []
        for testCase in testData:
            # distances = []
            neighbours = []
            for trainCase in trainData:
                distance = 0
                for p in properties:
                    distance += (testCase[p] - trainCase[p])**2

                distance = math.sqrt(distance)
                # distances.append(distance)
                if len(neighbours) < k: 
                    neighbours.append([distance, trainCase[4]])
                else:
                    neighbours = sorted(neighbours, key=operator.itemgetter(0)) 
                    if distance < neighbours[k-1][0]:
                        neighbours[k-1] = [distance, trainCase[4]]
            # smallestDistances = nsmallest(k, distances)
            # print(smallestDistances)
            neighboursSpecies = list(map(lambda n: n[1], neighbours))
            occurence_count = Counter(neighboursSpecies) 
            species = occurence_count.most_common(1)[0][0]
            if testCase[4] != species :
                currentErrors.append([testCase[4], species])
        
        errors.append(currentErrors)
    
    return errors

if __name__ == "__main__":
    main()