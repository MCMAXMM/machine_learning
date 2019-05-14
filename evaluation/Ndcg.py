def getNDCG(ranklist, gtItem):
    for i in range(len(ranklist)):
        item = ranklist[i]
        if item == gtItem:
            return math.log(2) / math.log(i+2)
    return 0
ranklist=[1,2,3,4,5,6,7,9,10,11,12,13,14]
b=[getNDCG(ranklist,i) for i in ranklist]
import matplotlib.pyplot as plt
plt.plot(ranklist,b)
plt.show()
