import numpy as np

liarliar = [0.558, 0.5, 0.481, 0.513, 0.532]
liarkaggle = [0.626, 0.615, 0.602, 0.551, 0.593]
liarBS = [0.656, 0.632, 0.584, 0.576, 0.588]

kagglekaggle = [0.983, 0.983, 0.982, 0.987, 0.970]
kaggleliar = [0.604, 0.599, 0.594, 0.590, 0.584]
kaggleBS = [0.770, 0.796, 0.799, 0.763, 0.793]

BSBS = [0.918, 0.912, 0.914, 0.899, 0.916]
BSliar = [0.609, 0.608, 0.604, 0.599, 0.604]
BSkaggle = [0.730, 0.730, 0.724, 0.733, 0.732]

def results(ls):
    print(np.round(np.mean(ls),4), np.round(np.std(ls),4))

results(liarliar)
results(liarkaggle)
results(liarBS)

results(kagglekaggle)
results(kaggleliar)
results(kaggleBS)

results(BSBS)
results(BSliar)
results(BSkaggle)
