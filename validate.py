from tensorflow import set_random_seed
import numpy as np
import random
from my_model_utils import train_and_test, plot_loss


scores = []
for i in [2, 16, 42]:

    random.seed(i)
    np.random.seed(i)
    set_random_seed(i)

    score, dev_score, history = train_and_test(num_epochs=100, learning_rate=0.00001, trainingdata="kaggle")
    scores.append(score)
    #plot_loss(history, "X_200lr0001.png")
    #score, history = train_and_test(num_epochs=200, learning_rate=0.00001, num_cells=256)
    #plot_loss(history, "X_256cellsLR00001.png")
print("kaggle AVERAGE=", np.mean(scores))


scores = []
for i in [2, 16, 42]:

    random.seed(i)
    np.random.seed(i)
    set_random_seed(i)

    score, history = train_and_test(num_epochs=100, learning_rate=0.00001, trainingdata="BS")
    scores.append(score)
    #plot_loss(history, "X_200lr0001.png")
    #score, history = train_and_test(num_epochs=200, learning_rate=0.00001, num_cells=256)
    #plot_loss(history, "X_256cellsLR00001.png")
print("BS AVERAGE=", np.mean(scores))


scores = []
for i in [2, 16, 42]:

    random.seed(i)
    np.random.seed(i)
    set_random_seed(i)

    score, history = train_and_test(num_epochs=100, learning_rate=0.00001, trainingdata="FNC")
    scores.append(score)
    #plot_loss(history, "X_200lr0001.png")
    #score, history = train_and_test(num_epochs=200, learning_rate=0.00001, num_cells=256)
    #plot_loss(history, "X_256cellsLR00001.png")
print("FNC AVERAGE=", np.mean(scores))
