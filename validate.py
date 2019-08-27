from tensorflow import set_random_seed
import numpy as np
import random
from my_model_utils import train_and_test, plot_loss


# validate with 5 random seeds:

scores = []
for i in [2, 16, 42, 1, 4]:

    random.seed(i)
    np.random.seed(i)
    set_random_seed(i)

    score, dev_score, history = train_and_test(TIMEDISTRIBUTED=False, num_epochs=100, learning_rate=0.00001, trainingdata="liar")
    scores.append(score)

print("liar AVERAGE=", np.mean(scores))

scores = []
for i in [2, 16, 42, 1, 4]:

    random.seed(i)
    np.random.seed(i)
    set_random_seed(i)
    score, dev_score, history = train_and_test(TIMEDISTRIBUTED=False, num_epochs=18, num_cells=256, dropout=0.2, r_dropout=0.4, learning_rate=0.0001, trainingdata="BS")
    scores.append(score)

print("BS AVERAGE=", np.mean(scores))

#scores = []
#for i in [2, 16, 42]:

#    random.seed(i)
#    np.random.seed(i)
#    set_random_seed(i)

#    score, history = train_and_test(num_epochs=100, learning_rate=0.00001, trainingdata="FNC")
#    scores.append(score)
    #plot_loss(history, "X_200lr0001.png")
    #score, history = train_and_test(num_epochs=200, learning_rate=0.00001, num_cells=256)
    #plot_loss(history, "X_256cellsLR00001.png")
#print("FNC AVERAGE=", np.mean(scores))
