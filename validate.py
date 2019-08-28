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

    epochs = 5
    score, dev_score, history = train_and_test(TIMEDISTRIBUTED=True, trainingdata="liar", num_epochs=epochs, num_cells=64, dropout=0.8, r_dropout=0.6, learning_rate=0.001)
    scores.append(score)

liar_avg = np.mean(scores)
print("liar timedist. AVERAGE=", np.mean(scores))

scores = []
for i in [2, 16, 42, 1, 4]:

    random.seed(i)
    np.random.seed(i)
    set_random_seed(i)

    epochs = 10
    score, dev_score, history = train_and_test(TIMEDISTRIBUTED=True, trainingdata="BS", num_epochs=epochs, num_cells=256, dropout=0.2, r_dropout=0.4, learning_rate=0.0001)
    scores.append(score)

BS_avg = np.mean(scores)
print("BS timedist. AVERAGE=", np.mean(scores))


scores = []
for i in [2, 16, 42, 1, 4]:

    random.seed(i)
    np.random.seed(i)
    set_random_seed(i)

    epochs = 11
    score, dev_score, history = train_and_test(TIMEDISTRIBUTED=True, trainingdata="kaggle", num_epochs=epochs, num_cells=128, dropout=0.6, r_dropout=0.4, learning_rate=0.001)
    scores.append(score)

kaggle_avg = np.mean(scores)
print("Kaggle timedist. AVERAGE=", np.mean(scores))


scores = []
for i in [2, 16, 42, 1, 4]:

    random.seed(i)
    np.random.seed(i)
    set_random_seed(i)

    epochs = 5
    score, dev_score, history = train_and_test(TIMEDISTRIBUTED=True, trainingdata="FNC", num_epochs=epochs, num_cells=64, dropout=0.6, r_dropout=0.2, learning_rate=0.001)
    scores.append(score)

FNC_avg = np.mean(scores)
print("FNC timedist. AVERAGE=", np.mean(scores))


print("-------------------------")
print("liar avg:", liar_avg)
print("BS_avg:", BS_avg)
print("Kaggle avg:", kaggle_avg)
print("FNC avg:", FNC_avg)

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
