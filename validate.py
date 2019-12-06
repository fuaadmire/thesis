from tensorflow import set_random_seed
import numpy as np
import random
from my_model_utils import train_and_test, plot_loss


# validate with 5 random seeds:

scores = []
f1_scores = []
for i in [2, 16, 42, 1, 4]:

    random.seed(i)
    np.random.seed(i)
    set_random_seed(i)

    epochs = 100
    score, dev_score, history, f1 = train_and_test(TIMEDISTRIBUTED=False, trainingdata="liar", num_epochs=epochs, num_cells=32, dropout=0.4, r_dropout=0.4, learning_rate=0.00001)
    scores.append(score)
    f1_scores.append(f1)

liar_avg = np.mean(scores)
liar_avg_std = np.std(scores)
liar_f1 = np.mean(f1_scores)
liar_f1_std = np.std(f1_scores)
print("liar AVERAGE=", np.mean(scores),"liar acc SD=",liar_avg_std, "liar f1 avg=", liar_f1, "liar f1 SD=", liar_f1_std)

scores = []
f1_scores = []
for i in [2, 16, 42, 1, 4]:

    random.seed(i)
    np.random.seed(i)
    set_random_seed(i)

    epochs = 18
    score, dev_score, history, f1 = train_and_test(TIMEDISTRIBUTED=False, trainingdata="BS", num_epochs=epochs, num_cells=256, dropout=0.2, r_dropout=0.4, learning_rate=0.0001)
    scores.append(score)
    f1_scores.append(f1)

BS_avg = np.mean(scores)
avg_std = np.std(scores)
f1 = np.mean(f1_scores)
f1_std = np.std(f1_scores)
print("BS AVERAGE=", np.mean(scores), "acc SD=", avg_std, "BS avg f1=", f1, "f1 SD", f1_std)


scores = []
f1_scores = []
for i in [2, 16, 42, 1, 4]:

    random.seed(i)
    np.random.seed(i)
    set_random_seed(i)

    epochs = 10
    score, dev_score, history, f1 = train_and_test(TIMEDISTRIBUTED=False, trainingdata="kaggle", num_epochs=epochs, num_cells=64, dropout=0.6, r_dropout=0.2, learning_rate=0.001)
    scores.append(score)
    f1_scores.append(f1)

kaggle_avg = np.mean(scores)
avg_std = np.std(scores)
f1 = np.mean(f1_scores)
f1_std = np.std(f1_scores)
print("Kaggle AVERAGE=", np.mean(scores), "acc SD=",avg_std, "kaggle f1=", f1, "f1 SD=", f1_std)


#scores = []
#for i in [2, 16, 42, 1, 4]:

 #   random.seed(i)
  #  np.random.seed(i)
   # set_random_seed(i)

   # epochs = 5
   # score, dev_score, history = train_and_test(TIMEDISTRIBUTED=True, trainingdata="FNC", num_epochs=epochs, num_cells=64, dropout=0.6, r_dropout=0.2, learning_rate=0.001)
   # scores.append(score)

#FNC_avg = np.mean(scores)
#print("FNC timedist. AVERAGE=", np.mean(scores))


print("-------------------------")
print("liar avg:", liar_avg)
print("BS_avg:", BS_avg)
print("Kaggle avg:", kaggle_avg)
#print("FNC avg:", FNC_avg)

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
