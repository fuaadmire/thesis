from tensorflow import set_random_seed
import numpy as np
import random
from my_model_utils import train_and_test, plot_loss


random.seed(42)
np.random.seed(42)
set_random_seed(42)

#score, dev_score, history = train_and_test(TIMEDISTRIBUTED=True, trainingdata="liar", num_epochs=200, num_cells=64, dropout=0.8, r_dropout=0.6, learning_rate=0.001)
#values = history.history['val_acc']
#print("liar best epoch:",values.index(max(values)))
#plot_loss(history, "_liar_timedistributed_numcells64_dropout8_rdropout6_lr001")


#score, dev_score, history = train_and_test(TIMEDISTRIBUTED=True, trainingdata="BS", num_epochs=100, num_cells=256, dropout=0.2, r_dropout=0.4, learning_rate=0.0001)
#values = history.history['val_acc']
#print("BS best epoch:",values.index(max(values)))
#plot_loss(history, "_BS_timedistributed_numcells256_dropout2_rdropout4_lr0001")


#score, dev_score, history = train_and_test(TIMEDISTRIBUTED=True, trainingdata="kaggle", num_epochs=100, num_cells=128, dropout=0.6, r_dropout=0.4, learning_rate=0.001)
#values = history.history['val_acc']
#print("Kaggle best epoch:",values.index(max(values)))
#plot_loss(history, "_Kaggle_timedistributed_numcells128_dropout6_rdropout4_lr001")


#score, dev_score, history = train_and_test(TIMEDISTRIBUTED=True, trainingdata="FNC", num_epochs=100, num_cells=64, dropout=0.6, r_dropout=0.2, learning_rate=0.001)
#values = history.history['val_acc']
#print("FNC best epoch:",values.index(max(values)))
#plot_loss(history, "_FNC_timedistributed_numcells64_dropout6_rdropout2_lr001")

score, dev_score, history = train_and_test(TIMEDISTRIBUTED=False, trainingdata="FNC", num_epochs=100, num_cells=32, dropout=0.6, r_dropout=0.4, learning_rate=0.001)
values = history.history['val_acc']
print("FNC best epoch:",values.index(max(values)))
plot_loss(history, "_FNC_biLSTM_numcells32_dropout6_rdropout4_lr001")


scores = []
for i in [2, 16, 42, 1, 4]:

    random.seed(i)
    np.random.seed(i)
    set_random_seed(i)

    score, dev_score, history = train_and_test(TIMEDISTRIBUTED=False, trainingdata="FNC", num_epochs=values.index(max(values)+1, num_cells=32, dropout=0.6, r_dropout=0.4, learning_rate=0.001)
    scores.append(score)

FNC_avg = np.mean(scores)
print("FNC biLSTM AVERAGE=", np.mean(scores))
