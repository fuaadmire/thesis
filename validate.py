from tensorflow import set_random_seed
import numpy as np
import random
from my_model_utils import train_and_test, plot_loss


scores = []
for i in [42]:#[2, 16, 42]:

    random.seed(i)
    np.random.seed(i)
    set_random_seed(i)

    score, history = train_and_test(num_epochs=200, learning_rate=0.0001)
    scores.append(score)
    plot_loss(history, "X_200lr0001.png")
    score, history = train_and_test(num_epochs=200, learning_rate=0.00001, num_cells=256)
    plot_loss(history, "X_256cellsLR00001.png")
print("AVERAGE=", np.mean(scores))
