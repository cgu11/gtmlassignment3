from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import validation_curve
from sklearn.neural_network import MLPClassifier

from timeit import default_timer as timer
import matplotlib.pyplot as plt
import numpy as np

def run_nn(train_x, train_y, test_x, test_y, hidden_layer_sizes, learning_rate, max_iter=100, batch_size=16, data_name=""):
    model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation='relu',
                                   solver='adam', alpha=0.01, batch_size=batch_size, learning_rate='constant',
                                   learning_rate_init=learning_rate, max_iter=100, tol=1e-4,
                                   early_stopping=False, validation_fraction=0.1, random_state=555555,
                                   warm_start=False)

    
    param_range = np.arange(1, max_iter+1)

    start_time = timer()
    train_scores, test_scores = validation_curve(
        model,
        train_x,
        train_y,
        param_name="max_iter",
        param_range=param_range,
        scoring="balanced_accuracy",
        n_jobs=-1,
    )
    training_time = timer() - start_time
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.title("Accuracy over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim(0.0, 0.5)
    lw = 2
    plt.plot(
        param_range, train_scores_mean, label="Training score", color="darkorange", lw=lw
    )
    plt.fill_between(
        param_range,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.2,
        color="darkorange",
        lw=lw,
    )
    plt.plot(
        param_range, test_scores_mean, label="Cross-validation score", color="navy", lw=lw
    )
    plt.fill_between(
        param_range,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.2,
        color="navy",
        lw=lw,
    )
    plt.legend(loc="best")
    plt.grid()
    plt.savefig(f"{data_name}_nn.png")
    plt.close()

    holdout_acc = balanced_accuracy_score(test_y, )

    return training_time, train_scores_mean, test_scores_mean