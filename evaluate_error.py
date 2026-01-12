import numpy as np


def evaluate_error(sp, act):
    A = np.squeeze(act)
    B = np.squeeze(sp)
    r = []
    x = []
    for i in range(len(A)):
        R = A[i]
        X = B[i]
        if R == 0 or X == 0:
            continue
        r.append(R)
        x.append(X)

    points = np.zeros(len(x))
    abs_r = np.zeros(len(x))
    abs_x = np.zeros(len(x))
    abs_r_x = np.zeros(len(x))
    abs_x_r = np.zeros(len(x))
    abs_r_x__r = np.zeros(len(x))
    for j in range(1, len(x)):
        points[j] = abs(x[j] - x[j - 1])
    for i in range(len(r)):
        abs_r[i] = abs(r[i])
    for i in range(len(r)):
        abs_x[i] = abs(x[i])
    for i in range(len(r)):
        abs_r_x[i] = abs(r[i] - x[i])
    for i in range(len(r)):
        abs_x_r[i] = abs(x[i] - r[i])
    for i in range(len(r)):
        abs_r_x__r[i] = abs((r[i] - x[i]) / r[i])
    mpe = (100 / len(x)) * sum(abs_r_x__r)  # Mean Error Percentage
    smape = (1 / len(x)) * sum(abs_r_x / ((abs_r + abs_x) / 2))
    mase = sum(abs_r_x) / ((1 / (len(x) - 1)) * sum(points))
    mae = sum(abs_r_x) / len(r)
    rmse = (sum(abs_x_r ** 2) / len(r)) ** 0.5
    nmse = (sum(abs_r_x ** 2) / sum(abs_r ** 2))  # Normalized Mean Squared Error
    mse = sum((r[i] - x[i]) ** 2 for i in range(len(r))) / len(r)  # Mean Squared Error
    onenorm = sum(abs_r_x)
    twonorm = (sum(abs_r_x ** 2) ** 0.5)
    infinitynorm = max(abs_r_x)
    mape = (100 / len(x)) * sum(abs_r_x / abs_r)  # Mean Absolute Percentage Error
    # accuracy = 100 - mape  # Accuracy
    accuracy = 1 - (np.abs(sum(np.asarray(r) - np.asarray(x))) / sum(np.asarray(r)))
    EVAL_ERR = [mpe, smape, rmse, mase, mae, mse, nmse, onenorm, twonorm, infinitynorm, mape, accuracy]
    return EVAL_ERR
