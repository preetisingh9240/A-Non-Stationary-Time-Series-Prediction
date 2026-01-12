import numpy as np
from Global_vars import Global_vars
from Model_MAR_ESN import Model_MAR_ESN
from evaluate_error import evaluate_error


def objfun_feat(Soln):
    Feat_1 = Global_vars.Feat_1
    Feat_2 = Global_vars.Feat_2
    Feat_3 = Global_vars.Feat_3
    Tar = Global_vars.Target
    Fitn = np.zeros(Soln.shape[0])
    dimension = len(Soln.shape)
    if dimension == 2:
        for i in range(Soln.shape[0]):
            sol = Soln[i]
            Feature_1 = Feat_1 * sol[0]
            Feature_2 = Feat_2 * sol[1]
            Feature_3 = Feat_3 * sol[2]
            Feat = np.concatenate((Feature_3, np.concatenate((Feature_1, Feature_2), axis=1)), axis=1)
            learnper = round(Feat.shape[0] * 0.75)
            Test_Target = Tar[learnper:, :]
            Eval, pred = Model_MAR_ESN(Feature_1, Feature_2, Feature_3, Tar, sol=sol)
            Eval = evaluate_error(Test_Target, pred[:Test_Target.shape[0]])
            Fitn[i] = (1 / Eval[11]) + Eval[2]  # (1 / Accuracy) + RMSE)
        return Fitn
    else:
        sol = Soln
        Feature_1 = Feat_1 * sol[0]
        Feature_2 = Feat_2 * sol[1]
        Feature_3 = Feat_3 * sol[2]
        Feat = np.concatenate((Feature_3, np.concatenate((Feature_1, Feature_2), axis=1)), axis=1)
        learnper = round(Feat.shape[0] * 0.75)
        Test_Target = Tar[learnper:, :]
        Eval, pred = Model_MAR_ESN(Feature_1, Feature_2, Feature_3, Tar, sol=sol)
        Eval = evaluate_error(Test_Target, pred[:Test_Target.shape[0]])
        Fitn = (1 / Eval[11]) + Eval[2]  # (1 / Accuracy) + RMSE)
        return Fitn
