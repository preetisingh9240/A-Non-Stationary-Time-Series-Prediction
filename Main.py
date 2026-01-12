import numpy as np
from numpy import matlib
from sklearn import decomposition
from scipy.stats import kurtosis, skew
import os
import pandas as pd
from FANO import FANO
from FFO import FFO
from Global_vars import Global_vars
from LOA import LOA
from Model_LSTM import Model_LSTM
from Model_MAR_ESN import Model_MAR_ESN
from Model_RNN import Model_RNN
from Model_BiLSTM_RF_MPA import Model_BiLSTM_RF_MPA
from Model_SA_STA import Model_STA_SA
from Obj_fun import objfun_feat
from PROPOSED import PROPOSED
from Plot_Results import *
from SOA import SOA

no_of_dataset = 3


def stats(val):
    v = np.zeros(9)
    v[0] = max(val)
    v[1] = min(val)
    v[2] = np.mean(val)
    v[3] = np.median(val)
    v[4] = np.std(val)
    v[5] = np.var(val)  # varience
    v[6] = kurtosis(val)  # kurtosis
    v[7] = skew(val)  # skewness
    v[8] = np.corrcoef(val)  # Correlation coefficient
    return v


def convert_to_numeric(df, columns):
    for col in columns:
        unique_values = df[col].unique()
        value_map = {val: idx for idx, val in enumerate(unique_values)}
        df[col] = df[col].map(value_map)
    return df


# Read the dataset 1
an = 0
if an == 1:
    Datasets = './Datasets/Dataset_1/'
    df = pd.read_csv(Datasets + 'train.csv')
    df.drop(['target_nitrogen_oxides'], inplace=True, axis=1)
    columns_to_convert = ['date_time']
    Datas = convert_to_numeric(df, columns_to_convert)
    # Nitrogen Oxides (NOx) is the most influential among the three targets
    Targets = pd.read_csv(Datasets + 'train.csv', usecols=['target_nitrogen_oxides'])
    np.save('Datas_1.npy', np.asarray(Datas))
    np.save('Targets_1.npy', np.asarray(Targets))

# Read the dataset 2
an = 0
if an == 1:
    Datasets = './Datasets/Dataset_2/'
    Traindf = pd.read_csv(Datasets + 'DailyDelhiClimateTrain.csv')
    Testdf = pd.read_csv(Datasets + 'DailyDelhiClimateTest.csv')
    df = pd.concat([Traindf, Testdf], ignore_index=True)
    Targets = df['meantemp']
    df.drop(['meantemp'], inplace=True, axis=1)
    columns_to_convert = ['date']
    Datas = convert_to_numeric(df, columns_to_convert)
    Targets = np.asarray(Targets).reshape(-1, 1)
    Datas = np.asarray(Datas)
    np.save('Datas_2.npy', Datas)
    np.save('Targets_2.npy', Targets)

# Read the dataset 3
an = 0
if an == 1:
    Datasets = './Datasets/Dataset_3/'
    All_Stocks_1 = pd.read_csv(Datasets + 'all_stocks_2006-01-01_to_2018-01-01.csv')
    All_Stocks_2 = pd.read_csv(Datasets + 'all_stocks_2017-01-01_to_2018-01-01.csv')
    df = pd.concat([All_Stocks_1, All_Stocks_2], ignore_index=True)
    Targets = df['Close']
    df.drop(['Close'], inplace=True, axis=1)
    df.fillna(0, inplace=True)
    columns_to_convert = ['Date', 'Name']
    Datas = convert_to_numeric(df, columns_to_convert)
    Targets = np.asarray(Targets).reshape(-1, 1)
    Datas = np.asarray(Datas)
    np.save('Datas_3.npy', Datas)
    np.save('Targets_3.npy', Targets)

# Feature (Deep Feature extraction using STA-SA)
an = 0
if an == 1:
    for n in range(no_of_dataset):
        Data = np.load('Datas_' + str(n + 1) + '.npy', allow_pickle=True)
        Targets = np.load('Targets_' + str(n + 1) + '.npy', allow_pickle=True)
        Feature = Model_STA_SA(Data, Targets)
        np.save('Feature_1_' + str(n + 1) + '.npy', np.asarray(Feature))

# Feature (Statistical Feature)
an = 0
if an == 1:
    for n in range(no_of_dataset):
        Data = np.load('Datas_' + str(n + 1) + '.npy', allow_pickle=True)
        Feat = []
        for j in range(len(Data)):
            print(j, len(Data))
            statistical = stats(Data[j])  # Statistical Feature
            Feat.append(np.asarray(statistical))
        np.save('Feature_2_' + str(n + 1) + '.npy', np.asarray(Feat))  # Save the Statistical Feature

# Feature (Distributed Principal Component Analysis (DPCA))
an = 0
if an == 1:
    for n in range(no_of_dataset):
        Data = np.load('Datas_' + str(n + 1) + '.npy', allow_pickle=True)
        pca = decomposition.PCA(n_components=4)
        pca_feature = pca.fit_transform(np.abs(Data))
        np.save('Feature_3_' + str(n + 1) + '.npy', pca_feature)  # Save the DPCA Features

# Optimization for classification
an = 0
if an == 1:
    FITNESS = []
    BESTSOL = []
    for n in range(no_of_dataset):
        Feat_1 = np.load('Feature_1_' + str(n + 1) + '.npy', allow_pickle=True)  # Load the Feat 1
        Feat_2 = np.load('Feature_2_' + str(n + 1) + '.npy', allow_pickle=True)  # Load the Feat 2
        Feat_3 = np.load('Feature_3_' + str(n + 1) + '.npy', allow_pickle=True)  # Load the Feat 3
        Target = np.load('Targets_' + str(n + 1) + '.npy', allow_pickle=True)  # Load the Target
        Global_vars.Feat_1 = Feat_1
        Global_vars.Feat_2 = Feat_2
        Global_vars.Feat_3 = Feat_3
        Global_vars.Target = Target
        Npop = 10
        Chlen = 3  # Hidden Neuron Count, Learning rate, Steps per epoch in ESN
        xmin = matlib.repmat(np.asarray([5, 0.01, 100]), Npop, 1)
        xmax = matlib.repmat(np.asarray([255, 0.99, 500]), Npop, 1)
        fname = objfun_feat
        initsol = np.zeros((Npop, Chlen))
        for p1 in range(initsol.shape[0]):
            for p2 in range(initsol.shape[1]):
                initsol[p1, p2] = np.random.uniform(xmin[p1, p2], xmax[p1, p2])
        Max_iter = 50

        print("FFO...")
        [bestfit1, fitness1, bestsol1, time1] = FFO(initsol, fname, xmin, xmax, Max_iter)  # CSA

        print("FANO...")
        [bestfit2, fitness2, bestsol2, time2] = FANO(initsol, fname, xmin, xmax, Max_iter)  # FANO

        print("LOA...")
        [bestfit3, fitness3, bestsol3, time3] = LOA(initsol, fname, xmin, xmax, Max_iter)  # LOA

        print("SOA...")
        [bestfit4, fitness4, bestsol4, time4] = SOA(initsol, fname, xmin, xmax, Max_iter)  # SOA

        print("PROPOSED...")
        [bestfit5, fitness5, bestsol5, time5] = PROPOSED(initsol, fname, xmin, xmax, Max_iter)  # Improved SOA

        BestSol = [bestsol1.squeeze(), bestsol2.squeeze(), bestsol3.squeeze(), bestsol4.squeeze(), bestsol5.squeeze()]
        FIT = [fitness1.squeeze(), fitness2.squeeze(), fitness3.squeeze(), fitness4.squeeze(), fitness5.squeeze()]

        FITNESS.append(FIT)
        BESTSOL.append(BestSol)
    np.save('BestSol.npy', np.asarray(BESTSOL))  # Best sol
    np.save('Fitness.npy', np.asarray(FITNESS))  # Fitness

# Classification
an = 0
if an == 1:
    Eval_all = []
    for n in range(no_of_dataset):
        Feat_1 = np.load('Feature_1_' + str(n + 1) + '.npy', allow_pickle=True)  # Load the Feat 1
        Feat_2 = np.load('Feature_2_' + str(n + 1) + '.npy', allow_pickle=True)  # Load the Feat 2
        Feat_3 = np.load('Feature_3_' + str(n + 1) + '.npy', allow_pickle=True)  # Load the Feat 3
        Target = np.load('Targets_' + str(n + 1) + '.npy', allow_pickle=True)  # Load the Target
        Feat = np.concatenate((Feat_1, Feat_2, Feat_3), axis=1)
        BestSol = np.load('BestSol.npy', allow_pickle=True)[n]  # Load the BestSolution
        k_fold = 5
        Per = 1 / k_fold
        EVAL = []
        Perc = round(Feat.shape[0] * Per)
        for i in range(k_fold):
            Test_Data = Feat[i * Perc: ((i + 1) * Perc), :]
            Test_Target = Target[i * Perc: ((i + 1) * Perc), :]
            test_index = np.arange(i * Perc, ((i + 1) * Perc))
            total_index = np.arange(Feat.shape[0])
            train_index = np.setdiff1d(total_index, test_index)
            Train_Data = Feat[train_index, :]
            Train_Target = Target[train_index, :]
            Eval = np.zeros((10, 12))
            for j in range(BestSol.shape[0]):
                print(j)
                sol = BestSol[j, :]
                Eval[j, :], pred0 = Model_MAR_ESN(Feat_1, Feat_2, Feat_3, Target, sol=sol, Perc=Perc)
            Eval[5, :], pred1 = Model_RNN(Train_Data, Train_Target, Test_Data, Test_Target)
            Eval[6, :], pred2 = Model_BiLSTM_RF_MPA(Train_Data, Train_Target, Test_Data, Test_Target)
            Eval[7, :], pred3 = Model_LSTM(Train_Data, Train_Target, Test_Data, Test_Target)
            Eval[8, :], pred4 = Model_MAR_ESN(Feat_1, Feat_2, Feat_3, Target, Perc=Perc)
            Eval[9, :] = Eval[4, :]
            EVAL.append(Eval)
        Eval_all.append(EVAL)
    np.save('Eval_All_Fold_error.npy', np.asarray(Eval_all))

plot_conv()
Plot_Kfold_error()
Plot_batchsize_error()
Plot_Learning_per_error()
