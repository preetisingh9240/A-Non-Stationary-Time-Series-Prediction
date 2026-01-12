import numpy as np
from prettytable import PrettyTable
import matplotlib.pyplot as plt
from matplotlib import pylab
from matplotlib.lines import Line2D


def stats(val):
    v = np.zeros(5)
    v[0] = max(val)
    v[1] = min(val)
    v[2] = np.mean(val)
    v[3] = np.median(val)
    v[4] = np.std(val)
    return v


def plot_conv():
    Fitness = np.load('Fitness.npy', allow_pickle=True)
    Algorithm = ['Terms', 'FFO-MA-ResESN', 'FANO-MA-ResESN', 'LOA-MA-ResESN', 'SOA-MA-ResESN', 'RSOA-RRP-MA-ResESN']
    Terms = ['Worst', 'Best', 'Mean', 'Median', 'Std']
    Values = np.zeros((Fitness.shape[-2], 5))
    for n in range(len(Fitness)):
        for j in range(len(Algorithm) - 1):
            Values[j, :] = stats(Fitness[n, j, :])

        Table = PrettyTable()
        Table.add_column(Algorithm[0], Terms)
        for j in range(len(Algorithm) - 1):
            Table.add_column(Algorithm[j + 1], Values[j, :])
        print('-------------------------------------------------- Statistical Report',
              ' Dataset ' + str(n + 1) + ' --------------------------------------------------')
        print(Table)

        length = np.arange(Fitness.shape[-1])
        Conv_Graph = Fitness[n]
        plt.plot(length, Conv_Graph[0, :], color='#e50000', linewidth=3, markersize=12, label=Algorithm[1])
        plt.plot(length, Conv_Graph[1, :], color='#0504aa', linewidth=3, markersize=12, label=Algorithm[2])
        plt.plot(length, Conv_Graph[2, :], color='#76cd26', linewidth=3, markersize=12, label=Algorithm[3])
        plt.plot(length, Conv_Graph[3, :], color='#b0054b', linewidth=3, markersize=12, label=Algorithm[4])
        plt.plot(length, Conv_Graph[4, :], color='k', linewidth=3, markersize=12, label=Algorithm[5])
        plt.xlabel('Iteration')
        plt.ylabel('Cost Function')
        plt.legend(loc=1)
        plt.savefig("./Results/Dataset_%s_Convergence.png" % (n + 1))
        fig = pylab.gcf()
        fig.canvas.manager.set_window_title('Convergence Curve')
        plt.show()


def Plot_batchsize_error():
    eval = np.load('Eval_All_BS_error.npy', allow_pickle=True)
    Terms = ['MPE', 'SMAPE', 'RMSE', 'MASE', 'MAE', 'MSE', 'NMSE', 'ONENORM', 'TWONORM', 'INFINITYNORM', 'MAPE',
             'Accuracy']
    Graph_Terms = [1, 2, 5, 11]
    Classifiers = ['ResESN', 'RSOA-RRP-\nMA-ResESN']
    Batch_size = ['Batch size: 4', 'Batch size: 8', 'Batch size: 16', 'Batch size: 32', ]
    Mtd_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#bf77f6']
    # Alg_colors = ['#fe4b03', '#021bf9', '#20f986', '#fe46a5', '#bf77f6']
    for n in range(eval.shape[0]):
        for j in range(len(Graph_Terms)):
            Graph = eval[n, :4, :, Graph_Terms[j]]
            Alg_Graph = np.array([Graph[:, 3], Graph[:, 4]])
            Datas = np.array(Alg_Graph)
            x = np.arange(Alg_Graph.shape[-1])
            width = 0.12
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = []
            for i in range(len(Classifiers)):
                bars.append(
                    ax.bar(x + (i - 2) * width, Datas[i], width, label=Classifiers[i], color=Mtd_colors[i])
                )
            ax.set_xticks([])
            table_data = [[f"{v:.4f}" for v in Datas[i]] for i in range(len(Classifiers))]
            table = plt.table(
                cellText=table_data,
                rowLabels=Classifiers,
                colLabels=Batch_size,
                rowColours=Mtd_colors,
                rowLoc='center',
                cellLoc='center',
                loc='bottom',
                bbox=[0.001, -0.42, 1, 0.42]
            )
            table.auto_set_font_size(False)
            table.set_fontsize(12)
            for col in range(len(Batch_size)):
                cell = table[0, col]
                cell.set_text_props(weight='bold')
            for row in range(1, len(Classifiers) + 1):
                cell = table[row, -1]
                cell.set_text_props(weight='bold', color='white')
            ax.set_ylabel(Terms[Graph_Terms[j]] + ' →', fontsize=12, fontweight='bold', color='#35530a')
            ax.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.7)
            ax.spines['top'].set_visible(False)
            ax.spines['left'].set_visible(True)
            ax.spines['right'].set_visible(False)
            plt.subplots_adjust(left=0.15, right=0.95, bottom=0.30, top=0.95)
            path = "./Results/Dataset %s Batch Size %s Mtd.png" % (n + 1, Terms[Graph_Terms[j]])
            fig = pylab.gcf()
            fig.canvas.manager.set_window_title('Classifier Batch Size  vs ' + Terms[Graph_Terms[j]])
            plt.savefig(path)
            plt.show()


def Plot_Kfold_error():
    eval = np.load('Eval_All_Fold_error.npy', allow_pickle=True)
    Terms = ['MPE', 'SMAPE', 'RMSE', 'MASE', 'MAE', 'MSE', 'NMSE', 'ONENORM', 'TWONORM', 'INFINITYNORM', 'MAPE',
             'Accuracy']
    # Graph_Terms = [0, 2, 3, 10, 11]
    Graph_Terms = [0, 1, 2, 3, 4, 5, 6, 10, 11]
    Algorithm = ['FFO-MA-ResESN', 'FANO-MA-ResESN', 'LOA-MA-ResESN', 'SOA-MA-ResESN', 'RSOA-RRP-MA-ResESN']
    Classifier = ['RNN', 'BiLSTM-RF-MPA', 'LSTM', 'ResESN', 'RSOA-RRP-MA-ResESN']
    Kfold = ['1', '2', '3', '4', '5']
    Kfold = np.asarray(Kfold)
    Algorithm = np.asarray(Algorithm)
    Classifier = np.asarray(Classifier)
    for n in range(eval.shape[0]):
        for j in range(len(Graph_Terms)):
            Graph = eval[n, :, :, Graph_Terms[j]]
            values = {
                Algorithm[0]: Graph[:, 0],
                Algorithm[1]: Graph[:, 1],
                Algorithm[2]: Graph[:, 2],
                Algorithm[3]: Graph[:, 3],
                Algorithm[4]: Graph[:, 4],
            }
            x = np.arange(len(Kfold))
            bar_width = 0.15
            fig = plt.figure()
            ax = fig.add_axes([0.12, 0.1, 0.8, 0.8])
            colors = ['dodgerblue', 'orange', 'green', 'red', 'purple']
            for i, (algorithm, color) in enumerate(zip(Algorithm, colors)):
                ax.bar(x + i * bar_width, values[algorithm], width=bar_width, label=algorithm,
                       color=color)
            ax.set_xticks(x + (len(Algorithm) - 1) * bar_width / 2)
            ax.set_xticklabels(Kfold, fontsize=10, fontweight='bold')
            ax.set_xlabel('K Fold', fontsize=12, fontweight='bold')
            ax.set_ylabel(Terms[Graph_Terms[j]], fontsize=12, fontweight='bold')
            ax.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.7)
            circle_markers = [Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i], markersize=10) for i in
                              range(len(Algorithm))]
            ax.legend(circle_markers, Algorithm, title="", fontsize=10, loc='upper center', bbox_to_anchor=(0.5, 1.1),
                      frameon=False, ncol=3)
            ax.spines['left'].set_linewidth(0.0)
            ax.spines['top'].set_color('lightgray')
            ax.spines['top'].set_linewidth(0.0)
            ax.spines['right'].set_color('lightgray')
            ax.spines['right'].set_linewidth(0.0)

            path = "./Results/Datasets %s K Fold %s Alg.png" % (n + 1, Terms[Graph_Terms[j]])
            fig = pylab.gcf()
            fig.canvas.manager.set_window_title('K Fold vs ' + Terms[Graph_Terms[j]])
            plt.savefig(path)
            plt.show()

            # Methods
            values = {
                Classifier[0]: Graph[:, 5],
                Classifier[1]: Graph[:, 6],
                Classifier[2]: Graph[:, 7],
                Classifier[3]: Graph[:, 8],
                Classifier[4]: Graph[:, 9],
            }
            x = np.arange(len(Kfold))
            bar_width = 0.15
            fig = plt.figure()
            ax = fig.add_axes([0.12, 0.1, 0.8, 0.8])
            colors = ['#ff000d', '#0cff0c', '#0652ff', '#e03fd8', 'black']
            for i, (algorithm, color) in enumerate(zip(Classifier, colors)):
                ax.bar(x + i * bar_width, values[algorithm], width=bar_width, label=algorithm,
                       color=color)
            ax.set_xticks(x + (len(Classifier) - 1) * bar_width / 2)
            ax.set_xticklabels(Kfold, fontsize=10, fontweight='bold')
            ax.set_xlabel('K Fold', fontsize=12, fontweight='bold')
            ax.set_ylabel(Terms[Graph_Terms[j]], fontsize=12, fontweight='bold')
            ax.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.7)
            circle_markers = [Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i], markersize=10) for i in
                              range(len(Classifier))]
            ax.legend(circle_markers, Classifier, title="", fontsize=10, loc='upper center', bbox_to_anchor=(0.5, 1.1),
                      frameon=False, ncol=3)
            ax.spines['left'].set_linewidth(0.0)
            ax.spines['top'].set_color('lightgray')
            ax.spines['top'].set_linewidth(0.0)
            ax.spines['right'].set_color('lightgray')
            ax.spines['right'].set_linewidth(0.0)
            path = "./Results/Datasets %s K Fold %s_Mtd.png" % (n + 1, Terms[Graph_Terms[j]])
            fig = pylab.gcf()
            fig.canvas.manager.set_window_title('K Fold vs ' + Terms[Graph_Terms[j]])
            plt.savefig(path)
            plt.show()


def Plot_Learning_per_error():
    eval = np.load('Eval_All_LP_error.npy', allow_pickle=True)
    Terms = ['MPE', 'SMAPE', 'RMSE', 'MASE', 'MAE', 'MSE', 'NMSE', 'ONENORM', 'TWONORM', 'INFINITYNORM', 'MAPE',
             'Accuracy']
    Table_Term = [0, 1, 2, 3, 4, 5, 6, 10, 11]
    Learn_per = ['35', '45', '55', '65', '75']
    Algorithm = ['Learning percentage', 'FFO-MA-ResESN', 'FANO-MA-ResESN', 'LOA-MA-ResESN', 'SOA-MA-ResESN',
                 'RSOA-RRP-MA-ResESN']
    Classifier = ['Learning percentage', 'RNN', 'BiLSTM-RF-MPA', 'LSTM', 'ResESN', 'RSOA-RRP-MA-ResESN']
    for n in range(len(eval)):
        for k in range(len(Table_Term)):
            value = eval[n, :, :, Table_Term[k]]
            Table = PrettyTable()
            Table.add_column(Algorithm[0], Learn_per)
            for j in range(len(Algorithm) - 1):
                Table.add_column(Algorithm[j + 1], value[:, j])
            print('--------------------------------------------------', str(Terms[Table_Term[k]]),
                  ' Learning percentage vs Algorithm Comparison of dataset ' + str(
                      n + 1) + ' --------------------------------------------------')
            print(Table)

            Table = PrettyTable()
            Table.add_column(Classifier[0], Learn_per)
            for j in range(len(Classifier) - 1):
                Table.add_column(Classifier[j + 1], value[:, len(Algorithm) + j - 1])
            print('-------------------------------------------------- ', str(Terms[Table_Term[k]]),
                  'Learning percentage vs Classifier Comparison of dataset ' + str(
                      n + 1) + ' --------------------------------------------------')
            print(Table)


if __name__ == '__main__':
    # plot_conv()
    Plot_Kfold_error()
    # Plot_batchsize_error()
    # Plot_Learning_per_error()
