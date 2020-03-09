import pickle
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import os

model_filename = ""
root = "./results/semixup/"

with open(os.path.join(root, model_filename), "rb") as f:
    results = pickle.load(f)

n_label = 50

colors = ["green", "blue", "darkorange", "yellow", "red", "cyan"]

max_t=6

n_labels_list = [50, 100, 500, 1000]

n_labels = 500

for chart_name in ["roc", "pre_rec"]:
    for n_labels in n_labels_list:
        plt.figure(num=None, figsize=(4.0, 4.0), dpi=600, facecolor='w', edgecolor='k')
        for t in range(1,max_t+1):
            print(f"Process {t}x")
            for row in results:
                if row["n_labels"] == n_labels and row["n_unlabels"] == n_labels*t:
                    bi_pred_probs = row["bi_preds"]
                    bi_targets = row["bi_targets"]
                    if chart_name == "roc":
                        fpr, tpr, _ = roc_curve(y_true=bi_targets, y_score=bi_pred_probs)
                        auc_val = auc(x=fpr, y=tpr)
                        plt.plot(fpr, tpr, color=colors[t-1], label='%d-%dN, AUC %0.2f' % (row["n_labels"], t, auc_val))
                        plt.legend(loc="lower right")
                        plt.xlabel('False Positive Rate')
                        plt.ylabel('True Positive Rate')
                    elif chart_name == "pre_rec":
                        precision, recall, _ = precision_recall_curve(y_true=bi_targets, probas_pred=bi_pred_probs)
                        average_precision = average_precision_score(y_true=bi_targets, y_score=bi_pred_probs)
                        plt.plot(recall, precision, color=colors[t - 1],
                                 label='%d-%dN, APR %0.2f' % (row["n_labels"], t, average_precision))
                        plt.legend(loc="lower left")
                        plt.xlabel('Recall')
                        plt.ylabel('Precision')
                    else:
                        raise ValueError("Invalid chart name {}".format(chart_name))

                    plt.xlim([0.0, 1.05])
                    plt.ylim([0.5, 1.05])
                    plt.grid(True)
                    ax = plt.gca()
                    ax.set_aspect('equal', 'box')

                    t += 1
                    break

        if chart_name == "roc":
            output_plot_filename = os.path.join(root, f"ROC_AUC_{n_labels}.pdf")
        elif chart_name == "pre_rec":
            output_plot_filename = os.path.join(root, f"PRE_REC_{n_labels}.pdf")
        else:
            raise ValueError("Invalid chart name {}".format(chart_name))

        plt.tight_layout()
        plt.savefig(output_plot_filename, dpi=600, format="pdf")
        plt.clf()
        # plt.show()
