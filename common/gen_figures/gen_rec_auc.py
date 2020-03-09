import pickle
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import os

root = "/home/hoang/workspace/Semix/ssgan/results/"

with open(os.path.join(root, "exp_ssl_semixup_bestmodel_most_result.pkl"), "rb") as f:
    results = pickle.load(f)

colors = ["tab:blue", "tab:green", "tab:purple", "tab:red", "dimgray"]

n_labels_list = [50, 100, 500, 1000, 31922]
z_orders = [5, 4, 3, 2, 1]

show_ap_auc = False

for chart_name in ["roc", "pre_rec"]:
    plt.figure(num=None, figsize=(2.8, 2.8), dpi=600, facecolor='w', edgecolor='k')
    for t, n_labels in enumerate(n_labels_list):

        for row in results:
            if row["n_labels"] == n_labels:
                if n_labels == 31922:
                    n_labels = "SL (full OAI)"
                    linestyle = "dashed"
                    zorder = z_orders[t]
                else:
                    linestyle = "solid"
                    zorder = z_orders[t]
                bi_pred_probs = row["bi_preds"]
                bi_targets = row["bi_targets"]
                if chart_name == "roc":
                    fpr, tpr, _ = roc_curve(y_true=bi_targets, y_score=bi_pred_probs)
                    auc_val = auc(x=fpr, y=tpr)
                    if show_ap_auc:
                        legend = '{}, AUC {:0.3f}'.format(n_labels, auc_val)
                    else:
                        legend = f'{n_labels} / KL grade' if isinstance(n_labels, int) else f'{n_labels}'
                    plt.plot(fpr, tpr, color=colors[t], label=legend,
                             linestyle=linestyle, zorder=zorder)
                    plt.legend(loc="lower right")
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                elif chart_name == "pre_rec":
                    precision, recall, _ = precision_recall_curve(y_true=bi_targets, probas_pred=bi_pred_probs)
                    average_precision = average_precision_score(y_true=bi_targets, y_score=bi_pred_probs)
                    if show_ap_auc:
                        legend = '{}, AP {:0.3f}'.format(n_labels, average_precision)
                    else:
                        legend = f'{n_labels} / KL grade' if isinstance(n_labels, int) else f'{n_labels}'
                    plt.plot(recall, precision, color=colors[t], linestyle=linestyle, zorder=zorder,
                             label=legend)
                    plt.legend(loc="lower left")
                    plt.xlabel('Recall')
                    plt.ylabel('Precision')
                else:
                    raise ValueError("Invalid chart name {}".format(chart_name))

    plt.axis('equal')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.grid(True)

    if chart_name == "roc":
        output_plot_filename = os.path.join(root, f"ROC_AUC_Semixup.pdf")
    elif chart_name == "pre_rec":
        output_plot_filename = os.path.join(root, f"PRE_REC_Semixup.pdf")
    else:
        raise ValueError("Invalid chart name {}".format(chart_name))

    plt.tight_layout()
    plt.gca()
    # ax.set_aspect('equal', 'box')
    plt.savefig(output_plot_filename, dpi=600, format="pdf")
    plt.clf()
    # plt.show()
