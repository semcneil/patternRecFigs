#
#
# ROC.py
#
# Receiver operating characteristics figures.
#
# based on code from:
# https://towardsdatascience.com/receiver-operating-characteristic-curves-demystified-in-python-bd531a4364d0
# but be sure to read comments since area under the curver (AUC)
# is incorrect: 
# https://medium.com/@jchblsc/nice-article-but-there-appears-to-be-inconsistency-between-the-graphed-curve-and-the-auc-c6ab05742d9
#
# Seth McNeill
# 2024 March 18

import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics  # from comment

def pdf(x, std, mean):
    cons = 1.0 / np.sqrt(2*np.pi*(std**2))
    pdf_normal_dist = cons*np.exp(-((x-mean)**2)/(2.0*(std**2)))
    return pdf_normal_dist

def plot_pdf(good_pdf, bad_pdf, ax):
    ax.fill(x, good_pdf, "b", alpha=0.5)
    ax.fill(x, bad_pdf,"r", alpha=0.5)
    ax.set_xlim([0,1])
    ax.set_ylim([0,5])
    ax.set_title("Probability Distribution", fontsize=14)
    ax.set_ylabel('Counts', fontsize=12)
    ax.set_xlabel('P(X="bad")', fontsize=12)
    ax.legend(["good","bad"])

def plot_roc(good_pdf, bad_pdf, ax):
    #Total
    total_bad = np.sum(bad_pdf)
    total_good = np.sum(good_pdf)
    #Cumulative sum
    cum_TP = 0
    cum_FP = 0
    #TPR and FPR list initialization
    TPR_list=[]
    FPR_list=[]
    #Iteratre through all values of x
    for i in range(len(x)):
        #We are only interested in non-zero values of bad
        if bad_pdf[i]>0:
            cum_TP+=bad_pdf[len(x)-1-i]
            cum_FP+=good_pdf[len(x)-1-i]
        FPR=cum_FP/total_good
        TPR=cum_TP/total_bad
        TPR_list.append(TPR)
        FPR_list.append(FPR)
    #Calculating AUC, taking the 100 timesteps into account
    # auc=np.sum(TPR_list)/100  # incorrect results
    auc = metrics.auc(FPR_list, TPR_list)
    #Plotting final ROC curve
    ax.plot(FPR_list, TPR_list)
    ax.plot(x,x, "--")
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    ax.set_title("ROC Curve", fontsize=14)
    ax.set_ylabel('TPR', fontsize=12)
    ax.set_xlabel('FPR', fontsize=12)
    ax.grid()
    ax.legend(["AUC=%.3f"%auc])

x = np.linspace(0, 1, num=100)
good_pdf = pdf(x,0.1,0.4)
bad_pdf = pdf(x,0.1,0.6)

# fig, ax = plt.subplots(1,1, figsize=(10,5))
# plot_pdf(good_pdf, bad_pdf, ax)
# plt.show()

# fig, ax = plt.subplots(1,1, figsize=(10,5))
# plot_roc(good_pdf, bad_pdf, ax)
# plt.show()

fig, ax = plt.subplots(1,2, figsize=(10,5))
plot_pdf(good_pdf, bad_pdf, ax[0])
plot_roc(good_pdf, bad_pdf, ax[1])
plt.tight_layout()
plt.savefig('ROC1.png')
plt.show()

x = np.linspace(0, 1, num=100)
fig, ax = plt.subplots(3,2, figsize=(10,12))
means_tuples = [(0.5,0.5),(0.45,0.55),(0.3,0.7)]
i=0
for good_mean, bad_mean in means_tuples:
    good_pdf = pdf(x, 0.1, good_mean)
    bad_pdf  = pdf(x, 0.1, bad_mean)
    fig, ax2 = plt.subplots(1,2, figsize=(10,5))
    plot_pdf(good_pdf, bad_pdf, ax2[0])
    plot_roc(good_pdf, bad_pdf, ax2[1])
    plt.savefig(f'ROCex{i}.png')
    plot_pdf(good_pdf, bad_pdf, ax[i,0])
    plot_roc(good_pdf, bad_pdf, ax[i,1])
    i+=1
plt.tight_layout()
plt.show()

