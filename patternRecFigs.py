#
# patternRecFigs.py
#
# The goal of this file is to show some pattern recognition ideas
#
# Seth McNeill
# 2022 March 29

import argparse  # for parsing input arguments
import datetime  # for time related functions
import numpy as np  # for datastructures
import matplotlib.pyplot as plt  # for actual plotting
import random  # for random sampling
import pdb  # for debugging
import sys  # to quit early (exit)
import pandas as pd  # for importing csv files
from scipy.stats import norm

salmonLength  = [0,7,17,17,16,11,11,12,21,21,20,20,5,3,2,3,8,5,10,1,10,2,3,10,8,3,2]
seabassLength = [0,0,4,5,1,1,8,5,5,3,3,6,8,8,9,16,16,18,22,24,24,8,6,6,6,0,0]

salmonColor = []
seabassColor = []

nClass1 = 150
nClass2 = 225
priors = [nClass1/(nClass1+nClass2), nClass2/(nClass1+nClass2)]
class1Mean = (4,13)
class2Mean = (8,24)
class1Std = (1,6)
class2Std = (2,8)

class1raw = np.random.normal(class1Mean, class1Std, (nClass1,2))
class2raw = np.random.normal(class2Mean, class2Std, (nClass2,2))

def plot_classes():
    # plot the prior probabilities
    fig, ax = plt.subplots(figsize=(11,8.5))
    plt.rcParams['font.size'] = '18'
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(18)
    plt.bar([1,2], priors, color=['b', 'r'], tick_label=['Salmon', 'Seabass'])
    plt.ylabel('Probability', fontsize=20)
    plt.title('Prior Probabilities')
    plt.grid()
    plt.savefig('priors.pdf')
    plt.savefig('priors.png')
    plt.show()

    # plot of running color discriminant across all values present
    colorMisClassified = []
    testDiscs = np.arange(min(min(class1raw[:,0]),0),max(class2raw[:,0]), 0.1)
    for vertDisc in testDiscs:
        colorMisClassified.append(sum(class2raw[:,0] < vertDisc) + sum(class1raw[:,0] > vertDisc))
    bestDisc = testDiscs[colorMisClassified.index(min(colorMisClassified))] 
    
    fig, ax = plt.subplots(figsize=(11,8.5))
    plt.rcParams['font.size'] = '18'
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(18)
    ax.plot(testDiscs, colorMisClassified, '.-', markersize=12)
    ylims = ax.get_ylim()
    ax.vlines(bestDisc, ylims[0], ylims[1], 'g', linewidth=2, label='Best Discriminant')
    plt.xlabel('Color Intensity', fontsize=20)
    plt.ylabel('Number Misclassified', fontsize=20)
    plt.title(f'Best Discriminant at {bestDisc:.1f} with {min(colorMisClassified)} Errors', fontsize=36)
    plt.legend(loc='lower right', frameon=False, edgecolor=None)
    plt.grid()
    plt.savefig('variedColor.pdf')
    plt.savefig('variedColor.png')
    plt.show()    
    

    # Base raw data
    fig, ax = plt.subplots(figsize=(11,8.5))
    plt.rcParams['font.size'] = '18'
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(18)
    ax.plot(class1raw[:,0], class1raw[:,1], '.b', label='Salmon', markersize=12)
    ax.plot(class2raw[:,0], class2raw[:,1], '.r', label='Seabass', markersize=12)
    plt.xlabel('Color Intensity', fontsize=20)
    plt.ylabel('Length', fontsize=20)
    ax.legend()
    plt.grid()
    plt.savefig('raw.pdf')
    plt.savefig('raw.png')
    plt.show()

    # Single linear discriminant
    colorBoundary = bestDisc
    lengthBoundary = 17.5

    fig, ax = plt.subplots(figsize=(11,8.5))
    plt.rcParams['font.size'] = '18'
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(18)
    ax.plot(class1raw[:,0], class1raw[:,1], '.b', label='Salmon', markersize=12)
    ax.plot(class2raw[:,0], class2raw[:,1], '.r', label='Seabass', markersize=12)
    ylims = ax.get_ylim()
    class2misIdx = class2raw[:,0] < colorBoundary
    class1misIdx = class1raw[:,0] > colorBoundary
    class1misCnt = sum(class1misIdx)
    class2misCnt = sum(class2misIdx)
    confusionMatrix = ((sum(np.invert(class1misIdx)), class1misCnt),(class2misCnt, sum(np.invert(class2misIdx))))
    ax.vlines(colorBoundary, ylims[0], ylims[1], label='Color Boundary', linewidth=2, color='g')
    ax.plot(class1raw[class1misIdx,0], class1raw[class1misIdx,1], 'ro', markersize=12, markerfacecolor='none')
    ax.plot(class2raw[class2misIdx,0], class2raw[class2misIdx,1], 'ro', markersize=12, markerfacecolor='none')
    ax.text(12,5, f'Confusion Matrix\n{confusionMatrix[0][0]} {confusionMatrix[0][1]}\n{confusionMatrix[1][0]}  {confusionMatrix[1][1]}', horizontalalignment='center', verticalalignment='center', fontsize=18)
    plt.xlabel('Color Intensity', fontsize=20)
    plt.ylabel('Length', fontsize=20)
    plt.suptitle(f'Linear Discriminant at {colorBoundary:.1f}', fontsize=36)
    plt.title(f'Total Misclassified: {class1misCnt+class2misCnt} ({class1misCnt} salmon, {class2misCnt} seabass)', fontsize=20)
    ax.legend(loc='upper left',)
    plt.grid()
    plt.savefig('colorDisc.pdf')
    plt.savefig('colorDisc.png')
    plt.show()

    print(confusionMatrix)

    counts = np.histogram(np.concatenate((class1raw[:,0],class2raw[:,0])), 20)

    fig, ax = plt.subplots(figsize=(11,8.5))
    plt.rcParams['font.size'] = '18'
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(18)
    plt.hist([class1raw[:,0],class2raw[:,0]], bins=counts[1], label=['Salmon','Seabass'], color=['b','r'])
    #plt.hist(class1raw[:,0], bins=counts[1], label='Salmon', color='b')
    #plt.hist(class2raw[:,0], bins=counts[1], label='Seabass', color='r')
    plt.grid()
    plt.xlabel('Color Intensity', fontsize=20)
    plt.ylabel('Count', fontsize=20)
    plt.title('Histogram of Color Intensity for Both Classes')
    plt.legend()
    plt.savefig('hist.pdf')
    plt.savefig('hist.png')
    plt.show()
    
    fig, ax = plt.subplots(figsize=(11,8.5))
    plt.rcParams['font.size'] = '18'
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(18)
    pdf_data = plt.hist([class1raw[:,0],class2raw[:,0]], bins=counts[1], label=['Salmon','Seabass'], color=['b','r'], density=True)
    #ax.hist(class1raw[:,0], bins=counts[1], density=True, label='Salmon', color='b')
    #ax.hist(class2raw[:,0], bins=counts[1], density=True, label='Seabass', color='r')
    plt.grid()
    plt.xlabel('Color Intensity', fontsize=20)
    plt.ylabel('Normalized Count', fontsize=20)
    plt.title('Probability Density Function of Color Intensity for Both Classes')
    plt.legend()
    plt.savefig('pdf.pdf')
    plt.savefig('pdf.png')
    plt.show()

    evidence = [sum(pdf_data[0][0]*priors[0]), sum(pdf_data[0][1]*priors[1])]
    test_pts = 4
    # plot posteriors and show boundary
    posteriors1 = pdf_data[0][0]*priors[0]/evidence[0]
    posteriors2 = pdf_data[0][1]*priors[1]/evidence[1]

    bins = counts[1]
    binWidth = (max(bins) - min(bins)) / (len(bins) - 1)
    binCenters = np.arange(min(bins)+binWidth/2, max(bins), binWidth)

    fig, ax = plt.subplots(figsize=(11,8.5))
    plt.rcParams['font.size'] = '18'
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(18) 
    plt.bar(binCenters, posteriors1, width=0.5, label='Salmon', color='b', alpha=0.5)
    plt.bar(binCenters, posteriors2, width=0.5, label='SeaBass', color='r', alpha=0.5)
    x_axis = np.arange(0,plt.xlim()[1],0.01)
    pdf1 = priors[0]*norm.pdf(x_axis,np.mean(class1raw,0)[0],np.std(class1raw,0)[0])
    pdf2 = priors[1]*norm.pdf(x_axis,np.mean(class2raw,0)[0],np.std(class2raw,0)[0])
    # find crossover point as threshold
    pdfDiff = pdf1 - pdf2
    indLastPosDiff = [i for i,x in enumerate(np.diff(np.sign(pdfDiff))) if x < 0]
    thresh = (x_axis[indLastPosDiff[0]] + x_axis[indLastPosDiff[0]+1])/2
    ax.vlines(thresh, plt.ylim()[0], plt.ylim()[1], label='Color Boundary', linewidth=2, color='g')
    plt.plot(x_axis, pdf1, color='b')
    plt.plot(x_axis, pdf2, color='r')
    plt.grid()
    plt.xlabel('Color Intensity', fontsize=20)
    plt.ylabel('Posterior Probability', fontsize=20)
    plt.suptitle('Posterior Probabilities of Color Intensity for Both Classes')
    plt.title(f'Threshold = {thresh:.2f}',fontsize=18)
    plt.legend()
    plt.savefig('posterior.pdf')
    plt.savefig('posterior.png')
    plt.show()
    pdb.set_trace()
    



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
#    parser.add_argument('--imu', help='Name of file containing IMU data')
#    parser.add_argument('--env', help='Name of file containing environmental data')
    args = parser.parse_args()
    start_time = datetime.datetime.now()

    plot_classes()

    end_time = datetime.datetime.now()
    print(f"{__file__} took {end_time - start_time} s")