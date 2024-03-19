#
#
# morseAnalysis.py
#
# Analyzes morse code timings.
#
# Seth McNeill
# 2022 November 06

import argparse  # for parsing input arguments
import datetime  # for time related functions
import numpy as np  # for datastructures
import matplotlib.pyplot as plt  # for actual plotting
import random  # for random sampling
import pdb  # for debugging
import sys  # to quit early (exit)
import pandas as pd  # for importing csv files
import glob  # for file seraching
from scipy.stats import norm
from sklearn.cluster import KMeans

plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\centering'

def plotMorse(dotFile,dashFile):
    priors = [1,1]
    lineHeight = 0.010
    rawDotData = pd.read_csv(dotFile)
    rawDashData = pd.read_csv(dashFile)
    dotTimes = rawDotData.loc[rawDotData['pol'] == 1]['dt']
    dotSpaces = rawDotData.loc[rawDotData['pol'] == 0]['dt']
    dotSpacesLong = dotSpaces.iloc[3::4]
    dotSpacesShort = pd.concat([dotSpaces.iloc[1::4], dotSpaces.iloc[2::4], dotSpaces.iloc[::4]])
    dashTimes = rawDashData.loc[rawDashData['pol'] == 1]['dt']
    dashSpaces = rawDashData.loc[rawDashData['pol'] == 0]['dt']
    dashSpacesLong = dashSpaces.iloc[2::3]
    dashSpacesShort = pd.concat([dashSpaces.iloc[::3], dashSpaces.iloc[1::3]])
    spacesLong = pd.concat([dashSpacesLong, dotSpacesLong])
    spacesShort = pd.concat([dashSpacesShort, dotSpacesShort])
    dotMean = np.mean(dotTimes)
    dashMean = np.mean(dashTimes)
    dotStd = np.std(dotTimes)
    dashStd = np.std(dashTimes)

    upDotTimes = rawDotData.loc[(rawDotData['pol'] == 0)]['dt'] 
    upDashTimes = rawDashData.loc[(rawDashData['pol'] == 0)]['dt'] 
    fig, ax = plt.subplots(figsize=(11,8.5))
    plt.rcParams['font.size'] = '18'
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(18)
    (n, bins, patches) = plt.hist([dotTimes,dashTimes], label=['Dot','Dash'],color=['b','r'], density=True)#, density=True)
    xlims = plt.xlim()
    ylims = plt.ylim()
    x_axis_dt = 0.01
    x_axis = np.arange(0,xlims[1],x_axis_dt)
    pdfDot = priors[0]*norm.pdf(x_axis,dotMean,dotStd)
    pdfDash = priors[1]*norm.pdf(x_axis,dashMean,dashStd)
    pdfDiff = pdfDot - pdfDash
    indLastPosDiff = [i for i,x in enumerate(np.diff(np.sign(pdfDiff))) if x < 0]
    thresh = (x_axis[indLastPosDiff[0]] + x_axis[indLastPosDiff[0]+1])/2
    ax.vlines(thresh, ylims[0], ylims[1], label='Dot/Dash Boundary', linewidth=2, color='g')
    # calculate how well this threshold discriminates between dots and dashes
    totalDots = len(dotTimes)
    totalDashes = len(dashTimes)
    nCorrectDot = sum(dotTimes < thresh)
    nCorrectDash = sum(dashTimes > thresh)
    nWrongDot = totalDots - nCorrectDot
    nWrongDash = totalDashes - nCorrectDash
    print('Confusion Matrix\nDot\tDash')
    print(f'{nCorrectDot}\t{nWrongDot}\n{nWrongDash}\t{nCorrectDash}')
    overallAccuracy = (nCorrectDash + nCorrectDot)/(totalDots + totalDashes)
    dotAccuracy = nCorrectDot/totalDots
    dashAccuracy = nCorrectDash/totalDashes
    dotPrecision = (nCorrectDot)/(nCorrectDot + nWrongDash)  # labeled as dot, but not dot
    dashPrecision = (nCorrectDash)/(nCorrectDash + nWrongDot)  # labeled as dash, but not dash
    overallPrecision = "Same as overall accuracy"
    dotRecall = nCorrectDot/(nCorrectDot + nWrongDot)  # labeled as dash, but were dot
    dashRecall = nCorrectDash/(nCorrectDash + nWrongDash)  # labeled as dot, but were dash
    overallRecall = (nCorrectDash + nCorrectDot)/(nCorrectDash + nCorrectDot + nWrongDot + nWrongDash)
    dotF1 = 2*(dotPrecision*dotRecall)/(dotPrecision+dotRecall)
    dashF1 = 2*(dashPrecision*dashRecall)/(dashPrecision+dashRecall)

    confMat = (r'$\underline{\mathrm{Confusion\ Matrix}}$\\' + r' $\begin{array}{ll} ' 
               r'\mathrm{Dot} & \mathrm{Dash}\\'
               f'{nCorrectDot}' + r' & ' f'{nWrongDot}' + 
               r' \\ ' + f'{nWrongDash}' + r'& ' + f'{nCorrectDash}' + r'\end{array}$')
    xw = xlims[1] - xlims[0]
    yh = ylims[1] - ylims[0]
    ax.text(-10,yh*0.3, confMat)
    performanceTxt = (f'Overall Accuracy = {overallAccuracy:.3f}\n'
                     f'dotAccuracy = {dotAccuracy:.3f}\n'
                     f'dashAccuracy = {dashAccuracy:.3f}\n'
                     f'dotPrecision = {dotPrecision:.3f}\n'
                     f'dashPrecision = {dashPrecision:.3f}\n'
                     f'Overall Precision = {overallPrecision}\n'
                     f'dotRecall = {dotRecall:.3f}\n'
                     f'dashRecall = {dashRecall:.3f}\n'
                     f'Overall Recall = {overallRecall:.3f}\n'
                     f'dot F1 = {dotF1:.3f}\n'
                     f'dash F1 = {dashF1:.3f}')
    ax.text(-10, 0.5*yh, performanceTxt)
    plt.plot(x_axis, pdfDot, color='b')
    plt.plot(x_axis, pdfDash, color='r')
    plt.grid()
    plt.xlabel('dt (ms)', fontsize=20)
    plt.ylabel('Normalized Count', fontsize=20)
    plt.suptitle('Histogram of Down Times (dots/dashes)')
    plt.title(f'thresh = {thresh} ms')
    plt.legend()
    plt.savefig('dotDashHist.pdf')
    # plt.savefig('dotDashHist.png')
    plt.show()

# ROC
    dotTPR = np.cumsum(pdfDot)/100
    dotFPR = np.cumsum(pdfDash)/100
    dashTPR = np.cumsum(np.flip(pdfDash,0))/100
    dashFPR = np.cumsum(np.flip(pdfDot,0))/100
    dotAUC = sum(dotTPR)/len(dotTPR)
    dashAUC = sum(dashTPR)/len(dashTPR)
    fig, ax = plt.subplots(figsize=(11,8.5))
    plt.rcParams['font.size'] = '18'
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(18)
    plt.plot(dotFPR,dotTPR, color='b', label='Dot ROC')
    plt.plot(dashFPR,dashTPR, color='g', label='Dash ROC')
    plt.plot([0,1.00],[0,1.00], color='r', linestyle='dashed', label='Worst Case')
    ax.text(0.8,0.3, f"dot AUC = {dotAUC:.2f}\ndash AUC = {dashAUC:.2f}")
    plt.grid()
    plt.xlabel('False Positive Rate', fontsize=20)
    plt.ylabel('True Positive Rate', fontsize=20)
    plt.suptitle('Receiver Operating Curves')
    plt.legend()
    plt.savefig('dotDashROC.pdf')
    # plt.savefig('dotDashHist.png')
    plt.show()
    
    fig, ax = plt.subplots(figsize=(11,8.5))
    plt.rcParams['font.size'] = '18'
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(18)
    (n, bins, patches) = plt.hist([spacesShort,spacesLong], label=['short','long'],color=['b','r'], density=True)#, density=True)
    x_axis = np.arange(0,plt.xlim()[1],0.01)
    pdfShort = priors[0]*norm.pdf(x_axis,np.mean(spacesShort),np.std(spacesShort))
    pdfLong = priors[1]*norm.pdf(x_axis,np.mean(spacesLong),np.std(spacesLong))
    pdfDiff = pdfShort - pdfLong
    indLastPosDiff = [i for i,x in enumerate(np.diff(np.sign(pdfDiff))) if x < 0]
    thresh = (x_axis[indLastPosDiff[0]] + x_axis[indLastPosDiff[0]+1])/2
    ax.vlines(thresh, plt.ylim()[0], plt.ylim()[1], label='Short/Long Boundary', linewidth=2, color='g')
    plt.plot(x_axis, pdfShort, color='b')
    plt.plot(x_axis, pdfLong, color='r')
    plt.grid()
    plt.xlabel('dt (ms)', fontsize=20)
    plt.ylabel('Normalized Count', fontsize=20)
    plt.suptitle('Histograms of Spaces (Up) Times (short/long)')
    plt.title(f'thresh = {thresh}')
    plt.legend()
    plt.savefig('spacesHist.pdf')
    # plt.savefig('dotDashHist.png')
    plt.show()


def plotMorseOld(filename):
    lineHeight = 0.010
    rawData = pd.read_csv(filename)
    downTimes = rawData.loc[rawData['pol'] == 1]['dt']
    downKmeans = KMeans(n_clusters=2).fit(downTimes.to_numpy().reshape(-1,1))
    upTimesAll = rawData.loc[(rawData['pol'] == 0)]['dt'] 
    upTimes = rawData.loc[(rawData['pol'] == 0) & 
                (rawData['dt'] < 2*max(downKmeans.cluster_centers_)[0])]['dt']

    upAllKmeans = KMeans(n_clusters=3).fit(upTimesAll.to_numpy().reshape(-1,1))
    upKmeans = KMeans(n_clusters=2).fit(upTimes.to_numpy().reshape(-1,1))
    print(f'down centers: {downKmeans.cluster_centers_}')
    print(f'up centers: {upKmeans.cluster_centers_}')
    print(f'up all centers: {upAllKmeans.cluster_centers_}')

    fig, ax = plt.subplots(figsize=(11,8.5))
    plt.rcParams['font.size'] = '18'
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(18)
    #plt.hist(frames, bins=counts[1], label=['Salmon','Seabass'], color=['b','r'])
    (n, bins, patches) = plt.hist(downTimes)#, density=True)
    plt.vlines(downKmeans.cluster_centers_, 0, max(n) + 1,colors='r')
    plt.grid()
    plt.xlabel('dt (ms)', fontsize=20)
    plt.ylabel('Count', fontsize=20)
    plt.title('Histogram of Down Times (dots/dashes)')
    #plt.legend()
    #plt.savefig('distHist.pdf')
    #plt.savefig('distHist.png')
    plt.show()

    fig, ax = plt.subplots(figsize=(11,8.5))
    plt.rcParams['font.size'] = '18'
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(18)
    #plt.hist(frames, bins=counts[1], label=['Salmon','Seabass'], color=['b','r'])
    (n, bins, patches) = plt.hist(upTimes)#, density=True)
    plt.vlines(upKmeans.cluster_centers_, 0, max(n)+1,colors='r')
    plt.grid()
    plt.xlabel('dt (ms)', fontsize=20)
    plt.ylabel('Count', fontsize=20)
    plt.title('Histogram of Up Times (pauses)')
    #plt.legend()
    #plt.savefig('distHist.pdf')
    #plt.savefig('distHist.png')
    plt.show()
    
    fig, ax = plt.subplots(figsize=(11,8.5))
    plt.rcParams['font.size'] = '18'
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(18)
    #plt.hist(frames, bins=counts[1], label=['Salmon','Seabass'], color=['b','r'])
    (n, bins, patches) = plt.hist(upTimesAll)#, density=True)
    plt.vlines(upAllKmeans.cluster_centers_, 0, max(n)+1,colors='r')
    plt.grid()
    plt.xlabel('dt (ms)', fontsize=20)
    plt.ylabel('Count', fontsize=20)
    plt.title('Histogram of All Up Times (pauses)')
    #plt.legend()
    #plt.savefig('distHist.pdf')
    #plt.savefig('distHist.png')
    plt.show()
  

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dots', help='Filename of Morse code dot (H) timings')
    parser.add_argument('--dashes', help='Filename of Morse code dashes (O) timings')
    args = parser.parse_args()
    start_time = datetime.datetime.now()

    plotMorse(args.dots, args.dashes)

    end_time = datetime.datetime.now()
    print(f"{__file__} took {end_time - start_time} s")