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

def plotMorse(filename):
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
    parser.add_argument('--filename', help='Filename of Morse code timings')
    args = parser.parse_args()
    start_time = datetime.datetime.now()

    plotMorse(args.filename)

    end_time = datetime.datetime.now()
    print(f"{__file__} took {end_time - start_time} s")