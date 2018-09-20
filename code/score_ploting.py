
import json
from statistics import mean, pvariance

import matplotlib.pyplot as plt
import numpy


def plot_scores(scores):
    #set the names up
    feats = ['Basic', 'Middle', 'Complicated']
    datasets = ['FILE', 'GMB', 'MIX']

    #means
    test_means = numpy.ndarray((3, 3,), dtype=numpy.float32)
    train_means = numpy.ndarray((3, 3,), dtype=numpy.float32)
    other_means = numpy.ndarray((2, 3,), dtype=numpy.float32)
    #variences
    test_vars = numpy.ndarray((3, 3,), dtype=numpy.float32)
    train_vars = numpy.ndarray((3, 3,), dtype=numpy.float32)
    other_vars = numpy.ndarray((2, 3,), dtype=numpy.float32)

    f1 = plt.figure()
    ax1 = f1.add_subplot(111)
    ax1.set_title('Test Scores')
    f2 = plt.figure()
    ax2 = f2.add_subplot(111)
    ax2.set_title('Train Scores')
    f3 = plt.figure()
    ax3 = f3.add_subplot(111)
    ax3.set_title('Other Data Scores')

    for idx1, feat in enumerate(feats):
        for idx2, dataset in enumerate(datasets):
            #get scores from the dict and plot them
            test_scores = [i[0] for i in scores[feat][dataset]]
            train_scores = [i[1] for i in scores[feat][dataset]]

            #plot test scores
            ax1.scatter([feat + " -- " + dataset for i in range(len(test_scores))], 
                                        test_scores)
            ax2.scatter([feat + " -- " + dataset for i in range(len(train_scores))], 
                                        train_scores)
            #save mean, varience
            test_means[idx1][idx2] = mean(test_scores)
            test_vars[idx1][idx2] = pvariance(test_scores)

            train_means[idx1][idx2] = mean(train_scores)
            train_vars[idx1][idx2] = pvariance(train_scores)

            if not dataset == 'MIX':
                #scores from testing on opposit data set
                other_scores = [i[2] for i in scores[feat][dataset]]

                #plot the 'other' scores
                ax3.scatter([feat + " -- " + dataset for i in range(len(other_scores))], 
                                        other_scores)

                train_means[idx1][idx2] = mean(other_scores)
                train_vars[idx1][idx2] = pvariance(other_scores)


    ticks = numpy.arange(0,3,1)
    ticks2 = numpy.arange(0,2,1)
    ###################################### heatmaps for means ######################################

    #make the heatma figure 
    hm = plt.figure()

    ################## means ###################
    ax = hm.add_subplot(231)
    ax.set_title('Test means')
    plotted = ax.matshow(test_means)
    hm.colorbar(plotted, ax=ax)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(feats)
    ax.set_yticklabels(datasets)

    ax = hm.add_subplot(232)
    ax.set_title('Train means')
    plotted = ax.matshow(train_means)
    hm.colorbar(plotted, ax=ax)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(feats)
    ax.set_yticklabels(datasets)

    ax = hm.add_subplot(233)
    ax.set_title('Other Data means')
    plotted = ax.matshow(other_means)
    hm.colorbar(plotted, ax=ax)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks2)
    ax.set_xticklabels(feats)
    ax.set_yticklabels(datasets[:-1])


    ################## variences ###################
    ax = hm.add_subplot(234)
    ax.set_title('Test variances')
    plotted = ax.matshow(test_vars)
    hm.colorbar(plotted, ax=ax)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(feats)
    ax.set_yticklabels(datasets)

    ax = hm.add_subplot(235)
    ax.set_title('Train variances')
    plotted = ax.matshow(train_vars)
    hm.colorbar(plotted, ax=ax)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(feats)
    ax.set_yticklabels(datasets)

    ax = hm.add_subplot(236)
    ax.set_title('Other Data variances')
    plotted = ax.matshow(other_vars)
    hm.colorbar(plotted, ax=ax)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks2)
    ax.set_xticklabels(feats)
    ax.set_yticklabels(datasets[:-1])

    plt.show()


if __name__ == '__main__':
    scorefile = '/home/eoshea/sflintro/scores/scores.json'
    with open(scorefile, 'r') as infile:
        scores = json.load(infile)

    if scores:
        plot_scores(scores)
    else:
        raise ValueError("Scores not loaded")