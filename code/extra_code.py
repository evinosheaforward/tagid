


#This is for plotting scores
    # limits = [0.75, 1.01]

    # for feat in feats: 
    #     for index, dataset in enumerate(datasets):
    #         ax = plt.subplot(2, 3, index+1)
    #         ax.set_title(dataset)
    #         ax.set_ylim(limits)
    #         plt.scatter([feat for i in range(len(scores[feat][dataset]))], 
    #                                      scores[feat][dataset])

    # for dataset in datasets: 
    #     for index, feat in enumerate(feats):        
    #         ax = plt.subplot(2, 3, index+1+len(feats))
    #         ax.set_title(feat)
    #         ax.set_ylim(limits)
    #         plt.scatter([dataset for i in range(len(scores[feat][dataset]))], 
    #                                     scores[feat][dataset])
    #
    # plt.show()