#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append('../')
from loglizer.models import InvariantsMiner
from loglizer import dataloader, preprocessing
max_dist = 0.3 # the threshold to stop the clustering process
anomaly_threshold = 0.3 # the threshold for anomaly detection


struct_log = '../data/SVTFS/svtfs_1k-8k_task.log_structured.csv' # The structured log file
struct_log1 = '../data/SVTFS/svtfs_8k-16k_task.log_structured.csv' # The structured log file

if __name__ == '__main__':
    ## 1. Load strutured log file and extract feature vectors
    # Save the raw event sequence file by setting save_df=True
    (x_train, _), (_, _) = dataloader.load_SVTFS_TASK(struct_log, window='session', save_df=True)
    feature_extractor = preprocessing.FeatureExtractor()
    x_train = feature_extractor.fit_transform(x_train, term_weighting='tf-idf', 
                                              normalization='zero-mean')
    
    print(x_train)

    ## 2. Train an unsupervised model
    print('Train phase:')
    # Initialize PCA, or other unsupervised models, LogClustering, InvariantsMiner
    model = InvariantsMiner()
    # Model hyper-parameters may be sensitive to log data, here we use the default for demo
    model.fit(x_train)
    # Make predictions and manually check for correctness. Details may need to go into the raw logs
    y_train = model.predict(x_train) 

    print(y_train)
    print("")
    print("")

    ## 3. Use the trained model for online anomaly detection
    print('Test phase:')
    print("")
    print("")
    print('Loading the same logfile to show no anomalies are detected')
    # Load the same log file used to train, it should show no anomalies...
    (x_test, _), (_, _) = dataloader.load_SVTFS_TASK(struct_log, window='session')
    x_test = feature_extractor.transform(x_test)
    y_test = model.predict(x_test)
    print(y_test)

    print("")
    print("")
    print('Loading a new logfile that nas no match. All new sequences should be anomalies')
    print("")
    # Load another new log file. 
    (x_test, _), (_, _) = dataloader.load_SVTFS_TASK(struct_log1, window='session')
    # Go through the same feature extraction process with training, using transform() instead
    x_test = feature_extractor.transform(x_test) 

    print(x_test)

    # Finally make predictions and alter on anomaly cases
    y_test = model.predict(x_test)

    print(y_test)
