#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append('../')
from loglizer.models import PCA
from loglizer import dataloader, preprocessing

#struct_log = '../data/HDFS/HDFS_100k.log_structured.csv' # The structured log file
struct_log = '../data/SVTFS/svtfs_1k-8k_task.log_structured.csv' # The structured log file
struct_log1 = '../data/SVTFS/svtfs_8k-16k_task.log_structured.csv' # The structured log file


if __name__ == '__main__':
    ## 1. Load strutured log file and extract feature vectors
    # Save the raw event sequence file by setting save_df=True
    (x_train, _), (_, _) = dataloader.load_HDFS(struct_log, window='session', save_df=True)
    feature_extractor = preprocessing.FeatureExtractor()
    x_train = feature_extractor.fit_transform(x_train, term_weighting='tf-idf', 
                                              normalization='zero-mean')
    
    ## 2. Train an unsupervised model
    print('Train phase:')
    # Initialize PCA, or other unsupervised models, LogClustering, InvariantsMiner
    model = PCA() 
    # Model hyper-parameters may be sensitive to log data, here we use the default for demo
    model.fit(x_train)
    # Make predictions and manually check for correctness. Details may need to go into the raw logs
    y_train = model.predict(x_train) 

    ## 3. Use the trained model for online anomaly detection
    print('Test phase:')
    # Load another new log file. Here we use struct_log for demo only
    (x_test, _), (_, _) = dataloader.load_HDFS(struct_log1, window='session')
    # Go through the same feature extraction process with training, using transform() instead
    x_test = feature_extractor.transform(x_test) 
    # Finall make predictions and alter on anomaly cases
    y_test = model.predict(x_test)
    

