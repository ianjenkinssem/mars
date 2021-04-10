#!/usr/bin/env python
#
# File: kmeans_new.py
# Author: Alexander Schliep (alexander@schlieplab.org)
#
#
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import time
import multiprocessing

def generateData(n, c):
    logging.info(f"Generating {n} samples in {c} classes")
    X, y = make_blobs(n_samples=n, centers = c, cluster_std=1.7, shuffle=False,
                      random_state = 2122)
    return X


def nearestCentroid(datum, centroids):
    # norm(a-b) is Euclidean distance, matrix - vector computes difference
    # for all rows of matrix
    dist = np.linalg.norm(centroids - datum, axis=1)
    return np.argmin(dist), np.min(dist)

# Added code for parallelization...
def invokeParallelCode(data, centroids):
    #N = len(data)
    #variation = np.zeros(k)
    #cluster_sizes = np.zeros(k, dtype=int)
    #for i in range(N):
    cluster, dist = nearestCentroid(data, centroids)
        #c[i] = cluster
        #cluster_sizes[cluster] += 1
        #variation[cluster] += dist ** 2
        #print("cluster cluster_size variation ==>", c[i], cluster_sizes[cluster], variation[cluster])

    return cluster, dist

def kmeans(workers, k, data, nr_iter = 2):
#def kmeans(k, data, nr_iter = 100):

    #Added code for parallelization
    #p = multiprocessing.Pool(workers)
    #N, new_X = p.map(invokeParallelCode, [(data, workers)] )

    N = len(data)

    # Choose k random data points as centroids
    centroids = data[np.random.choice(np.array(range(N)),size=k,replace=False)]
    logging.debug("Initial centroids\n", centroids)

    # The cluster index: c[i] = j indicates that i-th datum is in j-th cluster
    c = np.zeros(N, dtype=int)

    logging.info("Iteration\tVariation\tDelta Variation")
    total_variation = 0.0
    for j in range(nr_iter):  # For each iteration
        logging.debug("=== Iteration %d ===" % (j+1))
        print("Iteration# :", j+1)

        variation = np.zeros(k)
        cluster_sizes = np.zeros(k, dtype=int)

        # Added code for parallelization
        p = multiprocessing.Pool(workers)  # Split into number of processors
        new_data = np.array_split(data, workers)  # Split data to number of processors/workers
        print("============>",k, new_data)
        # End

        #Update N to the size of split data
        N=len(new_data[1])
        print("old N is", len(data))
        print("new N is", N)

        # Assign data points to nearest centroid
        for i in range(N):
            #cluster, dist = nearestCentroid(data[i],centroids)
            print("new_data[i]", i, list(new_data[i]))
            cluster, dist = p.starmap(invokeParallelCode,[list(new_data[i]),centroids])
            c[i] = cluster
            cluster_sizes[cluster] += 1
            variation[cluster] += dist**2
            print("cluster cluster_size variation ==>", c[i], cluster_sizes[cluster], variation[cluster])

        delta_variation = -total_variation
        total_variation = sum(variation) 
        delta_variation += total_variation

        print("iteration# total_variation delta_variation", j, total_variation, delta_variation)
        logging.info("%3d\t\t%f\t%f" % (j, total_variation, delta_variation))

        # Recompute centroids
        centroids = np.zeros((k,2)) # This fixes the dimension to 2
        for i in range(N):
            centroids[c[i]] += data[i]        
        centroids = centroids / cluster_sizes.reshape(-1,1)
        
        logging.debug(cluster_sizes)
        logging.debug(c)
        logging.debug(centroids)
        print("cluster cluster_size centroids =====>", c, cluster_sizes, centroids)

    return total_variation, c


def computeClustering(args):
    if args.verbose:
        logging.basicConfig(format='# %(message)s',level=logging.INFO)
    if args.debug: 
        logging.basicConfig(format='# %(message)s',level=logging.DEBUG)

    X = generateData(args.samples, args.classes)

    start_time = time.time()
    #
    # Modify kmeans code to use args.worker parallel threads
    total_variation, assignment = kmeans(args.workers, args.k_clusters, X, nr_iter = args.iterations)
    #
    #
    end_time = time.time()
    logging.info("Clustering complete in %3.2f [s]" % (end_time - start_time))
    print(f"Total variation {total_variation}")

    if args.plot: # Assuming 2D data
        fig, axes = plt.subplots(nrows=1, ncols=1)
        axes.scatter(X[:, 0], X[:, 1], c=assignment, alpha=0.2)
        plt.title("k-means result")
        #plt.show()        
        fig.savefig(args.plot)
        plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Compute a k-means clustering.',
        epilog = 'Example: kmeans_new.py -v -k 4 --samples 50 --classes 4 --plot result.png'
    )
    parser.add_argument('--workers', '-w',
                        default='2',
                        type = int,
                        help='Number of parallel processes to use (NOT IMPLEMENTED)')
    parser.add_argument('--k_clusters', '-k',
                        default='3',
                        type = int,
                        help='Number of clusters')
    parser.add_argument('--iterations', '-i',
                        ##default='100',
                        default='2',
                        type = int,
                        help='Number of iterations in k-means')
    parser.add_argument('--samples', '-s',
                        #default='10000',
                        default='10',
                        type = int,
                        help='Number of samples to generate as input')
    parser.add_argument('--classes', '-c',
                        default='3',
                        type = int,
                        help='Number of classes to generate samples from')   
    parser.add_argument('--plot', '-p',
                        type = str,
                        help='Filename to plot the final result')   
    parser.add_argument('--verbose', '-v',
                        action='store_true',
                        help='Print verbose diagnostic output')
    parser.add_argument('--debug', '-d',
                        action='store_true',
                        help='Print debugging output')
    args = parser.parse_args()
    computeClustering(args)

