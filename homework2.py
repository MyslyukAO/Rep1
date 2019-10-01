import numpy as np
import math
import time
np.random.seed(int(time.time()))

def task1():
    arr = np.random.random((10,3))
    print("Arr:\n", arr)
    args = np.argmin(np.abs(arr-0.5), axis=1)
    res = arr[range(10),args]
    print("Results:", res)

def task2():
    arr = np.random.random((6,6))
    print("Arr:\n", arr)
    s = np.sum(arr, axis=1)
    m = np.min(arr, axis=0)
    res = s/m
    print("Results:", res)

def task3():
    arr = np.array([6, 2, 0, 3, 0, 0, 5, 7, 0])
    print("Arr:\n", arr)
    mask = np.concatenate(([False], (arr==0)[:-1]))
    maxEl = np.max(arr[mask])
    print("Results:", maxEl)

def task4():
    x, i = np.ones(10),  np.array([0, 1, 2, 3, 5, 5, 5, 8])
    print('x:\n', x)
    print('i:\n', i)
    counts = np.bincount(i)
    counts.resize((10,))
    x += counts  
    print("Results:", x)

def task5():
    arr = np.arange(16).reshape(4, 4)
    print("Arr:\n", arr)
    rotated = np.rot90(arr) 
    iterator = np.arange(sum(arr.shape)-1) - arr.shape[0]+1 
    diagonals = map(lambda x: list(rotated.diagonal(x)), iterator) 
    res = dict(zip(np.arange(sum(arr.shape)-1), diagonals)) 
    print("Results:", res)

class task6():
    @classmethod
    def kmeans(cls, obs, k_or_guess, iter=20, thresh=1e-05):
        def sub_outer(a, b):
            return a[:, None] - b[None, :]
        def labels_n_distortion(obs, codebook):
            sustractedCubeMatrix = sub_outer(obs, codebook) 
            dist = np.linalg.norm(sustractedCubeMatrix, axis=2)

            labels = np.argmin(dist, axis=1)
            distortion = np.sum(dist[np.arange(dist.shape[0]), labels]) 
            return labels, distortion
        if type(k_or_guess) is int:
            stats = (obs.min(axis=0), obs.max(axis=0))
            def rnd_codebook():
                return np.random.rand(k_or_guess, obs.shape[1])*(stats[1]-stats[0]) + stats[0]

            resLabels, resCodebook, resDistortion = task6.kmeans(obs, rnd_codebook(), thresh=thresh)
            for i in range(iter):
                tryLabels, tryCodebook, tryDistortion = task6.kmeans(obs, rnd_codebook(), thresh=thresh)
                if tryDistortion < resDistortion: 
                    resCodebook = tryCodebook
                    resDistortion = tryDistortion
                    resLabels = tryLabels
            return resLabels, resCodebook, resDistortion 
        else:
            codebook = k_or_guess 
        labels, distortion = labels_n_distortion(obs, codebook)
        oldDistortion = distortion + 2*thresh
        while abs(oldDistortion-distortion) > thresh:
            codebook = np.array(list(map(lambda x: np.mean(obs[labels==x], axis=0), range(len(codebook)))))

            oldDistortion = distortion
            labels, distortion = labels_n_distortion(obs, codebook)
        return labels, codebook, distortion

    def __init__(self):    
        data = np.loadtxt('http://www.stat.cmu.edu/~larry/all-of-statistics/=data/faithful.dat', skiprows=26)[:, 1:]
        labels, centroids, distortion = task6.kmeans(data, 2)
        print(centroids)





if __name__ == "__main__":
    taskN = int(input('Task N: '))
    if taskN == 1:
        task1()
    if taskN == 2:
        task2()
    if taskN == 3:
        task3()
    if taskN == 4:
        task4()
    if taskN == 5:
        task5()
    if taskN == 6:
        task6()
