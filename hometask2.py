import numpy as np
import math

def task1():
    	array=np.random.random((10,3))
	print(array,end='\n\n')
	arrayD=np.abs(array-0.5)
	res=np.zeros(10)
	for i in range(len(arrayD[:,0])):
		res[i]=array[i,arrayD[i].argmin()]
	print(res)
def task2():
    arr = np.random.random((6,6))
    print(arr, end='\n\n')
    res = np.zeros(len(arr))
    for i in range(len(arr)):
        res[i] = sum(arr[i])/min(arr[:,i])
    print(res)

def task3():
    arr = np.array([6, 2, 0, 3, 0, 0, 5, 7, 0])
    print(arr, end='\n\n')
    maxEl = 0
    for i in range(1, len(arr)):
        if maxEl < arr[i] and arr[i-1] == 0:
            maxEl = arr[i]
    print(maxEl)

def task4():
    x, i = np.ones(10),  np.array([0, 1, 2, 3, 5, 5, 5, 8])
    print('x=', x, end='\n\n')
    prin	t('i=', i, end='\n\n')
    for it in i:
        x[it] += 1
    print(x)

def task5():
    	a=np.arange(16).reshape(4,4)
	print(a,end='\n\n')
	res=dict()
	for i in range(len(a)):
		res[i]=[a[j,i-j] for j in range(i+1)]
	for i in range(1,len(a)):
		res[i+len(a)-1]=[a[i+j,len(a)-j-1] for j in range(0,len(a)-i)]
	print(res)

def task6():
    def kmeans(obs, k_or_guess, iter=20, thresh=1e-05):
        def sub_outer(a, b):
            return a[:, None] - b[None, :]
        def labels_n_distortion(obs, codebook):
            sustractedCubeMatrix = sub_out(obs, codebook)
            dist = np.linalg.norm(sustractedCubeMatrix, axis=0)

            labels = np.argmin(dist, axis=0)
            distortion = np.sum(dist[labels, range(dist.shape[1])])

            return labels, distortion

        if type(k_or_guess) is int:
            stats = (data.min(axis=0), data.max(axis=0))
            codebook = np.random.rand(k_or_guess, obs.shape[1])*(stats[1]-stats[0]) + stats[0]
            labels, distortion = labels_n_distortion(obs, codebook)
            for i in range(iter):
                tryCodebook = np.random.rand(k_or_guess, obs.shape[1])*(stats[1]-stats[0]) + stats[0]
                labels, tryDistortion = labels_n_distortion(obs, tryCodebook)
                if tryDistortion < distortion:
                    codebook = tryCodebook
                    distortion = tryDistortion
        else:
            codebook = k_or_guess

        labels, distortion = labels_n_distortion(obs, codebook)
        while distortion > thresh:
            codebook = np.array([np.mean(obs[labels==i], axis=0), for i in range(len(codebook))])
            labels, distortion = labels_n_distortion(obs, codebook)
        return codebook, distortion
        




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
