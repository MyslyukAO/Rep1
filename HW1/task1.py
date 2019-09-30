import numpy as np
array=np.random.random((10,3))
print(array,end='\n\n')
arrayD=np.abs(array-0.5)
res=np.zeros(10)
for i in range(len(arrayD[:,0])):
	res[i]=array[i,arrayD[i].argmin()]
print(res)
