import numpy as np
array=np.array([6,2,0,3,0,0,5,7,0])
print(array,end='\n\n')
maxEL=0
for i in range(1,len(array)):
	if maxEL<array[i] and array[i-1]==0:
		maxEL=array[i]
print(maxEL)
