import numpy as np
array=np.random.random((6,6))
print(array,end='\n\n')
res=np.zeros(6)
for i in range(len(array[1])):
	res[i]=sum(array[i])/min(array[:,i])
print(res)
