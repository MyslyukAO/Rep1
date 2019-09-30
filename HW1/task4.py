import numpy as np
x=np.ones(10)
i=np.array([0,1,2,3,5,5,5,8])
print('x=',x,end='\n\n')
print('i=',i,end='\n\n')
for it in i:
	x[it]+=1
print(x)
