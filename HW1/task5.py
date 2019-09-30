import numpy as np

a=np.arange(16).reshape(4,4)
print(a,end='\n\n')
res=dict()
for i in range(len(a)):
	res[i]=[a[j,i-j] for j in range(i+1)]
for i in range(1,len(a)):
	res[i+len(a)-1]=[a[i+j,len(a)-j-1] for j in range(0,len(a)-i)]
print(res)
