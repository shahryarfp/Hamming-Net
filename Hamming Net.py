#!/usr/bin/env python
# coding: utf-8

# In[10]:


import numpy as np
import copy


# ### Functions

# In[2]:


def Hamming_distance(v1,v2):
    distance = 0
    if len(v1)==len(v2):
        for i in range(len(v1)):
            if v1[i]!= v2[i]:
                distance += 1
    return distance

def print_hamming_dis(v1,v2):
    print('Hamming distance between', v1, 'and', v2, 'is:')
    print(Hamming_distance(v1,v2))

def relu(x):
    if x>=0:
        return x
    else:
        return 0
    
def is_finished(v):
    count = 0
    for i in range(len(v)):
        if v[i] == 0:
            count += 1
    if count >= len(v)-1:
        return True
    else:
        return False


# ### Defining vectors

# In[3]:


im_1 = [1,-1,1,-1,1,-1,-1,1,-1,1,-1,1] #X
im_2 = [1,-1,1,-1,1,-1,-1,1,-1,-1,1,-1] #Y
im_3 = [-1,1,-1,1,-1,1,1,1,1,1,-1,1] #A
im_4 = [-1,1,1,1,-1,-1,1,-1,-1,-1,1,1] #C
ims = np.array([im_1, im_2, im_3, im_4])


# ### Hamming distance between every two vectors

# In[4]:


print_hamming_dis(im_1, im_2)
print_hamming_dis(im_1, im_3)
print_hamming_dis(im_1, im_4)
print_hamming_dis(im_2, im_3)
print_hamming_dis(im_2, im_4)
print_hamming_dis(im_3, im_4)


# ## Hamming Net

# In[5]:


n = 12
m = 4
w = np.array([im_1, im_2, im_3, im_4]).T/2
b = np.ones([1,m])*(n/2)
print('Weight Matrix:\n',w)
print()
print('Bias:\n',b)


# ### Finding closest node to input

# In[8]:


x = np.array([1,1,1,-1,1,-1,-1,1,-1,-1,1,-1])

y_in = []
for i in range(m):
    y_in.append(np.matmul(x, w[:,i]) + b[0][i])

input_array = np.array(y_in)
epsilon = 0.15


# ### using Max-Net

# In[12]:


w = np.eye(len(input_array))*(1+epsilon) - epsilon
my_list = []
answer = []
temp_input = copy.deepcopy(input_array)

while not is_finished(temp_input):
    temp_input = np.matmul(w, temp_input.T)
    for i in range(len(temp_input)):
        temp_input[i] = relu(temp_input[i])
        if temp_input[i] == 0 and i not in my_list:
            my_list.append(i)            

for i in range(len(temp_input)):
    if temp_input[i] != 0 :
        my_list.append(i)

my_list = my_list[::-1]
for i in range(len(my_list)):
    answer.append(my_list[i])
    
print('y_in:', y_in,'\n')
max_ind = answer[0]
print('closest image to\n', x.reshape(4,3), '\nis:\n',ims[max_ind].reshape(4,3))


# In[ ]:




