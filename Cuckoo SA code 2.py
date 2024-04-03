#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Code to find the maximum element of 8th column of an (N x 8) matrix using Cuckoo Search Algorithm


# In[ ]:


# Cuckoo SA code


# In[129]:


import numpy as np
import random
from scipy.special import gamma

# Load the matrix from the file
matrix = np.loadtxt('PSO_Test_23_9.txt', delimiter=',')

# Calculate lb and ub based on the minimum and maximum values in the first 7 columns
lb = np.min(matrix[:, :7], axis=0)
ub = np.max(matrix[:, :7], axis=0)

# Parameters
N = 10  # Number of cuckoos
D = 7  # Dimension (number of columns accessible)
pa = 0.4  # Probability of abandoning a nest
beta = 1.5
iterations = 20  # Number of iterations
# For each cuckoo we make 1 function call per iteration. So, a total of 200 function calls made

# Calculate sigma
sigma = ((gamma(1 + beta) * np.sin(np.pi * beta / 2)) / (gamma((1 + beta) / 2) * beta * (2**((beta - 1) / 2))))**(1/beta)

# Initialize cuckoos and best solution
X = []  # List of cuckoo positions for all 10 cuckoos
f = []  # List of cuckoo values for all 10 cuckoos

for i in range(N):
    random_row = random.randint(0, len(matrix) - 1)  # Initialize random positions for each cuckoo based on rows in the matrix
    X.append(matrix[random_row, :7]) # Each cuckoo position= 1 row of the matrix (first 7 columns)
    f.append(float('-inf')) # Initially all cuckoo values= -infinity, so that they can always be bettered

X_best = X[0].copy()
f_best = -float("inf")

# Define a function to calculate Euclidean distance between two arrays
def euclidean_distance(arr1, arr2):
    return np.linalg.norm(arr1 - arr2)

# Main Cuckoo Search loop
for t in range(iterations):
    for i in range(N):
        # Generate random steps
        u = np.random.randn(D) * sigma
        v = np.random.randn(D)
        s = u / (np.linalg.norm(v)**(1/beta))
        
        # Calculate new position
        X_new = X[i] + np.random.randn(D) * 0.01 * s * (X[i] - X_best)
        
        # Keep X_new within bounds
        for j in range(D):
            X_new[j] = max(lb[j], min(X_new[j], ub[j]))
            
        # Choose a random no. r from 0 to 1 and check if it is less than pa
        r = np.random.rand()
        if r < pa:
            d1, d2 = np.random.randint(0, N, 2)
            X_new = X[i] + np.random.rand(D) * (X[d1] - X[d2])
            
            # Keep X_new within bounds
            for j in range(D):
                X_new[j] = max(lb[j], min(X_new[j], ub[j]))
    
        # Find the row in the matrix with the minimum Euclidean distance to the new position
        distances = [euclidean_distance(X_new, row[:7]) for row in matrix]
        min_distance_index = np.argmin(distances)
        
        # Update current position
        X_current = matrix[min_distance_index, :7]

        # Update current value to be the value from the 8th column of the selected row
        f_new = matrix[min_distance_index, 7] # This is where function call happens
        
        # Update if the new solution is better
        if f_new > f[i]:
            f[i] = f_new
            X[i] = X_current
        
    # Find the cuckoo with the best fitness
    max_index = np.argmax(f)
    
    # Update best solution and fitness value
    if f[max_index] > f_best:
        X_best = X[max_index].copy()
        f_best = f[max_index]
    
# After the loop, f_best will contain the maximum value found
print(f'Maximum value in the 8th column: {f_best}')


# In[ ]:





# In[128]:


# AVERAGE FOR 100 RUNS

import numpy as np
import random
from scipy.special import gamma

# Load the matrix from the file
matrix = np.loadtxt('PSO_Test_23_9.txt', delimiter=',')

# Calculate lb and ub based on the minimum and maximum values in the first 7 columns
lb = np.min(matrix[:, :7], axis=0)
ub = np.max(matrix[:, :7], axis=0)

# Parameters
N = 10  # Number of cuckoos
D = 7  # Dimension (number of columns accessible)
pa = 0.4  # Probability of abandoning a nest
beta = 1.5
iterations = 20  # Number of iterations
# For each cuckoo we make 1 function call per iteration. So, a total of 200 function calls made

# Calculate sigma
sigma = ((gamma(1 + beta) * np.sin(np.pi * beta / 2)) / (gamma((1 + beta) / 2) * beta * (2**((beta - 1) / 2))))**(1/beta)

runs = 100 # Define the number of runs
max_values = [] # Initialize a list to store maximum values for each run

# Define a function to calculate Euclidean distance between two arrays
def euclidean_distance(arr1, arr2):
    return np.linalg.norm(arr1 - arr2)

# Run the Cuckoo Search algorithm multiple times
for run in range(runs):
    # Initialize cuckoos and best solution
    X = []  # List of cuckoo positions for all 10 cuckoos
    f = []  # List of cuckoo values for all 10 cuckoos

    for i in range(N):
        random_row = random.randint(0, len(matrix) - 1)  # Initialize random positions for each cuckoo based on rows in the matrix
        X.append(matrix[random_row, :7]) # Each cuckoo position= 1 row of the matrix (first 7 columns)
        f.append(float('-inf')) # Initially all cuckoo values= -infinity, so that they can always be bettered

    X_best = X[0].copy()
    f_best = -float("inf")
    
    # Main Cuckoo Search loop
    for t in range(iterations):
        for i in range(N):
            # Generate random steps
            u = np.random.randn(D) * sigma
            v = np.random.randn(D)
            s = u / (np.linalg.norm(v)**(1/beta))
        
            # Calculate new position
            X_new = X[i] + np.random.randn(D) * 0.01 * s * (X[i] - X_best)
        
            # Keep X_new within bounds
            for j in range(D):
                X_new[j] = max(lb[j], min(X_new[j], ub[j]))
            
            # Choose a random no. r from 0 to 1 and check if it is less than pa
            r = np.random.rand()
            if r < pa:
                d1, d2 = np.random.randint(0, N, 2)
                X_new = X[i] + np.random.rand(D) * (X[d1] - X[d2])
            
                # Keep X_new within bounds
                for j in range(D):
                    X_new[j] = max(lb[j], min(X_new[j], ub[j]))
    
            # Find the row in the matrix with the minimum Euclidean distance to the new position
            distances = [euclidean_distance(X_new, row[:7]) for row in matrix]
            min_distance_index = np.argmin(distances)
        
            # Update current position
            X_current = matrix[min_distance_index, :7]

            # Update current value to be the value from the 8th column of the selected row
            f_new = matrix[min_distance_index, 7] # This is where function call happens
        
            # Update if the new solution is better
            if f_new > f[i]:
                f[i] = f_new
                X[i] = X_current
        
        # Find the cuckoo with the best fitness
        max_index = np.argmax(f)
    
        # Update best solution and fitness value
        if f[max_index] > f_best:
            X_best = X[max_index].copy()
            f_best = f[max_index]
            
    # Append the maximum value from this run to the max_values list
    max_values.append(f_best)
    
# Calculate and print the average of maximum values over all runs
average_max_value = np.mean(max_values)
print(f'Average maximum value over {runs} runs: {average_max_value}')


# In[ ]:




