#!/usr/bin/env python
# coding: utf-8

# In[193]:


# Code to find the maximum element of 8th column of an (N x 8) matrix using Grey Wolf Optimization (GWO) Algorithm


# In[ ]:


# GWO code


# In[68]:


import numpy as np
import random

# Load the matrix from the file
matrix = np.loadtxt('PSO_Test_23_9.txt', delimiter=',')

# Calculate lb and ub based on the minimum and maximum values in the first 7 columns
lb = np.min(matrix[:, :7], axis=0)
ub = np.max(matrix[:, :7], axis=0)

# Initialize parameters
N = 10 # Number of wolves
D = 7 # Dimension (number of columns accessible)
max_iterations = 20 # Total number of iterations

# Initialize wolves and best solution
X = []  # List of wolf positions for all 10 wolves
f = []  # List of wolf values for all 10 wolves

for i in range(N):
    random_row = random.randint(0, len(matrix) - 1) # Initialize random positions for each wolf based on rows in the matrix
    X.append(matrix[random_row, :7])  # Each wolf position= 1 row of the matrix (first 7 columns)
    f.append(float('-inf'))  # Initially all wolf values= -infinity, so that they can always be bettered

X_best = X[0].copy()
f_best = float('-inf')

# Define a function to calculate Euclidean distance between two arrays
def euclidean_distance(arr1, arr2):
    return np.linalg.norm(arr1 - arr2)

# Main GWO loop
for iteration in range(max_iterations):
    a = 2 * (1 - (iteration / max_iterations))
    
    # Find the positions of alpha, beta, and gamma wolves
    index_alpha = np.argmax(f)
    X_alpha = X[index_alpha].copy() # Alpha wolf position

    f[index_alpha] = float('-inf')  # Remove f-value of alpha from the list `f'

    index_beta = np.argmax(f)
    X_beta = X[index_beta].copy() # Beta wolf position

    f[index_beta] = float('-inf')  # Remove f-value of beta from the list `f'
    
    index_gamma = np.argmax(f)
    X_gamma = X[index_gamma].copy() # Gamma wolf position
        
    # Update each wolf's position
    for i in range(N):
        # Update positions
        A1, A2, A3 = 2 * a * np.random.rand(3) - a # Three random numbers, each of the form 2a.rand()-a
        C1, C2, C3 = 2 * np.random.rand(3) # Three random numbers, each of the form 2.rand()

        D_alpha = np.abs(C1 * X_alpha - X[i])
        D_beta = np.abs(C2 * X_beta - X[i])
        D_gamma = np.abs(C3 * X_gamma - X[i])

        X1 = X_alpha - A1 * D_alpha
        X2 = X_beta - A2 * D_beta
        X3 = X_gamma - A3 * D_gamma

        X_new = (X1 + X2 + X3) / 3 # Formula for new position

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
        
        # Update the wolf value if current value is better
        if f_new > f[i]:
            f[i] = f_new
            X[i] = X_current

    # Find the best solution among all wolves
    max_index = np.argmax(f)
    
    # Update best solution and fitness value
    if f[max_index] > f_best:
        X_best = X[max_index].copy()
        f_best = f[max_index]
        
    print(f_best) # Prints f_best for each iteration
    
# After the loop, f_best will contain the maximum value found
print(f"Maximum value in the 8th column: {f_best}")


# In[ ]:





# In[67]:


# AVERAGE FOR 100 RUNS

import numpy as np
import random

# Load the matrix from the file
matrix = np.loadtxt('PSO_Test_23_9.txt', delimiter=',')

# Calculate lb and ub based on the minimum and maximum values in the first 7 columns
lb = np.min(matrix[:, :7], axis=0)
ub = np.max(matrix[:, :7], axis=0)

# Initialize parameters
N = 10 # Number of wolves
D = 7 # Dimension (number of columns accessible)
max_iterations = 20 # Total number of iterations

# Define a function to calculate Euclidean distance between two arrays
def euclidean_distance(arr1, arr2):
    return np.linalg.norm(arr1 - arr2)

runs = 100 # Define the number of runs
max_values = [] # Initialize a list to store maximum values for each run

# Run the GWO algorithm multiple times
for run in range(runs):
    # Initialize wolves and best solution
    X = []  # List of wolf positions for all 10 wolves
    f = []  # List of wolf values for all 10 wolves

    for i in range(N):
        random_row = random.randint(0, len(matrix) - 1) # Initialize random positions for each wolf based on rows in the matrix
        X.append(matrix[random_row, :7])  # Each wolf position= 1 row of the matrix (first 7 columns)
        f.append(float('-inf'))  # Initially all wolf values= -infinity, so that they can always be bettered

    X_best = X[0].copy()
    f_best = float('-inf')
    
    # Main GWO loop
    for iteration in range(max_iterations):
        a = 2 * (1 - (iteration / max_iterations))
    
        # Find the positions of alpha, beta, and gamma wolves
        index_alpha = np.argmax(f)
        X_alpha = X[index_alpha].copy() # Alpha wolf position

        f[index_alpha] = float('-inf')  # Remove f-value of alpha from the list `f'

        index_beta = np.argmax(f)
        X_beta = X[index_beta].copy() # Beta wolf position

        f[index_beta] = float('-inf')  # Remove f-value of beta from the list `f'
    
        index_gamma = np.argmax(f)
        X_gamma = X[index_gamma].copy() # Gamma wolf position
        
        # Update each wolf's position
        for i in range(N):
            # Update positions
            A1, A2, A3 = 2 * a * np.random.rand(3) - a # Three random numbers, each of the form 2a.rand()-a
            C1, C2, C3 = 2 * np.random.rand(3) # Three random numbers, each of the form 2.rand()

            D_alpha = np.abs(C1 * X_alpha - X[i])
            D_beta = np.abs(C2 * X_beta - X[i])
            D_gamma = np.abs(C3 * X_gamma - X[i])

            X1 = X_alpha - A1 * D_alpha
            X2 = X_beta - A2 * D_beta
            X3 = X_gamma - A3 * D_gamma

            X_new = (X1 + X2 + X3) / 3 # Formula for new position

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
        
            # Update the wolf value if current value is better
            if f_new > f[i]:
                f[i] = f_new
                X[i] = X_current

        # Find the best solution among all wolves
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




