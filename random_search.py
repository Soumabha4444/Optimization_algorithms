#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Code to find the maximum element of 8th column of an (N x 8) matrix using random search


# In[2]:


import numpy as np

matrix = np.loadtxt('PSO_Test_23_9.txt', delimiter=',') # Load the matrix from the text file
num_samples = 200 # Number of random samples to try in random search
max_value = -float('inf') # Initialize max_value as -infinity so that it can always be bettered

# Random search loop
for sample in range(num_samples):
    row_index = np.random.randint(0, len(matrix) - 1) # Generate a random row index within the range of the matrix's rows
    value = matrix[row_index, 7] # Get the value in the 8th column for the randomly selected row
    
    # Update the maximum value if necessary
    if value > max_value:
        max_value = value

# Print the maximum value in the 8th column
print(f"Maximum value in the 8th column: {max_value}")


# In[14]:


# AVERAGE FOR 100 RUNS

import numpy as np
matrix = np.loadtxt('PSO_Test_23_9.txt', delimiter=',')
num_samples = 200
num_runs = 100 # Number of runs

# Initialize a list to store the maximum values in each run
list_max_values = []

# Perform random search for each run
for run in range(num_runs):
    max_value = -float('inf') 
    
    # Random search loop
    for sample in range(num_samples):
        row_index = np.random.randint(0, len(matrix) - 1)
        value = matrix[row_index, 7]
    
        # Update the maximum value if necessary
        if value > max_value:
            max_value = value
    
    # Append the maximum value from this run to the list
    list_max_values.append(max_value)

# Calculate the average of maximum values
average_max_value = np.mean(list_max_values)

# Print the average maximum value
print(f"Average maximum value in {num_runs} runs: {average_max_value}")


# In[ ]:




