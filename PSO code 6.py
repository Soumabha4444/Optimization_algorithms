#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Code to find the maximum element of 8th column of an (N x 8) matrix using PSO (particle swarm optimization)


# In[ ]:


# PSO code


# In[15]:


import numpy as np
import random

matrix = np.loadtxt('PSO_Test_23_9.txt', delimiter=',')
num_particles = 10 # Define the number of particles
num_iterations = 20 # Define the number of iterations
# Each particle does 1 function call per iteration. So, a total of 200 function calls made

# Initialize PSO parameters
c1 = 1.5
c2 = 1.5
w = 0.5

# Initialize particle positions, velocities, and personal best positions
particle_positions = [] # List of particle positions for all 10 particles
particle_velocities = [] # List of particle velocities for all 10 particles
pbest_positions = [] # List of pbest positions for all 10 particles
pbest_values = [] # List of pbest values for all 10 particles

for particle in range(num_particles):
    random_row = random.randint(0, len(matrix) - 1) # Initialize random positions for each particle based on rows in the matrix
    particle_positions.append(matrix[random_row, :7]) # Each particle position= 1 row of the matrix (first 7 columns)
    particle_velocities.append(np.random.uniform(-0.1, 0.1, size=7)) # Initialize random velocities for each particle
    pbest_positions.append(particle_positions[-1].copy()) # Initially set personal best positions to be the same as initial positions
    # [-1] is used to access the last element (i.e. the current position) of the particle_positions list
    pbest_values.append(float('-inf')) # Initially all pbest values= -infinity, so that they can always be bettered
    
# Initialize global best position and value
gbest_position = particle_positions[0].copy() # Initialize gbest_position with one of the particle positions to avoid NoneType error
gbest_value = float('-inf') # so that it can always be bettered

# Define a function to calculate Euclidean distance between two arrays
def euclidean_distance(arr1, arr2):
    return np.linalg.norm(arr1 - arr2)

# Main PSO loop
for iteration in range(num_iterations):
    for i in range(num_particles):
        # Calculate the Euclidean distances between the current particle position and all rows of the matrix
        distances = [euclidean_distance(particle_positions[i], row[:7]) for row in matrix]
        
        # Find the index of the row with the minimum distance
        min_distance_index = np.argmin(distances)
        
        # Update current position to be the row with the minimum distance
        current_position = matrix[min_distance_index, :7]
        
        # Update current value to be the value from the 8th column of the selected row
        current_value = matrix[min_distance_index, 7] # this is where function call happens, and we extract 8th column value of a row
        
        # Update personal best position if needed
        if current_value > pbest_values[i]: # i.e. if current value is better than pbest value of the particle
            pbest_positions[i] = current_position.copy()
            pbest_values[i] = current_value
        
        # Update the global best position and value
        if pbest_values[i] > gbest_value:
                gbest_value = pbest_values[i]
                gbest_position = pbest_positions[i].copy()
                
        # Update the particle's velocity and position
        r = np.random.rand(2)
        inertia = w * particle_velocities[i]
        cognitive = c1 * r[0] * (pbest_positions[i] - particle_positions[i])
        social = c2 * r[1] * (gbest_position - particle_positions[i])
        particle_velocities[i] = inertia + cognitive + social
        particle_positions[i] = particle_positions[i] + particle_velocities[i]

# After the loop, global_best_value will contain the maximum value from the 8th column
print(f'Maximum value in the 8th column: {gbest_value}')


# In[19]:


# AVERAGE FOR 100 RUNS

import numpy as np
import random

matrix = np.loadtxt('PSO_Test_23_9.txt', delimiter=',')
num_particles = 10 # Define the number of particles
num_iterations = 10 # Define the number of iterations
# Each particle does 1 function call per iteration. So, a total of 200 function calls made
num_runs = 100 # Define the number of runs

# Initialize PSO parameters
c1 = 1.5
c2 = 1.5
w = 0.5
max_values = [] # Initialize a list to store maximum values for each run

# Define a function to calculate Euclidean distance between two arrays
def euclidean_distance(arr1, arr2):
    return np.linalg.norm(arr1 - arr2)

# Run the PSO algorithm multiple times
for run in range(num_runs):
    # Initialize particle positions, velocities, and personal best positions
    particle_positions = [] # List of particle positions for all 10 particles
    particle_velocities = [] # List of particle velocities for all 10 particles
    pbest_positions = [] # List of pbest positions for all 10 particles
    pbest_values = [] # List of pbest values for all 10 particles

    for particle in range(num_particles):
        random_row = random.randint(0, len(matrix) - 1) # Initialize random positions for each particle based on rows in the matrix
        particle_positions.append(matrix[random_row, :7]) # Each particle position= 1 row of the matrix (first 7 columns)
        particle_velocities.append(np.random.uniform(-0.1, 0.1, size=7)) # Initialize random velocities for each particle
        pbest_positions.append(particle_positions[-1].copy()) # Initially set personal best positions to be the same as initial positions
        # [-1] is used to access the last element (i.e. the current position) of the particle_positions list
        pbest_values.append(float('-inf')) # Initially all pbest values= -infinity, so that they can always be bettered
    
    # Initialize global best position and value
    gbest_position = particle_positions[0].copy() # Initialize gbest_position with one of the particle positions to avoid NoneType error
    gbest_value = float('-inf') # so that it can always be bettered

    # Main PSO loop
    for iteration in range(num_iterations):
        for i in range(num_particles):
            # Calculate the Euclidean distances between the current particle position and all rows of the matrix
            distances = [euclidean_distance(particle_positions[i], row[:7]) for row in matrix]
        
            # Find the index of the row with the minimum distance
            min_distance_index = np.argmin(distances)
        
            # Update current position to be the row with the minimum distance
            current_position = matrix[min_distance_index, :7]
        
            # Update current value to be the value from the 8th column of the selected row
            current_value = matrix[min_distance_index, 7] # this is where function call happens, and we extract 8th column value of a row
        
            # Update personal best position if needed
            if current_value > pbest_values[i]: # i.e. if current value is better than pbest value of the particle
                pbest_positions[i] = current_position.copy()
                pbest_values[i] = current_value
        
            # Update the global best position and value
            if pbest_values[i] > gbest_value:
                    gbest_value = pbest_values[i]
                    gbest_position = pbest_positions[i].copy()
                
            # Update the particle's velocity and position
            r = np.random.rand(2) # Generate 2 random numbers between [0,1)
            inertia = w * particle_velocities[i]
            cognitive = c1 * r[0] * (pbest_positions[i] - particle_positions[i])
            social = c2 * r[1] * (gbest_position - particle_positions[i])
            particle_velocities[i] = inertia + cognitive + social
            particle_positions[i] = particle_positions[i] + particle_velocities[i]

    # Append the maximum value from this run to the max_values list
    max_values.append(gbest_value)

# Calculate and print the average of maximum values over all runs
average_max_value = np.mean(max_values)
print(f'Average maximum value over {num_runs} runs: {average_max_value}')


# In[ ]:





# In[8]:


# Random search


# In[17]:


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


# In[20]:


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




