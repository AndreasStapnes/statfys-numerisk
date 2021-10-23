import numpy as np
k_b = 1.38e-23          #The boltzmann constant
T = 293                 #Example temperature (in Kelvin)
beta = 1 / (k_b * T)    #The inverse-temperature-constant for the example-temp

box_k = 30000*k_b*T       #Spring-constant for box-walls

hardcore_diameter = 2e-2        #Particle-diameter for hardcore interaction
hardcore_pot = 100000*k_b*T           #Particle-potential for hardcore interaction

N = 1000                        #Example amount of particles

epsilon = 10*k_b*T                     #Lennard-jones potential-constant
a = 0.1                         #Lennard-jones distance-constant
