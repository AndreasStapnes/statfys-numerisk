import numpy as np
k_b = 1.38e-23          #The boltzmann constant
#T = 293                 #Example temperature (in Kelvin)
beta = 5.1     #The inverse-temperature-constant for the example-temp
T = 1/(beta*k_b)

box_k = 300       #Spring-constant for box-walls

hardcore_diameter = 2e-2        #Particle-diameter for hardcore interaction
hardcore_pot = 100000*k_b*T           #Particle-potential for hardcore interaction

N = 1000                        #Example amount of particles

epsilon = 1                    #Lennard-jones potential-constant
a = 0.1                         #Lennard-jones distance-constant
