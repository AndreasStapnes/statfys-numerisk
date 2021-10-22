k_b = 1.38e-23          #The boltzmann constant
T = 293                 #Example temperature (in Kelvin)
beta = 1 / (k_b * T)    #The inverse-temperature-constant for the example-temp

box_k = 200*k_b*T       #Spring-constant for box-walls

hardcore_diameter = 0e-2        #Particle-diameter for hardcore interaction
hardcore_pot = 1000 * k_b * T   #Particle-potential for hardcore interaction

N = 1000                        #Example amount of particles

epsilon = 1                     #Lennard-jones potential-constant
a = 0.1                         #Lennard-jones distance-constant
