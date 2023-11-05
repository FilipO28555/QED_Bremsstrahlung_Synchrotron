import numpy as np
epsilon0 = 8.854187817e-12 # F/m
c = 299792458 # m/s


# This program is to visualize the Larmor formula for the angle of the radiation
# n: vector from electron to observer
# b: beta - velocity of electron / speed of light. Vector
# dbdt: time derivative of beta. Vector
# q: charge of electron
#returns: dP/domega - power radiated per unit solid angle
def Larmor_formula(n,b,dbdt,q):
    global epsilon0, c
    br1 = n-b
    br2 = np.cross(br1,dbdt)
    br3 = np.cross(n,br2)
    numerator = np.linalg.norm(br3)**2 * q**2
    denominator = 16*np.pi**2*epsilon0*c*(1 - np.dot(n,b))**5
    return numerator/denominator


v_e = 0.9
a_e = 0.5


b = np.array([0,0,v_e])
dbdt = np.array([0,0,a_e])
q = -1

#plot
angle1 = np.linspace(0,2*np.pi,15)
angle2 = np.linspace(0,np.pi,13)

P = []
for theta in angle1:
    for phi in angle2:
        n = np.array([np.sin(phi)*np.cos(theta),np.sin(phi)*np.sin(theta),np.cos(phi)])
        P.append(n*Larmor_formula(n,b,dbdt,q))
P = np.array(P)
#normalize to max power radiated
P = P/np.linalg.norm(np.max(P))*10
#create 3d plot - the higher power radiated the longer the arrow from center
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
#rotate by 85deg around y axis
ax.view_init(35, 85)
ax.set_xlim([-1,1])
ax.set_ylim([-1,1])
ax.set_zlim([-1,1])

ax.quiver(np.zeros(len(P)),np.zeros(len(P)),np.zeros(len(P)),P[:,0],P[:,1],P[:,2],length=0.1)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()

    
        







    
    
    