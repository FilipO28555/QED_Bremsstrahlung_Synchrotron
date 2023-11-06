import numpy as np
epsilon0 = 1 #8.854187817e-12 # F/m
c = 1 #299792458 # m/s


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
dbdt = np.array([0,0.1,a_e])
q = -1

na1 = 17
na2 = 13

def Plot_Larmor(b,a_angle,save=False,norm_factor=1):
    dbdt = np.array([0,np.cos(a_angle)*a_e,np.sin(a_angle)*a_e])


    #plot
    angle1 = np.linspace(0,2*np.pi,na1)
    angle2 = np.linspace(0,np.pi,na2)

    P = []
    for theta in angle1:
        for phi in angle2:
            n = np.array([np.sin(phi)*np.cos(theta),np.sin(phi)*np.sin(theta),np.cos(phi)])
            P.append(n*Larmor_formula(n,b,dbdt,q))
    P = np.array(P)
    #normalize to max power radiated
    if norm_factor == 1:
        P = P/np.linalg.norm(np.max(P))
    else:
        P = P/norm_factor
    #create 3d plot - the higher power radiated the longer the arrow from center
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    #rotate by 85deg around y axis
    ax.view_init(5, 0)
    ax.set_xlim([-1,1])
    ax.set_ylim([-1,1])
    ax.set_zlim([-1,1])

    ax.quiver(np.zeros(len(P)),np.zeros(len(P)),np.zeros(len(P)),P[:,0],P[:,1],P[:,2],length=1)
    #plot b and dbdt in the corner: (1,1,0)
    ax.quiver([0.5,0.5],[0.5,0.5],[0,0],[b[0],dbdt[0]],[b[1],dbdt[1]],[b[2],dbdt[2]],length=1,color=['r','g'])

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    
    if save:
        plt.savefig(f'Plots/Larmor_{int(100*a_angle)}.png',dpi=300)
    else:
        plt.show()
    plt.close()

def get_norm_factor(b,a_angle):
    dbdt = np.array([0,np.cos(a_angle)*a_e,np.sin(a_angle)*a_e])
    angle1 = np.linspace(0,2*np.pi,na1)
    angle2 = np.linspace(0,np.pi,na2)

    P = []
    for theta in angle1:
        for phi in angle2:
            n = np.array([np.sin(phi)*np.cos(theta),np.sin(phi)*np.sin(theta),np.cos(phi)])
            P.append(n*Larmor_formula(n,b,dbdt,q))
    P = np.array(P)
    return np.linalg.norm(np.max(P))

#get max norm factor
max_norm_factor =0
for a in np.linspace(0,2*np.pi,80):
    if get_norm_factor(b,a) > max_norm_factor:
        max_norm_factor = get_norm_factor(b,a)

# for a in np.linspace(0,2*np.pi,80):
#     Plot_Larmor(b,a,save=True,norm_factor=max_norm_factor)
# Plot_Larmor(b,np.pi/4)



def makeGif(folder = 'Plots'):
    import imageio
    import os
    images = []
    files = os.listdir(folder)
    #sort files by number
    files = sorted(files, key=lambda x: int(x.split('_')[1].split('.')[0]))
    for filename in files:
        images.append(imageio.imread(f'{folder}/{filename}'))
    imageio.mimsave('Larmor.gif',images,fps=10, loop=0)
    
makeGif("Plots")
    




    
    
    