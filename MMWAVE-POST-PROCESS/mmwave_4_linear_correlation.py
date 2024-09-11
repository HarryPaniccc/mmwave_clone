import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import glob
import os
def func(x,a,b):
    return a*x+b


# exp 10 lots of movement
# exp 12 lots of movement


plt.rcParams['lines.linewidth'] = 3
plt.rcParams['font.size'] = 13

working_dir = os.path.dirname(__file__)
fig = plt.figure()
ax = fig.add_subplot(111)

def accuracy(truth,measured):
    return 1 - np.mean((abs(truth-measured)/truth))
    
polar_total=[]
ftt_total=[]

for dog in ["Kasey","Sassy","Bullet","Paddy","Guster"]: # 
    polar_data=[]
    ftt_data=[]
    
    radar_paths = glob.glob(os.path.join(working_dir,"data","heart_data",dog,"Radar*.npy"))
    polar_paths = glob.glob(os.path.join(working_dir,"data","heart_data",dog,"Polar*.npy"))
    
    for r_path,p_path in zip(radar_paths,polar_paths):
        polar_data.extend(np.load(p_path))
        ftt_data.extend(np.load(r_path))
    
    ax.scatter(polar_data,ftt_data,s=10,label=dog)
    polar_total.extend(polar_data)
    ftt_total.extend(ftt_data)
    fig.show()



alpha = optimize.curve_fit(func, xdata = polar_total, ydata = ftt_total)[0]
gradient = alpha[0]
offset = alpha[1]
print("Gradient mapping: %.2f [bpm/bpm]" %gradient)
print("Offset mapping: %.2f [bpm]" %offset)
print("Pearson Coefficient: %.2f" % np.corrcoef(polar_total,ftt_total)[1][0])

x = np.linspace(20,200,180)

grid = np.zeros((len(x),len(x)))
for p_heart in range(len(x)):
    for r_heart in range(len(x)):
        grid[p_heart,r_heart] = 1 - (abs(x[p_heart]-x[r_heart])/x[p_heart])
        # print("GT %.2f | MD %.2f | Acc %.2f" % (x[p_heart],x[r_heart],grid[p_heart,r_heart]))

# grid = grid - np.min(grid)
# grid = grid/np.max(grid)
# grid = np.flipud(grid)
# ax.imshow(grid, extent=[20,200,20,200],alpha=0.5,aspect="auto") 


r_line = func(x,alpha[0],alpha[1])
ideal = func(x,1,0)
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.plot(x,r_line,c="orange",label="R-Line")
ax.plot(x,ideal,c="blue",label="Ideal")

ax.set_title("Polar Heart Rate vs Radar Heart Rate")
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

ax.set_xlim(20,200)
ax.set_xlabel("Polar Heart Rate [bpm]")

ax.set_ylim(20,200)
ax.set_ylabel("Radar Heart Rate [bpm]")



plt.show()