# plot all loss file in given folder
import os
import glob

result_folder='checkpoints'
filelist = glob.glob(f'{result_folder}/*loss.pt')
#print(filelist)

import time
now = time.time()
delta = 600 # seconds
plotted=[]
for f in filelist:
    #print(f)
    t=os.path.getmtime(f)
    if (now - t) < delta:        
        #print(t,f)
        os.system(f'make f={f} plot')
        plotted.append(f)

print('finish plotting for:\n','\n'.join(plotted))
