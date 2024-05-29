# plot all loss file in given folder
import os
import glob
fig_folder='fig'
result_folder='checkpoints'
filelist = glob.glob(f'{result_folder}/*loss.pt')
#print(filelist)

import time
#now = time.time()
#delta = 600 # seconds
plotted=[]
for f in filelist:
    #print(f)
    t=os.path.getmtime(f)
    filename=f.split('/')[-1]
    filename_fig=f'{fig_folder}/{filename}.pdf'
    t_fig = os.path.getmtime(filename_fig)
    if t > t_fig:
        print(f'data file updated for {filename_fig}. replotting...')
#        continue
#    if False or (now - t) < delta:        
        #print(t,f)
        os.system(f'make f={f} plot')
        plotted.append(f)

print('finish plotting for:')
print('\n'.join(plotted))
