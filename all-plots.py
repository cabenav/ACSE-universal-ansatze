# plot all loss file in given folder
import os
import glob
fig_folder='fig'
result_folder='checkpoints'
filelist = glob.glob(f'{result_folder}/*loss.pt')
#print(filelist)


from threading import Thread

def plot(f): #f is the name of the data file to be plotted
    os.system(f'make f={f} plot')
    
PLOT_ALL=False # True for plotting all; False for new data only

import time
#now = time.time()
#delta = 600 # seconds
thread_pool=[]
plotted=[]
skipped=[]
for f in filelist:
    #print(f)
    t=os.path.getmtime(f)
    filename=f.split('/')[-1]
    filename_fig=f'{fig_folder}/{filename}.pdf'
    if os.path.exists(filename_fig):
        t_fig = os.path.getmtime(filename_fig)
    else:
        t_fig = t-1 #plot for newly-created data file
    if PLOT_ALL or t > t_fig:
        print(f'data file updated for {filename_fig}. replotting...')
        #        continue
        #    if False or (now - t) < delta:        
        #print(t,f)
        thread = Thread(target = plot, args = (f, ))
        thread_pool.append(thread)
        thread.start()        
        #os.system(f'make f={f} plot')
        plotted.append(f)
    else:
        print(f'skip {f}')
        skipped.append(f)

for thread in thread_pool:
    thread.join()

print('#'*120)
print('skipped')
print('\n'.join(skipped))
print('finish plotting for:')
print('\n'.join(plotted))
print('#'*120)
