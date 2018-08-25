
# coding: utf-8

# In[114]:


# AWS
drive_path = '/data/allen-brain-observatory/visual-coding-2p'

import numpy as np
import pandas as pd
import os
import sys
import h5py
import scipy
# import peakutils

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from allensdk.core.brain_observatory_cache import BrainObservatoryCache

manifest_file = os.path.join(drive_path,'manifest.json')
print(manifest_file)

boc = BrainObservatoryCache(manifest_file=manifest_file)


# In[186]:


#--- get all experiment containers with epileptiform tags ---#
exps = boc.get_experiment_containers(include_failed=True)
idx = []
for ii,_ in enumerate(exps):
    if len(exps[ii]['tags']):
        idx.append(ii)


# In[74]:


#--- get all experiment containers including the ones that failed the qc ---#
exps = boc.get_experiment_containers(include_failed=True)
norm_exps = boc.get_experiment_containers(cre_lines=['Scnn1a-Tg3-Cre'])


# In[144]:


tmp = boc.get_ophys_experiments(include_failed=True)


# In[121]:


print(len(exps))


# In[145]:


tmp=pd.DataFrame(tmp)
tmp[tmp['id']==511510998]


# In[125]:


pd.DataFrame(tmp)


# In[4]:


#--- iterate over all experiment containers and extract those with epileptiform tags ---#
idx = []
# iterate over all experiments
for ii,_ in enumerate(exps):
    # check if had tag 
    if len(exps[ii]['tags']):
        idx.append(ii)
        # print out tags
        print(exps[ii]['tags'])
print(idx)


# In[5]:


print(len(idx))


# In[187]:


#--- get session id for natural scene for specified container id ---#
# exp_container_id = exps[143]['id']
exp_container_id = exps[16]['id']
print(exp_container_id)
session_id = boc.get_ophys_experiments(include_failed=True,experiment_container_ids=[exp_container_id],stimuli=['natural_scenes'])[0]['id']
print(session_id)


# In[154]:


#--- get session id for natural scene for healthy mouse ---#
norm_exp_container_id = norm_exps[1]['id']
print(norm_exp_container_id)
norm_session_id = boc.get_ophys_experiments(experiment_container_ids=[norm_exp_container_id],stimuli=['natural_scenes'])[0]['id']
print(norm_session_id)


# In[155]:


#--- extract data set corresponding to session id defined above ---#
data_set = boc.get_ophys_experiment_data(ophys_experiment_id=session_id)
norm_data_set = boc.get_ophys_experiment_data(ophys_experiment_id=norm_session_id)


# In[156]:


#--- get dF/F trace, compute the mean across all cells and plot ---#
ts, dff = data_set.get_dff_traces()
mu_dff = np.mean(dff,axis=0)

ts, norm_dff = norm_data_set.get_dff_traces()
norm_mu_dff = np.mean(norm_dff,axis=0)


# In[157]:


fig, (ax1,ax2) = plt.subplots(2)
ax1.plot(mu_dff)
ax2.plot(norm_mu_dff)
xm = 4000
# ax.set_xlim(xm,xm+10000)
ax.set_ylabel('mean dF/F')
ax.set_xlabel('Frames')


# In[70]:


ts, pupil_size = data_set.get_pupil_size()
print(pupil_size.shape)

ts, norm_pupil_size = norm_data_set.get_pupil_size()
print(norm_pupil_size.shape)


# In[65]:


dxcm, ts = data_set.get_running_speed()
print(dxcm.shape)

norm_dxcm, ts = norm_data_set.get_running_speed()
print(norm_dxcm)


# In[41]:


help(data_set.get_running_speed)


# In[46]:


x = mu_dff
y = pupil_size
z = dxcm

f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
ax1.plot(x,'k')
ax2.plot(y,'k')
ax3.plot(z,'k')

ax1.set_title('Mean dF/F & Pupil Size')
# Fine-tune figure; make subplots close to each other and hide x ticks for
# all but bottom plot.
f.subplots_adjust(hspace=0)
plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
f.sav


# In[12]:


#--- determine threshold as 2 std away from mean ---#
thresh = np.mean(mu_dff)+(np.std(mu_dff)*2)
thresh_line = np.ones(len(mu_dff))*thresh
fs = 30. # sampling rate
dt = 1./fs
time = range(0,len(mu_dff))


# In[13]:


for ii in time:
    time[ii] = float(time[ii])*dt


# In[14]:


#--- find times of when signal went above threshold ---#
indices = [ii for ii,v in enumerate(mu_dff > thresh) if v]


# In[15]:


for ii,_ in enumerate(indices):
    indices[ii] = float(indices[ii])*dt


# In[16]:


plt.plot(time,mu_dff,'k')
plt.plot(time,thresh_line,'r')
plt.plot(indices,np.ones(len(indices))*0.4,'*')
plt.xlim(1400,1600)
# plt.xlim(0,10)


# In[17]:


# tmp = mu_dff[int(1200./dt):int(1250./dt)].copy()
tmp = mu_dff[int(1400./dt):int(1600./dt)].copy()


# In[18]:


plt.plot(tmp)
# plt.plot(ind,np.ones(len(ind))*0.4,'*r')


# In[19]:


ind = [i for i, v in enumerate(tmp>thresh) if v]
for ii,_ in enumerate(ind):
    ind[ii] = float(ind[ii])*dt
t2 = range(0,len(tmp))
for ii in t2:
    t2[ii] = float(t2[ii])*dt


# In[20]:


#--- function for computing the moving average ---#
def mavg_smooth(data,mwin):
    box_fun = np.ones(mwin)/mwin
    data_smooth = np.convolve(data,box_fun,mode='same')
    
    return data_smooth


# In[21]:


#--- smooth data using moving average ---#
smooth_tmp = mavg_smooth(tmp,10)


# In[22]:


plt.plot(t2,smooth_tmp,'k')


# In[23]:


tmp_thresh = np.mean(smooth_tmp)+(np.std(smooth_tmp)*2)
tmp_thresh_line = np.ones(len(smooth_tmp))*thresh
idx = [ii for ii, v in enumerate(smooth_tmp>thresh) if v]
for ii,_ in enumerate(idx):
    idx[ii] = float(idx[ii])*dt
plt.plot(t2,smooth_tmp,'k')
plt.plot(t2,tmp_thresh_line,'r')
plt.plot(idx,np.ones(len(idx))*0.35,'*')


# In[24]:


#--- event detection ---#
# this version would you a threshold to detect events
def event_detection(data,thresh,nonOverlap,dt):
    # find indicies for times when data was above thresh
    fp = [ii for ii, v in enumerate(data>thresh) if v]
    if(len(fp)):
        for jj,_ in enumerate(fp):
            fp[jj] = float(fp[jj])*dt
        dfp = np.diff(fp)
        fdfp = [ii for ii, v in enumerate(dfp>nonOverlap) if v]
        return fdfp
    else:
        return
    


# In[ ]:


#--- event detection ---#
# try using the events data to indicate the onset of a Ca event
def event_detection(data,thresh,nonOverlap,dt):
    # find indicies for times when data was above thresh
    fp = [ii for ii, v in enumerate(data>thresh) if v]
    if(len(fp)):
        for jj,_ in enumerate(fp):
            fp[jj] = float(fp[jj])*dt
        dfp = np.diff(fp)
        fdfp = [ii for ii, v in enumerate(dfp>nonOverlap) if v]
        return fdfp
    else:
        return


# In[25]:


st_times = event_detection(smooth_tmp,thresh,4,dt)


# In[26]:


type(st_times)


# In[34]:


# for ii,_ in enumerate(st_times):
#     st_times[ii] = float(st_times[ii])*dt
# plt.plot(t2,smooth_tmp,'k')
# plt.plot(t2,tmp_thresh_line,'r')
# plt.plot(st_times,np.ones(len(st_times))*0.35,'*')


# In[27]:


st_times


# In[28]:


float(st_times[0])*dt


# In[31]:


from swdb_2018_tools.events import get_events
events = get_events(session_id)


# In[33]:


events.shape


# In[75]:


smooth_events = mavg_smooth(events[0],20)
plt.plot(time,events[0])
plt.plot(time,smooth_events)
# plt.xlim(40,60)


# In[37]:


help(get_events)


# In[200]:


#--- function to load specified number of ophys movie frames ---#
def get_ophys_movie_frames(id, start_frame=0, end_frame=20000):
    movie_file_path = '/data/allen-brain-observatory/visual-coding-2p/ophys_movies/ophys_experiment_'
    movie_file = movie_file_path + str(id) + '.h5'
    with h5py.File(movie_file) as f:
        movie_frames = f['data'][start_frame:end_frame,:,:]
    
    return movie_frames


# In[151]:


#--- modified version of get_epilepsy_probability to handle passing data passed to it ---#
def get_epilepsy_prob(data, signal_threshold=10.0, min_width=0.1, max_width=0.3):
    """Get the probability of epilepsy in this physio movie.

    Args: 
        save_image:  A boolean specifying whether an image should be saved for qc reports
        start_frame, end_frame:  The start and end frames used to splice the movie for this metric
        signal_threshold:  A bound above which events are tagged as epileptic events
        min_width, max_width:  Bounds on the widths within which events are tagged as epileptic

    Returns:  
        A float representing the probability of epileptic events.
    """
    try:
        from oasis.functions import deconvolve as oasis_deconvolve
    except ImportError:
        logger.debug('OASIS not found. Install @ https://github.com/j-friedrich/OASIS.git.')
    
    # average frames and use deconvolitions to remove nosie from signal 
    avg_frams = np.mean(data,axis=(1,2))
    perc_diff = -100 * (1 - avg_frames / avg_frames.mean())
    denoised_signal, spike_events, _, _, _ = oasis_deconvolve(perc_diff, penalty=1)
        
    if set(denoised_signal) == {0.0}: # No signal
        print("Error: No signal")
        return 0

    prominence_data, widths = get_spike_prominence_and_width(denoised_signal, spike_events)    
    nb_epileptic_events = len(prominence_data[(prominence_data > signal_threshold) & (widths > min_width) & (widths < max_width)])

    return float(nb_epileptic_events) / len(prominence_data)


# In[116]:


#--- modified to take in numpy array ---##
def get_spike_prominence_and_width(denoised_signal, spike_events, spike_threshold=0.05, fps=30.0):
    """Get the local prominence and width of each spike in a calcium trace.

    Local prominence is a measure of the strength of the spike relative to other spikes nearest to it.

    Args:  
        denoised_signal:  The denoised fluorescence signal
        spike_events:  Discretized deconvolved neural activity (spikes)
        spike_threshold:  A float below which spikes will be thrown out
        fps:  Frames per second of the signal, used to convert width to seconds
    Returns:  
        A tuple, (prominence array, width array)
    """
    event_idxs = np.where(spike_events > spike_threshold)[0] 
    filtered_signal = denoised_signal[event_idxs] 

    _prominences = []
    _widths = []

    for idx, event_idx in enumerate(event_idxs):
        
        # Prominence calculation
        local_amplitude = denoised_signal[event_idx]
        local_diff = local_amplitude - filtered_signal
        larger_spike_idxs = event_idxs[local_diff < 0]

        pre_spike_idxs = larger_spike_idxs[larger_spike_idxs < event_idx]
        if not pre_spike_idxs.any():
            pre_spike_idx = 0
        else:
            pre_spike_idx = np.max(pre_spike_idxs) # Nearest spike larger than local spike on left
        
        # Minimum signal between prev large spike and current spike
        pre_spikes = denoised_signal[pre_spike_idx:event_idx]
        pre_min = np.min(pre_spikes)
        
        post_peak_idxs = larger_spike_idxs[larger_spike_idxs > event_idx]
        if not post_peak_idxs.any():
            post_peak_idx = len(denoised_signal)
        else:
            post_peak_idx = np.min(post_peak_idxs) # Nearest spike larger than local spike on right  
        
        # Minimum signal between post large spike and current spike
        post_spikes = denoised_signal[event_idx:post_peak_idx]
        post_min = np.min(post_spikes)

        local_min = np.max([pre_min, post_min]) # TODO: Should be min here?
        local_prominence = local_amplitude - local_min
        _prominences.append(local_prominence)
     
        # Width calculation
        pre_half_idxs = np.where(pre_spikes < local_amplitude - local_prominence / 2)[0]
        if not pre_half_idxs.any():
            pre_half_idx = len(pre_spikes)
        else: 
            pre_half_idx = np.min(len(pre_spikes) - pre_half_idxs)

        post_half_idxs = np.where(post_spikes < local_amplitude - local_prominence / 2)[0]
        if not post_half_idxs.any():
            post_half_idx = len(post_spikes)
        else: 
            post_half_idx = np.min(post_half_idxs)
        
        local_width = post_half_idx + pre_half_idx
        _widths.append(local_width)
         
    return np.array(_prominences), np.array(_widths) / fps


# In[173]:


#--- modified from code Marina sent ---#
def get_epilepsy_plot(prominence_data,widths):
    """
    """   
    fig, axs = plt.subplots(2, 1)
    from scipy import stats
    # Calculate the point density
    xy = np.vstack([widths, prominence_data])
    try:
        z = stats.gaussian_kde(xy)(xy)
    except LinAlgError:
        raise DataCorruptionError('Unable to form PDF for epilepsy - check video for anomalies.')

    plt.sca(axs[0])
    plt.scatter(widths, prominence_data, c=z, s=10, edgecolor='') 
    plt.xlabel('Event Width (s)')
    plt.xlim(0, 1)
    plt.ylabel('Event Size')

    plt.sca(axs[1])
    plt.plot(perc_diff, 'b-')
    plt.xlabel('Frame number')
    plt.ylabel('DFF')
    
    plt.tight_layout()
    return fig


# In[211]:


def deconv(data):
    from oasis.functions import deconvolve as oasis_deconvolve
    perc_diff = -100 * (1 - data / data.mean())
    denoised_signal, spike_events, _, _, _ = oasis_deconvolve(perc_diff, penalty=1)
    
    return denoised_signal, spike_events


# In[217]:


#--- get all experiment containers with epileptiform tags ---#
exps = boc.get_experiment_containers(include_failed=True)
idx = []
for ii,_ in enumerate(exps):
    if len(exps[ii]['tags']):
        idx.append(ii)
print(idx)


# In[223]:


exp_container_id = exps[18]['id']
session_id = boc.get_ophys_experiments(include_failed=True,experiment_container_ids=[exp_container_id],stimuli=['natural_scenes'])[0]['id']
print(session_id)


# In[216]:


#--- get frames from raw movie ---#
fid = session_id
stf = 40000 
mframes = get_ophys_movie_frames(fid,start_frame=stf,end_frame=stf+20000)
#--- get means in x and y direction ---#
mu_mframes = np.mean(mframes,axis=(1,2))
#--- deconvolve data ---#
denoised_signal, spike_events = deconv(mu_mframes)
#--- plot signals ---#
fig, (ax1,ax2,ax3) = plt.subplots(3)
ax1.plot(denoised_signal)
ax2.plot(spike_events)
ax3.plot(mu_mframes)


# In[221]:


prob = get_epilepsy_prob(mframes, signal_threshold=2.5, min_width=0.1, max_width=0.3)
print(prob)


# In[219]:


prominence_data, widths = get_spike_prominence_and_width(denoised_signal, spike_events, spike_threshold=0.05, fps=30.0)


# In[225]:


fig = get_epilepsy_plot(prominence_data,widths)
fig.savefig('interictal_info_example.pdf')


# In[228]:


plt.imshow(mframes[1],cmap='gray')

