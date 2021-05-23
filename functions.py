import pandas as pd
import numpy as np
import random
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt

import plotly.express as px 
import plotly.graph_objects as go

def generate_data(n_arm,n_obs,prob_arm):
	data = {}
	probs = []
	if prob_arm == "Random (similar across arms)":
		for s in range(n_arm):
		  current_label = 'slot_' + str(s)
		  data[current_label] = np.random.randint(2, size=n_obs)
	elif prob_arm == "Bias":
		for s in range(n_arm):
		  current_label = 'slot_' + str(s)
		  data[current_label] = np.random.binomial(1, (s+0.5)*0.1, size=n_obs)

	data = pd.DataFrame(data)
	
	for s in range(n_arm):
		current_label = 'slot_' + str(s)
		probs.append(data[current_label].sum()/n_obs)

	return data, probs

def single_step_sim(df):
	# SETUP
	N_SLOT = len(df.columns)
	N_OBS = len(df)
	slot_selected = []
	rewards = np.ones(N_SLOT,dtype=int).tolist() # alpha
	penalties = np.ones(N_SLOT,dtype=int).tolist() # beta
	total_reward = 0
	beta_params = np.empty((N_OBS,N_SLOT),dtype=object)

	for n in range(0, N_OBS): # iterate through observation
	  bandit = 0
	  beta_max = 0
	  beta_dist_snapshot = []
	  for i in range(0, N_SLOT): # iterate through slot / arm
	    beta_d = random.betavariate(rewards[i], penalties[i]) # draw random beta distribution for current slot, in the beginning alpha & beta equal 1
	    
	    if beta_d > beta_max:
	      beta_max = beta_d
	      bandit = i
	  # After draw random beta dist from each slot, we pick slot with the highest probability; 
	  # in the beginning the pick is random since all slots initiated with same alpha & beta

	  slot_selected.append(bandit)
	  reward = df.values[n, bandit] # Pull the chosen slot, and see whether it gives 0 or 1
	  # update alpha / beta for the chosen slot, based on the result; if 1 then increase alpha, if 0 then increase beta
	  if reward > 0:
	    rewards[bandit] = rewards[bandit] + 1
	  else:
	    penalties[bandit] = penalties[bandit] + 1
	  
	  total_reward = total_reward + reward
	  
	  # ab_list = [(rewards[i], penalties[i]) for i in range(0, len(rewards))] # save the updated parameter for beta distribution of each slot
	  # beta_params[n] = ab_list
	  for bpi in range(0, len(rewards)):
	  	beta_params[n][bpi] = (rewards[bpi], penalties[bpi])

	  # At the end of the first iteration of observation, 1 slot will have different beta distribution since the alpha & beta have been updated
	beta_params = pd.DataFrame(beta_params, columns = df.columns)
	beta_params['obs_idx'] = np.arange(0,N_OBS,1,dtype=int).tolist()
	return slot_selected, rewards, penalties, total_reward, beta_params

def get_df_distribution(beta_params):
	N_SNAPSHOT = 10
	N_ARM = len(beta_params.columns)-1
	N_OBS = len(beta_params)
	snapshot = np.arange(0,len(beta_params),len(beta_params)/N_SNAPSHOT,dtype=int).tolist()
	snapshot.append(len(beta_params)-1)

	snapshot_param = beta_params.iloc[snapshot,:]
	slot_cols = [c for c in snapshot_param.columns if 'slot_' in c] 
	# print(slot_cols)
	x = np.linspace(0.0, 1.0, N_OBS)
	beta = stats.beta
	snapshot_dist_df = pd.DataFrame()

	for idx, row in snapshot_param.iterrows():
	  # print(idx)
	  current_snapshot = pd.DataFrame()
	  # current_mean = pd.DataFrame()
	  for sc in slot_cols:
	    current_y_label = 'y_' + sc
	    a, b = row[sc]
	    current_snapshot[current_y_label] = beta.pdf(x, a, b)
	    # current_mean_label = 'mean_' + sc
	    # if a > 1:
		   #  current_mean[current_mean_label] = a/(a+b)
	    # else:
	    #   current_mean[current_mean_label] = 0
	    # current_mean_param_label = 'param_' + sc
	    # current_mean[current_mean_param_label] = tuple(par for par in [a,b])
	  current_snapshot['x'] = x
	  current_snapshot['iteration'] = idx
	  # current_mean['iteration'] = idx
	  if len(snapshot_dist_df) == 0:
	    snapshot_dist_df = current_snapshot
	    # snapshot_mean_df = current_mean
	  else:
	    # print('append')
	    snapshot_dist_df = snapshot_dist_df.append(current_snapshot, ignore_index=True)
	    # snapshot_mean_df = snapshot_mean_df.append(current_mean, ignore_index=True)
	unpivot_snapshot = pd.melt(snapshot_dist_df, id_vars=['x','iteration'], value_vars=[y for y in snapshot_dist_df.columns if 'y_' in y], var_name='y_group', value_name='y_values')
	return snapshot_param, unpivot_snapshot


# def plot_arm_dist_plotly(beta_params):
# 	N_OBS = len(beta_params.columns)
# 	beta = stats.beta
# 	x = np.linspace(0.0, 1.0, N_OBS*100)
# 	val = N_OBS - 1
# 	params = beta_params[val]
# 	c_index = 0
# 	colors = ["red","blue","green","orange","purple","pink","grey"]
	
# 	fig = go.Figure()

# 	return(fig)

def plot_arm_dist(beta_params):
	N_OBS = len(beta_params)
	N_SLOT = len(beta_params.columns) - 1
	beta = stats.beta

	x = np.linspace(0.0, 1.0, N_OBS)

	# plt.figure(figsize=(12,7))

	## plot final arm distribution
	fig, axs = plt.subplots(nrows=1,figsize=(12,7))
	val = N_OBS - 1
	params = beta_params.iloc[val,:N_SLOT].tolist()
	c_index = 0
	colors = ["red","blue","green","orange","purple","pink","grey"]
	for a, b in params:
	    y = beta.pdf(x, a, b)
	    c = colors[c_index]
	    axs.plot(x,y,label = f"({a},{b})",lw = 3, color = c)
	    axs.fill_between(x, 0, y, alpha = 0.2, color = c)
	      
	    if a > 1:
	        mean = a/(a+b)
	        axs.vlines(mean, 0, beta.pdf(mean, a, b), colors = c, linestyles = "--", lw = 2)    
	      
	    axs.autoscale(tight=True)
	    axs.set_title("Iteration {}".format(val))
	    axs.legend(loc = 'upper left', title="(a,b) parameters")
	    c_index += 1
	
	# fig, axs = plt.subplots(nrows=len(param_check_index),figsize=(12,7*len(param_check_index)))
	# colors = ["red","blue","green","orange"]

	# for idx,val in enumerate(param_check_index):
	#   params = beta_params_snapshot[val]
	#   c_index = 0
	#   for α, β in params:
	#       y = beta.pdf(x, α, β)
	#       c = colors[c_index]
	#       axs[idx].plot(x,y,label = f"({α},{β})",lw = 3, color = c)
	#       axs[idx].fill_between(x, 0, y, alpha = 0.2, color = c)
	      
	#       if α > 1:
	#           mean = α/(α+β)
	#           axs[idx].vlines(mean, 0, beta.pdf(mean, α, β), colors = c, linestyles = "--", lw = 2)    
	      
	#       axs[idx].autoscale(tight=True)
	#       axs[idx].set_title("Iteration {}".format(val))
	#       axs[idx].legend(loc = 'upper left', title="(α,β) parameters")
	#       c_index += 1
	      
	# plt.show()
	return fig, axs
