import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
from string import ascii_lowercase


def get_df(fname_list):
	col_names = ['t','x','v_x','y','v_y','e_j']
	df_list = [pd.read_csv(fname, delim_whitespace=True, names=col_names) for fname in fname_list]
	# print (df_list)
	for df,fname in zip(df_list,fname_list):
		df['time_len'] = fname
	df = pd.concat(df_list, ignore_index=False)
	df['time_len']=df['time_len'].astype('category')
	return df


def plot_trajectories(df,x2_values):
	sns.set()
	fname_list = ['x2_'+str(i) for i in x2_values]
	g = sns.FacetGrid(data=df, hue='time_len', col='time_len', col_order=fname_list, aspect=2, height=4.5)
	# g.map_dataframe(sns.lineplot)
	g.map(plt.plot,'x','y', linewidth=0.7)
	g.map(sns.scatterplot,'x','y', s=10)
	axes = g.axes.flatten()
	for i,axis in enumerate(axes):
		axis.set_title('('+str(ascii_lowercase[i])+') Final time = '+str(float(x2_values[i])))
	plt.savefig('./plots/q4.jpg',bbox_inches='tight',pad_inches=0.5,dpi=480)
	# plt.show()
	return


#------------------------------------------------------------------------------

x2_values = [5,20]
fname_list = ['x2_'+str(i) for i in x2_values]
# print (fname_list)
df = get_df(fname_list)
plot_trajectories(df,x2_values)