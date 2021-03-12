import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
from string import ascii_lowercase


def get_df(dir_list,vx_values):
	col_names = ['nu','prw','log_prw']
	df_list = [pd.read_csv(str(subdir+'/fort.25'), delim_whitespace=True, names=col_names) for subdir in dir_list]
	# print (df_list)
	for df,vx in zip(df_list,vx_values):
		df['vx_init'] = vx
	df = pd.concat(df_list, ignore_index=False)
	df['vx_init']=df['vx_init'].astype('category')
	df['nu'] = df['nu'].apply(lambda a: 2*np.pi*(a-1)/(8192*0.02))
	return df

def plot_ospsd_log(df,vx_values):
	sns.set()
	g = sns.FacetGrid(data=df, 
		hue='vx_init', 
		col='vx_init', 
		col_order=vx_values, 
		col_wrap=2,
		aspect=2,
		height=3)
	g.map(sns.lineplot,'nu','log_prw', linewidth=0.5)
	g.map(sns.scatterplot,'nu','log_prw', s=3)
	g.set_xlabels('Frequency, $\\nu$')
	g.set_ylabels('$log(PSD)$')
	# g.set_titles(col_template='$v_x = ${col_name}') 		# add plot index
	axes=g.axes.flatten()
	for i,vx_value in enumerate(vx_values):
		axes[i].set_title('('+ascii_lowercase[i]+') $v_x =$ '+str(vx_value))
	# plt.show()
	plt.savefig('./plots/q1-5_log.jpg',bbox_inches='tight',pad_inches=0.5,dpi=480)
	return

def plot_ospsd_lin(df,vx_values):
	sns.set()
	g = sns.FacetGrid(data=df, 
		hue='vx_init', 
		col='vx_init', 
		col_order=vx_values, 
		col_wrap=2,
		aspect=2,
		height=3)
	g.map(sns.lineplot,'nu','prw', linewidth=0.5)
	g.map(sns.scatterplot,'nu','prw', s=3)
	g.set_xlabels('Frequency, $\\nu$')
	g.set_ylabels('PSD')
	# g.set_titles(col_template='$v_x = ${col_name}') 		# add plot index
	axes=g.axes.flatten()
	for i,vx_value in enumerate(vx_values):
		axes[i].set_title('('+ascii_lowercase[i]+') $v_x =$ '+str(vx_value))
	# plt.show()
	plt.savefig('./plots/q1-5_lin.jpg',bbox_inches='tight',pad_inches=0.5,dpi=480)
	return


# ------------------------------------------------------------

vx_values = [-1.635,-1.64,-1.68,-1.15]
dir_list = ['./q1q2','./q3','./q4','./q5']
df = get_df(dir_list,vx_values)
plot_ospsd_log(df,vx_values)
plot_ospsd_lin(df,vx_values)