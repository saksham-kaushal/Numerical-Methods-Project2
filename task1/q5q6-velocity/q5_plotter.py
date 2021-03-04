import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns


def get_df(fname_list):
	col_names = ['t','x','v_x','y','v_y','e_j']
	df_list = [pd.read_csv(str(fname), delim_whitespace=True, names=col_names) for fname in fname_list]
	# print (df_list)
	for df,fname in zip(df_list,fname_list):
		df['vx_init'] = fname
	df = pd.concat(df_list, ignore_index=False)
	df['vx_init']=df['vx_init'].astype('category')
	return df

def plot_trajectories_sep(df,fname_list):
	sns.set()
	g = sns.FacetGrid(data=df, 
		hue='vx_init', 
		col='vx_init', 
		col_order=fname_list, 
		col_wrap=2,
		aspect=2,
		height=3)
	g.map(plt.plot,'x','y', linewidth=0.5)
	g.map(plt.scatter,'x','y', s=1.5)
	g.set_titles(col_template='$v_x = ${col_name}')
	# plt.show()
	plt.savefig('./plots/q6_sep.jpg',bbox_inches='tight',pad_inches=0.5,dpi=480)
	return

def plot_trajectories_combined(df, vx_values):
	sns.set()
	g = sns.FacetGrid(data=df, hue='vx_init', aspect=3.33, height=7)
	# g.map_dataframe(sns.lineplot)
	g.map(plt.plot,'x','y', linewidth=1.0)
	g.map(plt.scatter,'x','y', s=4)
	handles, labels = plt.gca().get_legend_handles_labels()
	plt.legend(title='$v_x$',
		title_fontsize='medium',
		fontsize='medium',
		markerscale=3,
		handles=handles[-len(vx_values):],
		loc=0,
		labels=vx_values)
	# plt.show()
	plt.savefig('./plots/q6_combined.jpg',bbox_inches='tight',pad_inches=0.5,dpi=480)
	return


# -----------------------------------------------------

# vx_values = [-1.64,-1.635,-1.63,-1.58,-1.15,-0.6,-0.524,-0.4] 		#q5q6 combined
# vx_values = [-1.68,-1.64,-1.635]							#q5 only
vx_values = [-1.63,-1.58,-1.15,-0.6,-0.524,-0.4] 			#q6 only
fname_list = vx_values
df = get_df(fname_list)
# print(df)
# fname_list = [-1.64,-1.635]
plot_trajectories_sep(df,fname_list)
plot_trajectories_combined(df,vx_values)