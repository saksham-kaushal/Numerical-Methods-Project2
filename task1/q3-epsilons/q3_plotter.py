import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns

def get_df(fname_list):
	col_names = ['t','x','v_x','y','v_y','e_j']
	df_list = [pd.read_csv(fname, delim_whitespace=True, names=col_names) for fname in fname_list]
	# print (df_list)
	for df,fname in zip(df_list,fname_list):
		df['epsilon'] = fname
	df = pd.concat(df_list, ignore_index=True)
	df['epsilon']=df['epsilon'].astype('category')
	return df

# def q3_plot(df):
# 	sns.set()
# 	palette = sns.cubehelix_palette(len(df['epsilon'].unique()),rot=0.5,dark=0.75,light=0.25)
# 	g=sns.relplot(data=df, x='t', y='e_j', hue='epsilon', s=5, linewidth=0, aspect=2, kind='scatter')
	
# 	# plt.legend()
# 	g._legend.set_title('$\epsilon$')
# 	plt.xlabel('Time', fontsize=9)
# 	plt.ylabel('Jacobi integral', fontsize=9)
# 	labels=['$10^{-5}$','$10^{-6}$','$10^{-7}$','$10^{-8}$','$10^{-9}$']
# 	for t, l in zip(g._legend.texts, labels): t.set_text(l)
# 	plt.savefig('./plots/q3.jpg',bbox_inches='tight',pad_inches=0.5,dpi=480)
# 	return

def q3_plot(df):
	sns.set()
	palette = sns.color_palette('magma',5,desat=0.9)
	g = sns.FacetGrid(data=df, hue='epsilon', aspect=2, height=4.5, palette=palette)
	g.map(plt.plot,'t','e_j', linewidth=0.4)
	g.map(plt.scatter,'t','e_j', s=3)
	plt.xlabel('Time', fontsize=9)
	plt.ylabel('Jacobi integral', fontsize=9)
	handles, labels = plt.gca().get_legend_handles_labels()
	plt.legend(title='Integration accuracy parameter, $\epsilon$',
		title_fontsize='small',
		fontsize='small',
		markerscale=2.5,
		handles=handles[-5:],
		labels=['$10^{-5}$','$10^{-6}$','$10^{-7}$','$10^{-8}$','$10^{-9}$'], 
		loc=0,
		ncol=2)
	plt.savefig('./plots/q3.jpg',bbox_inches='tight',pad_inches=0.5,dpi=480)
	# plt.show()
	return

def get_stats(df):
	return df.groupby('epsilon')['e_j'].std()

fname_list = ['eps_e-'+str(i) for i in range(5,10)]
# print (fname_list)
df = get_df(fname_list)
q3_plot(df)
# print(get_stats(df))

