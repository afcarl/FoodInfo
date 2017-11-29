import seaborn as sns
import plotly.plotly as py
import plotly.graph_objs as go
import plotly.offline as offline
from sklearn.manifold import TSNE
import time
import itertools
import numpy as np
from sklearn.cluster import KMeans
"""
Plot Points with Labels
"""
def make_plot_only_labels(name, points, labels, publish):
	traces = []
	traces.append(go.Scattergl(
			x = points[:, 0],
			y = points[:, 1],
			mode = 'markers',
			marker = dict(
				color = sns.xkcd_rgb["black"],
				size = 8,
				opacity = 0.6,
				#line = dict(width = 1)
			),
			text = labels,
			hoverinfo = 'text',
		)
		)
				  
	layout = go.Layout(
		xaxis=dict(
			autorange=True,
			showgrid=False,
			zeroline=False,
			showline=False,
			autotick=True,
			ticks='',
			showticklabels=False
		),
		yaxis=dict(
			autorange=True,
			showgrid=False,
			zeroline=False,
			showline=False,
			autotick=True,
			ticks='',
			showticklabels=False
		)
		)
				  
	fig = go.Figure(data=traces, layout=layout)
	if publish:
		plotter = py.iplot
	else:
		plotter = offline.plot
	plotter(fig, filename=name + '.html')

"""
Plot Points with Labels and Legends
"""

def make_plot_with_labels_legends(name, points, labels, legend_labels, legend_order, legend_label_to_color, pretty_legend_label, publish):
	lst = zip(points, labels, legend_labels)
	full = sorted(lst, key=lambda x: x[2])
	traces = []
	for legend_label, group in itertools.groupby(full, lambda x: x[2]):
		group_points = []
		group_labels = []
		for tup in group:
			point, label, _ = tup
			group_points.append(point)
			group_labels.append(label)
		group_points = np.stack(group_points)
		traces.append(go.Scattergl(
			x = group_points[:, 0],
			y = group_points[:, 1],
			mode = 'markers',
			marker = dict(
				color = legend_label_to_color[legend_label],
				size = 8,
				opacity = 0.6,
				#line = dict(width = 1)
			),
			text = ['{} ({})'.format(label, pretty_legend_label(legend_label)) for label in group_labels],
			hoverinfo = 'text',
			name = legend_label
		)
		)
	# order the legend
	ordered = [[trace for trace in traces if trace.name == lab] for lab in legend_order]
	traces_ordered = flatten(ordered)
	def _set_name(trace):
		trace.name = pretty_legend_label(trace.name)
		return trace
	traces_ordered = list(map(_set_name, traces_ordered))
	
	"""
	annotations = []
	for index in range(50):
		new_dict = dict(
				x=points[:, 0][index],
				y=points[:, 1][index],
				xref='x',
				yref='y',
				text=labels[index],
				showarrow=True,
				arrowhead=7,
				ax=0,
				ay=-10
			)
		annotations.append(new_dict)
	"""
	
	layout = go.Layout(
		xaxis=dict(
			autorange=True,
			showgrid=False,
			zeroline=True,
			showline=True,
			autotick=True,
			ticks='',
			showticklabels=False
		),
		yaxis=dict(
			autorange=True,
			showgrid=False,
			zeroline=True,
			showline=True,
			autotick=True,
			ticks='',
			showticklabels=False
		),
		#annotations=annotations
	)
	fig = go.Figure(data=traces_ordered, layout=layout)
	if publish:
		plotter = py.iplot
	else:
		plotter = offline.plot
	plotter(fig, filename=name + '.html')

"""
TSNE

"""
def load_TSNE(data, dim=2):
	print "\nt-SNE Started... "
	time_start = time.time()

	X = []
	for x in data:
		X.append(x)
	tsne = TSNE(n_components=dim)
	X_tsne = tsne.fit_transform(X)

	print "t-SNE done!"
	print "Time elapsed: {} seconds".format(time.time()-time_start)

	return X_tsne

if __name__ == '__main__':
    data = np.load("npy/composer_weight.npy")
    label_list = np.load("npy/id2comp.npy")
    X = []
    labels = []
    
    for i, vector in enumerate(data):
        labels.append(label_list[i])
        X.append(vector)

    X_tsne = load_TSNE(X)

    withLegends = False

    flatten = lambda l: [item for sublist in l for item in sublist]
    # Prettify ingredients
    pretty_food = lambda s: ' '.join(s.split('_')).capitalize().lstrip()
    pretty_category = lambda s: ''.join(map(lambda x: x if x.islower() else " "+x, s)).lstrip()

    #Legend Load
    if withLegends:
        categories_color = list(set(categories))
        # generate category colors
        #colors = sns.color_palette("deep", len(categories_color)).as_hex()
        colors = sns.color_palette("Set1", n_colors=len(categories_color), desat=.5).as_hex()
        category2color = {}
        count = 0
        for category in categories_color:
            category2color[category] = colors[count]
            count += 1

        # category_order
        category_order = categories_color

        # plot
        make_plot_with_labels_legends(name='results_with_legends',
            points=X_tsne, 
            labels=labels, 
            legend_labels=categories, 
            legend_order=category_order, 
            legend_label_to_color=category2color, 
            pretty_legend_label=pretty_category,
            publish=False)

    else:
        make_plot_only_labels(name='results_without_legends',
            points=X_tsne, 
            labels=labels, 
            publish=False)
