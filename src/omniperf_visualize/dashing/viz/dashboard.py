
import numpy as np
import plotly.offline as py
import dash
from dash import dcc
from dash import html
import webbrowser
import threading
import time
import plotly.io as pio
from plotly.graph_objects import Figure
import plotly.graph_objects as go

def dashboard_init(data_loaders, global_options):
	webpages = {}
	for app_name, data_loader in data_loaders.items():
		if data_loader.options['charts']:
			webpages[app_name] = create_page(data_loader)
#			for fig in data_loader.options['charts']:
#				fig.write_image("images/%s.pdf" % app_name)
	
	port = global_options['port'] if 'port' in global_options else 7050 

	if len(webpages.keys()) > 0:
		start_server(webpages, port)

# creates a div element that represents a whole webpage
# containing all visualizations for an app
def create_page(data_loader):

	charts = data_loader["charts"]
	title = data_loader.get_option('title', 'untitled chart')
	chart_elems = list(html.H1(children=title))
	instructions = ['This tool provides Dashing\'s important analysis for the tuning parameters of the tuning problem.',
	html.Br(),
	html.B('What is Importance analysis?'),
	html.Br(),
	'Importance analysis reveals the importance of a resource against an objective function. A resource is given more importance if the resource function closely matches the trends of the target function.',
	html.Br(),
	html.B('What is Dashing?'),
	html.Br(),
	'Dashing implements several visualization tools for the interactive exploration of performance data with machine learning-based performance analysis techniques.',
	html.Br(),
	html.B('How to read the importance analysis?'),
	html.Br(),
	'Sunburst charts: There are four layers in the following sunburst chart. At each layer, the more important an element is the more area it gets.',
	html.Ul([html.Li('The first and central layer shows the name of the objective function.'),
		  html.Li('The second layer lists the importance of different phases during an application run.'),
		  html.Li('The third layer lists the importance of the resource group.'),
		  html.Li('The final and outer layer shows the importance of individual resources.')]),
	html.Br(),
	'If a sunburst chart is missing it means that the analysis found no resource important enough to show. Click on a sunburst region to expand it further.']
	# chart_elems.append(html.Div(html.P(instructions)))
	for chart in charts:
		chart_elems.append(html.Div([
			html.Div(dcc.Graph(id=str(chart), figure=chart), style={'width': '59%', 'display': 'inline-block'}),
			html.Div(html.P(instructions), style={'width': '40%', 'display': 'inline-block', 'float': 'right'})
		]))

		# chart_elems.append(html.Div(dcc.Graph(id=str(chart), figure=chart, style={float:'left'})))
		# chart_elems.append(html.Div(html.P(instructions),style={float:'right'}))


		# chart.write_html("/home/mohammad/gptune-data/" + str(charts.index(chart)) + '.html')
		# chart.write_json("/home/mohammad/gptune-data/" + str(charts.index(chart)) + '.json')
		# chart2 = {}
		# with open('/home/mohammad/gptune-data/0.json', 'r') as f:
		# 	chart2 = go.Figure(pio.from_json(f.read()))
		# 	print("Zayed")
		# 	print(type(chart2))
		# plot = py(chart2,output_type="Figure")
		# chart_elems.append(html.Div(dcc.Graph(id=str(chart2), figure=chart2)))
	return html.Div(children=chart_elems)


def start_server(webpages, port):

	external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
	index_app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

	# nav bar
	nav_divs = []
	for page in webpages:
		nav_divs.append(html.Li(dcc.Link(page, href='/'+page)))
	nav = html.Ul(children = nav_divs, className="navbar")
	
	# home page
	index_divs = [dcc.Location(id='url', refresh=False)]
	index_divs.append(nav)
	index_divs.append(html.Div(id='page-content'))
	print(index_divs)
	index_app.layout = html.Div(children=index_divs)
	

	# mimics having different webpages depending on the URL
	@index_app.callback(dash.dependencies.Output('page-content', 'children'),
	[dash.dependencies.Input('url', 'pathname')])
	def display_page(pathname):
		if pathname is None: return
		key = pathname[1:]	
		if key in webpages:
			return webpages[key]
		else:
			return



	app_thread = threading.Thread(target=index_app.run_server,
		kwargs={'debug':True, 'port':port, 'use_reloader':False})
	#index_app.run_server(debug=True, port=port, use_reloader=False)
	app_thread.start()
	webbrowser.open(f'http://127.0.0.1:{port}')




#shelved

#	#search for events/resources/regions, highlight those divs?
#	@index_app.callback(
#		dash.dependencies.Output('placeholder', 'children'),
#		[dash.dependencies.Input('searchbar', 'value')])
#	def search(value):
#		print("searching for: ", value)
#		pagecontent = index_divs[2]
#		print(pagecontent)
		
	
