
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
from omniperf_visualize.visualize_cli import visualize_cli
from omniperf_visualize.visualize_base import OmniVisualize_Base
from omniperf_visualize.dashing_setup import dashing_setup
from dash.exceptions import PreventUpdate
from openai import OpenAI
import os
import csv

def beautify_text(input_text):
	lines = input_text.split("\n")
	components = []
	for line in lines:
		print('Zayed new line')
		print(line)
		if line.startswith("### "): 
			components.append(html.H3(line[4:], style={"margin-top": "20px"}))
		elif line.startswith("1. "):
			components.append(html.Ol([
                html.Li(line[3:])
            ], style={"line-height": "1.6", "margin-left": "20px"}))
		elif line.startswith("- "):
			components[-1].children.append(html.Li(line[2:]))
		elif line.startswith('Optimization '):
			components.append(html.H2(line))
		else:
			components.append(html.P(line))
            
	return components

def dashboard_init(data_loaders, global_options):
	webpages = {}
	for app_name, data_loader in data_loaders.items():
		if data_loader.options['charts']:
			webpages[app_name] = create_page(data_loader)
#			for fig in data_loader.options['charts']:
#				fig.write_image("images/%s.pdf" % app_name)
	
	port = global_options['port'] if 'port' in global_options else 7050 

	# print(global_options['rerun'])
	if len(webpages.keys()) > 0:
		start_server(webpages, port)

# creates a div element that represents a whole webpage
# containing all visualizations for an app
def create_page(data_loader):
	print('Zayed Data loaded')
	# print(data_loader.get_events())
	charts = data_loader["charts"]
	title = data_loader.get_option('title', 'untitled chart')
	chart_elems = list(html.H1(children=title))
	options = ['Select a target metrics']
	options.extend(data_loader.get_events())
	# print(options)
	chart_elems.append(html.Div([dcc.Dropdown(options, 'Select a target metrics', id='metric-dropdown')]))
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
			html.Div([
			html.Div(dcc.Graph(id='chart', figure=chart), style={'width': '59%', 'display': 'inline-block'}),
			html.Div(html.P(instructions), style={'width': '40%', 'display': 'inline-block', 'float': 'right'})
		])]))

	client = OpenAI(
		api_key='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',
	)

	rsm_ev_errors = data_loader['rsm_ev_errors']
	rsm_alphas = data_loader['rsm_alphas']
	rsm_norm_data = data_loader['rsm_norm_data']
	rsm_results = data_loader['rsm_results']

	# print('rsm_ev_errors')
	# print(rsm_ev_errors)
	# print('rsm_alphas')
	# print(rsm_alphas)
	# print('rsm_norm_data')
	# print(rsm_norm_data)
	# print('rsm_results')
	# print(rsm_results)

	regions = data_loader.get_regions()
	for event, ev_err in rsm_ev_errors.items():
		# print(event)
		x = rsm_ev_errors[event]
		sort_x = {k: v for k, v in sorted(x.items(), key=lambda item: item[1])}
		# print(sort_x)

	descriptions = {}
	if os.path.exists(data_loader['desc_path']):
		with open(data_loader['desc_path'], 'r') as f:
			reader = csv.reader(f)
			descriptions = {}
			for rows in reader:
				val = ""
				key = rows[0]
				for r in range(1, len(rows)):
					val += rows[r]
				descriptions[key] = val
	
	# print(descriptions)

	for key in sort_x.keys():
		if key.find('OI_')!=-1:
			continue
		else:
			imp_ev = key
			break
	# imp_ev = list(sort_x.keys())[0]
	imp_ev_desc = descriptions[imp_ev]
	print('Zayed dottttttt')
	print(imp_ev,imp_ev_desc)

	query = 'How do you optimize ' + imp_ev_desc + ' amd gpu? '
	stream = client.chat.completions.create(
		model="gpt-4o-mini",
		messages=[{"role": "user", "content": query}],
		stream=True,
	)

	print('Zayyeddddd chatgpt reply')
	reply = 'Optimization insights for ' + event + 'kernel\n:'
	for chunk in stream:
		reply += chunk.choices[0].delta.content or ""
		# print(chunk.choices[0].delta.content or "", end="\n")
		# print('Zayed newline')

	chart_elems.append(html.Div(html.P(beautify_text(reply))))

	return html.Div(children=chart_elems)


def start_server(webpages, port):

	external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
	index_app = dash.Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)

	# vis_cli = visualize_cli()
	# nav bar
	nav_divs = []
	for page in webpages:
		nav_divs.append(html.Li(dcc.Link(page, href='/'+page)))
	nav = html.Ul(children = nav_divs, className="navbar")
	
	# home page
	index_divs = [dcc.Location(id='url', refresh=False)]
	index_divs.append(nav)
	index_divs.append(html.Div(id='page-content'))
	# print(index_divs)
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

	@index_app.callback(dash.dependencies.Output('chart', 'figure'),
				dash.dependencies.Input('metric-dropdown', 'value'))
	def update_output(value):
		if value == 'Select a target metrics' or value == '':
			raise PreventUpdate
		dashing_set = dashing_setup()
		# print('Running')
		# print(value)
		data_l = dashing_set.run_visualize('/home/mohammad/omni_inte/BabelStream',[value], str(port + 1) , 'False')
		raise PreventUpdate
		for _, data_loader in data_l.items():
			charts = data_loader["charts"]
			for chart in charts:
				return chart	

	app_thread = threading.Thread(target=index_app.run_server,
		kwargs={'debug':False, 'port':port, 'use_reloader':False})
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
		
	
