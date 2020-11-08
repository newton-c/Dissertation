import dash
import dash_html_components as html
import dash_core_components as dcc 
from dash.dependencies import Input, Output

import plotly.graph_objects as go
from igraph import Graph, EdgeSeq

app = dash.Dash()

tab_style1 = {
    'height': '5px',
    'width': '325px',
    'fontSize': 18
}

tab_style2 = {
    'height': '44px',
    'marginLeft': '355px'
}

# Creating the forest graph for the appendix
nr_vertices = 25
v_label = list(map(str, range(nr_vertices)))
G = Graph.Tree(nr_vertices, 2) # 2 stands for children number
lay = G.layout('rt')

position = {k: lay[k] for k in range(nr_vertices)}
Y = [lay[k][1] for k in range(nr_vertices)]
M = max(Y)

es = EdgeSeq(G) # sequence of edges
E = [e.tuple for e in G.es] # list of edges

L = len(position)
Xn = [position[k][0] for k in range(L)]
Yn = [2*M-position[k][1] for k in range(L)]
Xe = []
Ye = []
for edge in E:
    Xe+=[position[edge[0]][0],position[edge[1]][0], None]
    Ye+=[2*M-position[edge[0]][1],2*M-position[edge[1]][1], None]

labels = v_label

fig = go.Figure()
fig.add_trace(go.Scatter(x=Xe,
                   y=Ye,
                   mode='lines',
                   line=dict(color='rgb(210,210,210)', width=1),
                   hoverinfo='none'
                   ))
fig.add_trace(go.Scatter(x=Xn,
                  y=Yn,
                  mode='markers',
                  name='bla',
                  marker=dict(symbol='circle-dot',
                                size=18,
                                color='#6175c1',    #'#DB4551',
                                line=dict(color='rgb(50,50,50)', width=1)
                                ),
                  text=labels,
                  hoverinfo='text',
                  opacity=0.8
                  ))

def make_annotations(pos, text, font_size=10, font_color='rgb(250,250,250)'):
    L=len(pos)
    if len(text)!=L:
        raise ValueError('The lists pos and text must have the same len')
    annotations = []
    for k in range(L):
        annotations.append(
            dict(
                text=labels[k], # or replace labels with a different list for the text within the circle
                x=pos[k][0], y=2*M-position[k][1],
                xref='x1', yref='y1',
                font=dict(color=font_color, size=font_size),
                showarrow=False)
        )
    return annotations

axis = dict(showline=False, # hide axis line, grid, ticklabels and  title
            zeroline=False,
            showgrid=False,
            showticklabels=False,
            )

fig.update_layout(title= 'Tree with Reingold-Tilford Layout',
              annotations=make_annotations(position, v_label),
              font_size=12,
              showlegend=False,
              xaxis=axis,
              yaxis=axis,
              margin=dict(l=40, r=40, b=85, t=100),
              hovermode='closest',
              plot_bgcolor='rgb(248,248,248)'
              )

app.layout = html.Div([
    dcc.Tabs(id='tabs-example', value='tab-1', children=[
        dcc.Tab(label='Title', value='tab-1', 
            style = tab_style1, selected_style=tab_style1),
        dcc.Tab(label='Criminal Violence & Negative Peace', value='tab-2', 
            style = tab_style1, selected_style=tab_style1),
        dcc.Tab(label='M&M: Rwanda', value='tab-3', 
            style = tab_style1, selected_style=tab_style1),
        dcc.Tab(label='M&M: Effect of Media', value='tab-4', 
            style = tab_style1, selected_style=tab_style1),
        dcc.Tab(label='M&M: Reintegration', value='tab-5', 
            style = tab_style1, selected_style=tab_style1),
        dcc.Tab(label='M&M: Theory', value='tab-6', 
            style = tab_style1, selected_style=tab_style1),
        dcc.Tab(label='M&M: Colombia', value='tab-7', 
            style = tab_style1, selected_style=tab_style1),
        dcc.Tab(label='M&M: Sentiment Analysis', value='tab-8', 
            style = tab_style1, selected_style=tab_style1),
        dcc.Tab(label='M&M: Research Design', value='tab-9', 
            style = tab_style1, selected_style=tab_style1),
        dcc.Tab(label='M&M: Initial Results', value='tab-10', 
            style = tab_style1, selected_style=tab_style1),
        dcc.Tab(label='M&M: Next Steps', value='tab-11', 
            style = tab_style1, selected_style=tab_style1),
        dcc.Tab(label='RR&H: Organizational Explanations', 
            value='tab-21', style = tab_style1, selected_style=tab_style1),
        dcc.Tab(label='RR&H: Post Conflict', 
            value='tab-22', style = tab_style1, selected_style=tab_style1),
        dcc.Tab(label='RR&H: Research Design', 
            value='tab-23', style = tab_style1, selected_style=tab_style1),
        dcc.Tab(label='RR&H: Next Steps', 
            value='tab-24', style = tab_style1, selected_style=tab_style1),
        dcc.Tab(
            label='P&TV: Social Disorganization', 
            value='tab-31', style = tab_style1, selected_style=tab_style1),
        dcc.Tab(
            label='P&TV: Political Economy of Crime', 
            value='tab-32', style = tab_style1, selected_style=tab_style1),
        dcc.Tab(
            label='P&TV: Organizational Theory', 
            value='tab-33', style = tab_style1, selected_style=tab_style1),
        dcc.Tab(label='P&TV: Maching Learning', 
            value='tab-34', style = tab_style1, selected_style=tab_style1),
        dcc.Tab(label='P&TV: Research Design', 
            value='tab-35', style = tab_style1, selected_style=tab_style1),
        dcc.Tab(label='P&TV: Initial Results', 
            value='tab-36', style = tab_style1, selected_style=tab_style1),
        dcc.Tab(label='P&TV: Next Steps', 
            value='tab-37', style = tab_style1, selected_style=tab_style1),
        dcc.Tab(id='tab-99', label='Appendicies', value='tab-99', 
            style = tab_style1, selected_style=tab_style1, children=[]),
    ],
    style = tab_style1, vertical = True),
    html.Div(id='tabs-example-content')
])

@app.callback(Output('tabs-example-content', 'children'),
                [Input('tabs-example', 'value')])

def render_content(tab):
    if tab == 'tab-1': #Title
        return html.Div([
            html.H1('The War is Over but the Killing Remains',
            style={'textAlign': 'center',
            'marginTop': 75}
            ),
            html.H2('Criminal Violence in Postconflict States',
            style={'textAlign': 'center'}
            ),
            html.H3(),
            html.H3('Christopher Newton',
            style={'textAlign': 'center',
            'marginTop': 50,
            'marginBottom': 50}
            ),
            html.H3('Committee Chair: Professor Jacob Kathman',
            style={'textAlign': 'center'}
            ),
            html.H3('Committee Chair: Professor Michelle A. Benson',
            style={'textAlign': 'center'}
            ),
            html.H3('Committee Chair: Professor Jacob Neiheisel',
            style={'textAlign': 'center'}
            ),
            html.Img(src=app.get_asset_url('UB_Horizontal.png'),
            style={'position': 'fixed',
            'bottom': 0,
            'left': 700,
            'height': 70,
            'width': 450
            })
        ])

    elif tab == 'tab-2': # Criminal violence and negative peace
        return html.Div([
            html.H2('Criminal Violence & Negative Peace',
            style={'textAlign': 'center'}),
            dcc.Markdown('''
            * Negative versus Positive Peace

            * Criminal versus Political Violence
            ''',
            style={
                'marginTop': 100,
                'marginLeft': 375,
                'fontSize': 24}),
            html.Img(src=app.get_asset_url('UB_Horizontal.png'),
            style={'position': 'fixed',
            'bottom': 0,
            'left': 700,
            'height': 70,
            'width': 450
            })
        ])
        
    elif tab == 'tab-3': # Media & Murder: Rwanda
        return html.Div([ 
            html.H2('Media & Murder',
            style={'textAlign': 'center'}),
            dcc.Markdown('''
            * _Kangura_

            * Radio Télévision Libre des Mille Collines (RTLM)
            ''',
            style={
                'marginTop': 100,
                'marginLeft': 375,
                'fontSize': 24}),
            html.Img(src=app.get_asset_url('UB_Horizontal.png'),
            style={'position': 'fixed',
            'bottom': 0,
            'left': 700,
            'height': 70,
            'width': 450
            })
        ])

    elif tab == 'tab-4': # Media & Murder: Effect of media
        return html.Div([
            html.H2('Media & Murder',
            style={'textAlign': 'center'}),
            dcc.Markdown('''
            * Media matters:
                * Elections
                * Trust in institutions
                * Violence

            * Reintegration 
            ''',
            style={
                'marginTop': 100,
                'marginLeft': 375,
                'fontSize': 24}),
            html.Img(src=app.get_asset_url('UB_Horizontal.png'),
            style={'position': 'fixed',
            'bottom': 0,
            'left': 700,
            'height': 70,
            'width': 450
            })
        ])

    elif tab == 'tab-5': # Media & Murder: Reintegration
        return html.Div([
            html.H2('Media & Murder',
            style={'textAlign': 'center'}),
            dcc.Markdown('''
            * Individual characteristics:
                * Unit atrocities
                * Acceptace by family & community
                * Forming of connections outside of unit

            * Broader programs have limited effects
            ''',
            style={
                'marginTop': 100,
                'marginLeft': 375,
                'fontSize': 24
            }),
            html.Img(src=app.get_asset_url('UB_Horizontal.png'),
            style={'position': 'fixed',
            'bottom': 0,
            'left': 700,
            'height': 70,
            'width': 450
            })
        ])

    elif tab == 'tab-6': # Media & Murder: Theory
        return html.Div([
            html.H2('Media & Murder',
            style={'textAlign': 'center'}),
            dcc.Markdown('''
            * Negative media coverage with cause animus towards former rebels

            * Animus will prevent successful reintegration
                * Lack of oppotrunities within society will lead to criminality
                * Animosity against frmer rebels can lead to violnce against them
            
            _Hypothesis 1: Increased negative media of former rebels will lead to an increase in criminal violence._
            ''',
            style={
                'marginTop': 100,
                'marginLeft': 375,
                'fontSize': 24
            }),
            html.Img(src=app.get_asset_url('UB_Horizontal.png'),
            style={'position': 'fixed',
            'bottom': 0,
            'left': 700,
            'height': 70,
            'width': 450
            })
        ])

    elif tab == 'tab-7': # Media & Murder: Colombia
        return html.Div([
            html.H2('Media & Murder',
            style={'textAlign': 'center'}),
            dcc.Markdown('''
            * Highly developed

            * Recent end to FARC conflict

            * Spanish speaking
            ''', 
            style={
                'marginTop': 100,
                'marginLeft': 375,
                'fontSize': 24
            }),
            html.Img(src=app.get_asset_url('UB_Horizontal.png'),
            style={'position': 'fixed',
            'bottom': 0,
            'left': 700,
            'height': 70,
            'width': 450
            })
        ])

    elif tab == 'tab-8': # Media & Murder: Sentiment Analysis 
        return html.Div([
            html.H2('Media & Murder',
            style={'textAlign': 'center'}),
            dcc.Markdown('''
            Sentiment anaysis:
            * pretrained model
            * 0-1
            ''',
            style={
                'marginTop': 0,
                'marginLeft': 375,
                'fontSize': 24}),
            html.P('The insurgents attacked the village, killing more that 1,000 innocent people. = 0.06',
            style={
                'marginTop': 50,
                'marginLeft': 375,
                'fontSize': 24}),
            html.P('(Los guerrilleros atacaron al pueblo, matando a mas de un mil personas inocentes.)',
            style={
                'marginTop': 0,
                'marginLeft': 375,
                'fontSize': 24}),
            html.P('The heroes helped the village rebuild their houses. = 0.84',
            style={
                'marginTop': 50,
                'marginLeft': 375,
                'fontSize': 24}),
		    html.P('(Los héroes ayudaron al pueblo reconstuir las casas.)',
            style={
                'marginTop': 0,
                'marginLeft': 375,
                'fontSize': 24}),
            html.P('The people did a thing in a place. = 0.59',
            style={
                'marginTop': 50,
                'marginLeft': 375,
                'fontSize': 24}),
            html.P('(Las personas hicieron una cosa en un lugar.)',
            style={
                'marginTop': 0,
                'marginLeft': 375,
                'fontSize': 24}),
            html.Img(src=app.get_asset_url('UB_Horizontal.png'),
            style={'position': 'fixed',
            'bottom': 0,
            'left': 700,
            'height': 70,
            'width': 450
            })
        ])

    elif tab == 'tab-9': # Media & Murder: Research Design
        return html.Div([
            html.H2('Media & Murder',
            style={'textAlign': 'center'}),
            dcc.Markdown('''
           * DV: Homicide Count

           * IV: Sentiment of Media
                * Continuous (0-1) 
                * Scraped from Colombian newspaper
                * One week lag

            * Model: OLS

            * Controls: Unemployment, BRDs, OSV, DDR, Coca growth

            * Unit of analysis: Department-week

            * Endogeneity
                * Granger causality test
            ''',
            style={
                'marginTop': 100,
                'marginLeft': 350,
                'fontSize': 24}),
            html.Img(src=app.get_asset_url('UB_Horizontal.png'),
            style={'position': 'fixed',
            'bottom': 0,
            'left': 700,
            'height': 70,
            'width': 450
            })
        ])

    elif tab == 'tab-10': # Media & Murder: Initial Results
        return html.Div([
            html.H2('Media & Murder',
            style={'textAlign': 'center'}),
            html.Img(src=app.get_asset_url('media&murder_ir.png'),
            style={'marginLeft': 475}),
            html.Img(src=app.get_asset_url('UB_Horizontal.png'),
            style={'position': 'fixed',
            'bottom': 0,
            'left': 700,
            'height': 70,
            'width': 450
            })
        ])

    elif tab == 'tab-11': # Media & Murder: Next Steps 
        return html.Div([
            html.H2('Media & Murder',
            style={'textAlign': 'center'}),
            dcc.Markdown('''
            * End of Deceber: Finish scraping and cleaning text data
            * End of December: Conduct sentiment analysis
            * January: Merging all data
            * February: Statistical analysis
            * March and April: Writing and editing
            ''',
            style={
                'marginTop': 100,
                'marginLeft': 350,
                'fontSize': 24}),
            html.Img(src=app.get_asset_url('UB_Horizontal.png'),
            style={'position': 'fixed',
            'bottom': 0,
            'left': 700,
            'height': 70,
            'width': 450
            })
        ])

    elif tab == 'tab-21': # Rebels, Resources, & Homicide: Organizational Explanations 
        return html.Div([
            html.H2('Rebels, Resrouces, & Homicide',
            style={'textAlign': 'center'}),
            dcc.Markdown('''
            * Organizational structure, resource access, and resruitment strategies effect rebels behavior

            * Two types of recruits
                * Activists
                * Opportunists

            * Resource-rich groups will pay recruits to moblize a groups that can threaten the state
                * Attracts opportunists
                * Hard to sanction bad behavior
                * Recruits likely indifferent to long-term goals of group
                * Civilian victimization high
            
            * Resource-poor groups will mobalize based on ideoligical or ethnig lines
                * Attracts activists
                * Recruits vetted through social and communal networks
                * Long-term goals of recruits and group similar
                * Civilian victimization low

            ''',
            style={
                'marginTop': 100,
                'marginLeft': 350,
                'fontSize': 24}),
            html.Img(src=app.get_asset_url('UB_Horizontal.png'),
            style={'position': 'fixed',
            'bottom': 0,
            'left': 700,
            'height': 70,
            'width': 450
            })
        ])

    elif tab == 'tab-22': # Rebels, Resources, & Homicide: Post Conflict 
        return html.Div([
            html.H2('Rebels, Resrouces, & Homicide',
            style={'textAlign': 'center'}),
            dcc.Markdown('''
            * Opportunists learned how to use violence to extract resources and exploit population
                * When conflict ends, use skills for criminal profit

            * Activists only use violence for political reasons
                * When conflict ends, accept outcome or remobilize

            _Hypothesis 1:  The more a rebel group relies on profits from 
            contraband during a conflict, the higher postconflict homicide rates will be._
            ''',
            style={
                'marginTop': 100,
                'marginLeft': 350,
                'fontSize': 24}),
            html.Img(src=app.get_asset_url('UB_Horizontal.png'),
            style={'position': 'fixed',
            'bottom': 0,
            'left': 700,
            'height': 70,
            'width': 450
            })
        ])

    elif tab == 'tab-23': # Rebels, Resources, & Homicide: Research Design 
        return html.Div([
            html.H2('Rebels, Resrouces, & Homicide',
            style={'textAlign': 'center'}),
            dcc.Markdown('''
            * DV: Homicide rate

            * IV: Years with illicit funding

            * Model: OLS

            * Controls: Unemployment, GDP per capita, Polity5, conflict duration, 
            BRDs, OSV, DDR, conflict termination, peacekeeping

            * Unit of analysis: Dyad-year

            ''',
            style={
                'marginTop': 100,
                'marginLeft': 350,
                'fontSize': 24}),
            html.Img(src=app.get_asset_url('UB_Horizontal.png'),
            style={'position': 'fixed',
            'bottom': 0,
            'left': 700,
            'height': 70,
            'width': 450
            })
        ])

    elif tab == 'tab-24': # Rebels, Resources, & Homicide: Next Steps 
        return html.Div([
            html.H2('Rebels, Resrouces, & Homicide',
            style={'textAlign': 'center'}),
            dcc.Markdown('''
            * November: Complete merging and generating variables
            * December: Statistical analysis
            * January & February: Writing and editing
            ''',
            style={
                'marginTop': 100,
                'marginLeft': 350,
                'fontSize': 24}),
            html.Img(src=app.get_asset_url('UB_Horizontal.png'),
            style={'position': 'fixed',
            'bottom': 0,
            'left': 700,
            'height': 70,
            'width': 450
            })
        ])

    elif tab == 'tab-31': # Prediction & Theories of Violence: Social Disorganization 
        return html.Div([
            html.H2('Prediction & Theories of Violence',
            style={'textAlign': 'center'}),
            dcc.Markdown('''
            * Social disorganization
                * Population turnover
                * Heterogeneity

            * Social disorganization causes crime as well as other ills, 
            such as poor economic performance
            
            _Hypothesis 1: A model that includes enthnolinguistic 
            fractionalization, changes in population, and IDPs will better 
            predict postconflict crime than the null model._
            ''',
            style={
                'marginTop': 100,
                'marginLeft': 350,
                'fontSize': 24}),
            html.Img(src=app.get_asset_url('UB_Horizontal.png'),
            style={'position': 'fixed',
            'bottom': 0,
            'left': 700,
            'height': 70,
            'width': 450
            })
        ])

    elif tab == 'tab-32': # Prediction & Theories of Violence: Political Economy of Crime
        return html.Div([
            html.H2('Prediction & Theories of Violence',
            style={'textAlign': 'center'}),
            dcc.Markdown('''
            * Unemployment

            * Inequality

            _Hypothesis 2: A model that includes unemployment and income 
            inequality will predict postconflict crime better than the 
            null model._
            ''',
            style={
                'marginTop': 100,
                'marginLeft': 350,
                'fontSize': 24}),
            html.Img(src=app.get_asset_url('UB_Horizontal.png'),
            style={'position': 'fixed',
            'bottom': 0,
            'left': 700,
            'height': 70,
            'width': 450
            })
        ])

    elif tab == 'tab-33': # Prediction & Theories of Violence: Organizational Theory
        return html.Div([
            html.H2('Prediction & Theories of Violence',
            style={'textAlign': 'center'}),
            dcc.Markdown('''

            _Hypothesis 3: A model that includes rebel contraband will predict 
            postconflict crime better than the null model._
            ''',
            style={
                'marginTop': 100,
                'marginLeft': 350,
                'fontSize': 24}),
            html.Img(src=app.get_asset_url('UB_Horizontal.png'),
            style={'position': 'fixed',
            'bottom': 0,
            'left': 700,
            'height': 70,
            'width': 450
            })
        ])

    elif tab == 'tab-34': # Prediction & Theories of Violence: Machine Learning
        return html.Div([
            html.H2('Prediction & Theories of Violence',
            style={'textAlign': 'center'}),
            html.Img(src=app.get_asset_url('overfitting.png'),
            style={'marginLeft': 475}),
            html.Img(src=app.get_asset_url('good_fit.png'),
            style={'marginLeft': 475}),
            html.Img(src=app.get_asset_url('UB_Horizontal.png'),
            style={'position': 'fixed',
            'bottom': 0,
            'left': 700,
            'height': 70,
            'width': 450
            })
        ])

    elif tab == 'tab-35': # Prediction & Theories of Violence: Research Design
        return html.Div([
            html.H2('Prediction & Theories of Violence',
            style={'textAlign': 'center'}),
            dcc.Markdown('''
            * Null model 
                * DV: Homicide rate
                * Controls: BRDs, OSV, conflict duration, GDP per capita, Polity5
                * Model: OLS, SDG, random forests
            
            * Social Disorganization
                * IVs: Ethnolinguistic fractionalizaton, population change, IDPs

            * Political Economy
                * Unemployment, Gini

            * Organizational Theory
                * Illicit funding
            ''',
            style={
                'marginTop': 100,
                'marginLeft': 350,
                'fontSize': 24}),
            html.Img(src=app.get_asset_url('UB_Horizontal.png'),
            style={'position': 'fixed',
            'bottom': 0,
            'left': 700,
            'height': 70,
            'width': 450
            })
        ])

    elif tab == 'tab-36': # Prediction & Theories of Violence: Initial Results 
        return html.Div([
            html.H2('Prediction & Theories of Violence',
            style={'textAlign': 'center'}),
            html.Img(src=app.get_asset_url('null_model_ir.png'),
            style={'marginLeft': 555,
            'marginTop': 50,
            'marginBottom': 50}),
            html.Img(src=app.get_asset_url('random_forests_ir.png'),
            style={'marginLeft': 455}),
            html.Img(src=app.get_asset_url('UB_Horizontal.png'),
            style={'position': 'fixed',
            'bottom': 0,
            'left': 700,
            'height': 70,
            'width': 450
            })
        ])

    elif tab == 'tab-37': # Prediction & Theories of Violence: Nest Steps 
        return html.Div([
            html.H2('Prediction & Theories of Violence',
            style={'textAlign': 'center'}),
            dcc.Markdown('''
            * November and December: building datasets with slow and fast moving variables
            * December: optimizing the null model
            * January: traning and optimizing the theoretical models
            * February: robustness and stability testing
            * March and April: writing and editing
            ''',
            style={
                'marginTop': 100,
                'marginLeft': 350,
                'fontSize': 24}),
            html.Img(src=app.get_asset_url('UB_Horizontal.png'),
            style={'position': 'fixed',
            'bottom': 0,
            'left': 700,
            'height': 70,
            'width': 450
            })
        ])


@app.callback(Output('tab-99', 'children'),
    [Input('tabs-example', 'value')])
def update_tabs(value):
    if value == 'tab-99': # Appendicies
        return dcc.Tabs(id="subtabs", value="subtab-1", children=[
            dcc.Tab(label='Media & Murder code', id='subtab1',
            value='subtab1', children=[
                dcc.Markdown('''

                Scraping the news (Shell):

                ```sh
                #!/bin/bash -w

                for i in {1..1059} # from page 1 to the last page of the section
               do
                      curl "https://www.elespectador.com/tags/farc/$i/" | # get the html
                      grep -o "<a class=\"Card-FullArticle\" href=.*" | # find the links 
                      # replace the text before and after the link to get a usable url
                      sed "s/<a class=\"Card-FullArticle\" href=\"/https\:\/\/www\.elespectador\.com/" | 
                      sed "s/\">Ver noticia completa.*//" |
                      uniq # make sure each url only occurs once
              done > elespectador/elespectador_urls.txt # save urls
  
              counter=1
              for i in $(cat elespectador/elespectador_urls.txt)
              do
                      curl $i | # get htlm for each saved url from the last loop
                      html2text | # convert to plain text
                      sed -n '/\*\*\*\*\*\*/,$p'  | # cut everything before article
                      # Cut everything after end of article
                      sed -n '/Comparte en redes/q;p' > elespectador/elespectador$counter.txt
                      counter=$((counter+1))
              done

                ```
                General scraping program (Shell):
                ```sh
                #!/bin/bash -w

                maxpages=2
                url=https://www.elpais.com.co/noticias/farc
                link_name="href=.*class=\"news-title\""
                pre_link_find="^.*href=\""
                pre_link_replace="https\:\/\/www\.elpais\.com\.co"
                post_link_find="\"\stitle.*"
                post_link_replace=
                
                function getURLS { 
                for i in {1..2}
                do         
                    curl $url |
                    grep "$link_name" |
                    sed "s/$pre_link_find/$pre_link_replace/" |
                    sed "s/$post_link_find/$post_link_replace/" |
                    uniq 
                done > test_urls.txt
                }
                getURLS                                                                                                      
                                                                                                              
                file_name="elpais/elpais_urls.txt"                                                                           
                cut_head='Escuchar este artículo/,$p'                                                                        
                cut_tail='Conecta_con_la_verdad._Suscríbete_a_elpais.com.co/q;p'                                             
                                                                                                                 
                function getArticles {                                                                                       
                    for $i in $(cat $file_name)                                                                                               
                    do                                                                                                           
                        curl $i |                                                                                    
                        html2text |                                                                                  
                        sed -n "$cut_head" |                                                                         
                        sed -n "$cut_tail" |                                                                         
                        tail -n+$cut_lines > test_text.txt                                                           
                        counter=$((counter+1))                                                                       
                    done                                                                                                 
                }
                getArticles  
                ```

                Sentiment analysis example (Python):

                ```py
                from classifier import *

                clf = SentimentClassifier()

                ex_1 = "Las FARC estaban ahi" 
                ex_2 = "Las personas estaban ahi"
                print(clf.predict(ex_1))
                print(clf.predict(ex_2))

                ex_3 = "Los guerrilleros atacaran al pueblo, matando a mas de un mil de personas inocentes"
                ex_4 = "Los heroes ayudaron al pueblo reconstuir las casas"
                ex_5 = "Las personas hicieron una cosa en un lugar"
                print(clf.predict(ex_3))
                print(clf.predict(ex_4))
                print(clf.predict(ex_5))
                ```

                Statistical analysis (R):

                ```r
                library(stargazer)
                library(tidyverse)
                
                dat <- read_csv('data/diss_propopal_media&murder.csv')

                m1 <- lm(homicidios ~ sent + gdp, data = dat)
                m2 <- lm(homicidios ~ sent + growth, data = dat)
                m3 <- lm(homicidios ~ sent + desempleo, data = dat)
                m4 <- lm(homicidios ~ sent + growth + desempleo, data = dat)
                stargazer(m1, m2, m3, m4, type = 'text', title = 'table 2', style = 'ajps',             
                    covariate.labels = c('Sentiment', 'GDP', 'GDP Growth', 'Unemployment'), dep.var.caption = NULL,
                    dep.var.labels = c('Homicides'),
                    star.char = c("+", "*", "**", "***"),
                    star.cutoffs = c(0.1, 0.05, 0.01, 0.001),
                    notes = c("+ p<0.1; * p<0.05; ** p<0.01; *** p<0.001"), 
                    notes.append = F)
                ```
                ''',
                style={'marginLeft': 375})
            ]),

            dcc.Tab(label='Forest Models', 
            id='subtab3', value='subtab3', children=[
                dcc.Graph(figure=fig)
                ]),

            dcc.Tab(label='Prediction & Theories of Violence code', 
            id='subtab4', value='subtab4', children=[
                dcc.Markdown('''
                ```py
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
            from sklearn.impute import SimpleImputer
            from sklearn.linear_model import LinearRegression
            from sklearn.linear_model import SGDRegressor
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.preprocessing import StandardScaler
            
            import pandas as pd
            import matplotlib.pyplot as plt 
            import numpy as np
            import random
            
            random.seed(3476)
            # importing the dataset
            fp = '~/Desktop/dissertation/predict/data/newton_data4-30-2020.csv'
            data = pd.read_csv(fp, encoding='latin_1')
            print(data.columns)
            data = data[data['hom_rate'].notna()]
            
            fp = '~/Desktop/dissertation/predict/data/F&Lethnic.csv'
            ef = pd.read_csv(fp)
            cols = ['year', 'mtnest', 'ef', 'ccode']
            ef = ef[cols]
            data = pd.merge(data, ef,
                            how="left", on=["ccode", "year"])
            
            # Null model
            y = data.hom_rate # target feature
            
            features = ['con_dur', 'peace_dur', 'totalbrds', 'totalosv', 'mtnest', 
                  'v2x_libdem']
            
            X = data[features] # imput features
            
            
            # splitting the data
            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
            
            # imputing NaNs
            my_imputer = SimpleImputer()
            X_train_imputed = pd.DataFrame(my_imputer.fit_transform(X_train))
            X_test_imputed = pd.DataFrame(my_imputer.transform(X_test))
            
            X_train_imputed.columns = X_train.columns
            X_test_imputed.columns = X_test.columns
            
            # feature scaling
            sc = StandardScaler()
            X_train_sc = sc.fit_transform(X_train_imputed)
            X_test_sc = sc.transform(X_test_imputed)
            
            # linear regression
            lin_reg = LinearRegression()
            lin_reg.fit(X_train_sc, y_train)
            preds = lin_reg.predict(X_test_sc)
            
            print('\nNull Models\n')
            print('Linear Regression\n')
            print('Mean absoulte error: %.2f'
                    % mean_absolute_error(y_test, preds))
            
            print('Coefficients: \n', lin_reg.coef_)
            # The mean squared error
            print('Mean squared error: %.2f'
                  % mean_squared_error(y_test, preds))
            # The coefficient of determination: 1 is perfect prediction
            print('Coefficient of determination: %.2f'
                  % r2_score(y_test, preds))
            
            # stochastic gradient descent
            sgd_reg = SGDRegressor(max_iter=500, penalty=None, eta0=0.001)
            sgd_reg.fit(X_train_sc, y_train)
            preds = sgd_reg.predict(X_test_sc)
            
            print('\nStocastic Gradient Descent\n')
            print('Mean absoulte error: %.2f'
                    % mean_absolute_error(y_test, preds))
            
            print('Coefficients: \n', sgd_reg.coef_)
            # The mean squared error
            print('Mean squared error: %.2f'
                  % mean_squared_error(y_test, preds))
            # The coefficient of determination: 1 is perfect prediction
            print('Coefficient of determination: %.2f'
                  % r2_score(y_test, preds))
            
            # random forest
            def rf_models(X_train, y_train, X_test, y_test, n_estimators,
                  random_state, model_name):
                  rf_reg = RandomForestRegressor(n_estimators=n_estimators,
                        random_state=random_state)
                  rf_reg.fit(X_train, y_train)
                  preds = rf_reg.predict(X_test)
                  print('\nRandom Forest: %s\n' % model_name)
                  print('Mean absolute error: %.2f' % mean_absolute_error(y_test, preds))
                  print('Mean squared error: %.2f' % mean_squared_error(y_test, preds))
                  print('Coefficient of determination: %.2f' % r2_score(y_test, preds))
            
            
            rf_models(X_train_sc, y_train, X_test_sc, y_test, 200, 0, "Null Model")
            
            # Social Disorganization Model
            
            features = ['con_dur', 'peace_dur', 'totalbrds', 'totalosv', 'mtnest', 
                  'v2x_libdem', 'ef']
            
            X = data[features] # imput features
            
            # splitting the data
            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
            
            # imputing NaNs
            my_imputer = SimpleImputer()
            X_train_imputed = pd.DataFrame(my_imputer.fit_transform(X_train))
            X_test_imputed = pd.DataFrame(my_imputer.transform(X_test))
            
            X_train_imputed.columns = X_train.columns
            X_test_imputed.columns = X_test.columns
            
            # feature scaling
            sc = StandardScaler()
            X_train_sc = sc.fit_transform(X_train_imputed)
            X_test_sc = sc.transform(X_test_imputed)
            
            rf_models(X_train_sc, y_train, X_test_sc, y_test, 
                  200, 0, "Social Disorganization Model")
            
            # Political Economy Model
            features = ['con_dur', 'peace_dur', 'totalbrds', 'totalosv', 'mtnest', 
                  'v2x_libdem', 'rgdppc']
            
            X = data[features] # imput features
            
            # splitting the data
            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
            
            # imputing NaNs
            my_imputer = SimpleImputer()
            X_train_imputed = pd.DataFrame(my_imputer.fit_transform(X_train))
            X_test_imputed = pd.DataFrame(my_imputer.transform(X_test))
            
            X_train_imputed.columns = X_train.columns
            X_test_imputed.columns = X_test.columns
            
            # feature scaling
            sc = StandardScaler()
            X_train_sc = sc.fit_transform(X_train_imputed)
            X_test_sc = sc.transform(X_test_imputed)
            
            rf_models(X_train_sc, y_train, X_test_sc, y_test, 
                  200, 0, "Political Economy Model")
            
            # Organizational model
            
            features = ['con_dur', 'peace_dur', 'totalbrds', 'totalosv', 'mtnest', 
                  'v2x_libdem', 'drugs_any', 'gems_any', 'agriculture_any', 
                  'minerals_any', 'fuel_any']
            
            X = data[features] # imput features
            
            # splitting the data
            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
            
            # imputing NaNs
            my_imputer = SimpleImputer()
            X_train_imputed = pd.DataFrame(my_imputer.fit_transform(X_train))
            X_test_imputed = pd.DataFrame(my_imputer.transform(X_test))
            
            X_train_imputed.columns = X_train.columns
            X_test_imputed.columns = X_test.columns
            
            # feature scaling
            sc = StandardScaler()
            X_train_sc = sc.fit_transform(X_train_imputed)
            X_test_sc = sc.transform(X_test_imputed)
            
            rf_models(X_train_sc, y_train, X_test_sc, y_test, 
                  200, 0, "Organizational Model")
                                ``` 
                                ''',
                                style={'marginLeft': 375})
                        ])
                    ])
             
if __name__ == '__main__':
    app.run_server(debug=True)