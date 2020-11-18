import dash
import dash_html_components as html
import dash_core_components as dcc 
from dash.dependencies import Input, Output

app = dash.Dash()

subtab_style = {
    'width': '367px'
}

app.layout = html.Div([
    dcc.Tabs(id='main_tabs', value='tab-1', children=[
        dcc.Tab(label='Introduction', id='tab-1', value='tab-1', children=[]),
        dcc.Tab(label='Media & Murder', id='tab-2', value='tab-2', children=[]),
        dcc.Tab(label='Rebels, Resources, & Homicide', id='tab-3', value='tab-3', children=[]),
        dcc.Tab(label='Prediction & Theories of Violence', id='tab-4', value='tab-4', children=[]),
        dcc.Tab(label='Appendicies', id='tab-5', value='tab-5', children=[]),
    ]),
    html.Div(id='main_tabs_content')
])

@app.callback(Output('main_tabs_content', 'children'),
    [Input('main_tabs', 'value')])

def render_content(value):
    if value == 'tab-1':
        return html.Div(html.Img(src=app.get_asset_url('UB_Horizontal.png'),
            style={'position': 'fixed',
            'bottom': 0,
            'left': 700,
            'height': 70,
            'width': 450})),
    
    elif value == 'tab-2':
        return html.Div(html.Img(src=app.get_asset_url('UB_Horizontal.png'),
            style={'position': 'fixed',
            'bottom': 0,
            'left': 700,
            'height': 70,
            'width': 450})),
    
    elif value == 'tab-3':
        return html.Div(html.Img(src=app.get_asset_url('UB_Horizontal.png'),
            style={'position': 'fixed',
            'bottom': 0,
            'left': 700,
            'height': 70,
            'width': 450})),
    
    elif value == 'tab-4':
        return html.Div(html.Img(src=app.get_asset_url('UB_Horizontal.png'),
            style={'position': 'fixed',
            'bottom': 0,
            'left': 700,
            'height': 70,
            'width': 450})),
    
    elif value == 'tab-5':
        return html.Div()

@app.callback(Output('tab-1', 'children'),
    [Input('main_tabs', 'value')])

def update_content(value):
    if value == 'tab-1':
        return dcc.Tabs(id='subtabs', value='subtab-1', children=[
            dcc.Tab(label='Title', id='subtab1', value='subtab1', children=[
                html.Div([
            dcc.Markdown('''

            &nbsp;

            &nbsp;

            &nbsp;

            # The War is Over but the Killing Remains
            ## Criminal Violence in Postconflict States

            &nbsp;

            &nbsp;

            &nbsp;
            
            ### Christopher Newton

            &nbsp;

            &nbsp;
            
            &nbsp;

            ### Committee Chair: Professor Jacob Kathman
            ### Committee Member: Professor Michelle A. Benson
            ### Committee Member: Professor Jacob Neiheisel
            ''', style={'textAlign': 'center',
            'marginLeft': 200}),
            ])], style=subtab_style, selected_style=subtab_style),

            dcc.Tab(label='Criminal Violence & Negative Peace', id='subtab2', value='subtab2', children=[
            html.Div([dcc.Markdown('''
            &nbsp;

            ## Criminal Violence & Negative Peace

            * Criminal versus Political Violence

            * Negative versus Positive Peace
            ''',
            style={
                'marginTop': 50,
                'marginLeft': 200,
                'fontSize': 24}),
            ])])
        ], vertical=True),

@app.callback(Output('tab-2', 'children'),
    [Input('main_tabs', 'value')])

def update_content(value):
    if value == 'tab-2':
        return dcc.Tabs(id='subtabs', value='subtab-2', children=[
            dcc.Tab(label='Rwanda', id='subtab1', value='subtab1', children=[
                html.Div([ 
                    dcc.Markdown('''
                    ## Media & Murder

                    * _Kangura_

                    * Radio Télévision Libre des Mille Collines (RTLM)
                    ''',
                    style={
                        'marginTop': 100,
                        'marginLeft': 200,
                        'fontSize': 24}),
                    ])], style=subtab_style, selected_style=subtab_style),
            dcc.Tab(label='Effect of Media', id='subtab2', value='subtab2', children=[
                html.Div([
            dcc.Markdown('''
            ## Media & Murder

            * Media matters:
                * Elections
                * Trust in institutions
                * Violence

            * Reintegration 
            ''',
            style={
                'marginTop': 100,
                'marginLeft': 200,
                'fontSize': 24}),
            ])]),
            dcc.Tab(label='Reintegration', id='subtab3', value='subtab3', children=[
                html.Div([ dcc.Markdown('''
            ## Media & Murder

            * Individual characteristics:
                * Unit atrocities
                * Acceptace by family & community
                * Forming of connections outside of unit

            * Broader programs have limited effects
            ''',
            style={
                'marginTop': 100,
                'marginLeft': 200,
                'fontSize': 24}),
            ])]),
            dcc.Tab(label='Theory', id='subtab4', value='subtab4', children=[
                html.Div([dcc.Markdown('''
            ## Media & Murder

            * Negative media coverage will cause animosity towards former rebels

            * Animosity will prevent successful reintegration
                * Lack of opportunities within society will lead to criminality
                * Animosity against former rebels can lead to violence against them
            
            _Hypothesis 1: Increased negative media of former rebels will lead to an increase in criminal violence._
            ''',
            style={
                'marginTop': 100,
                'marginLeft': 200,
                'marginRight': 100,
                'fontSize': 24}),
            ])]),
            dcc.Tab(label='Colombia', id='subtab5', value='subtab5', children=[
                html.Div([
            dcc.Markdown('''
            ## Media & Murder

            * Highly developed

            * Recent end to FARC conflict

            * Spanish speaking
            ''',
            style={
                'marginTop': 100,
                'marginLeft': 200,
                'fontSize': 24}),
            ])]),
            dcc.Tab(label='Sentiment Analysis', id='subtab6', value='subtab6', children=[
                html.Div([
            dcc.Markdown('''
            ## Media & Murder

            Sentiment anaysis:
            * pretrained model
            * 0-1
            ''',
            style={
                'marginTop': 100,
                'marginLeft': 200,
                'fontSize': 24}),
                html.P('The insurgents attacked the village, killing more that 1,000 innocent people. = 0.06',
            style={
                'marginTop': 50,
                'marginLeft': 200,
                'fontSize': 24}),
            html.P('(Los guerrilleros atacaron al pueblo, matando a mas de un mil personas inocentes.)',
            style={
                'marginTop': 0,
                'marginLeft': 200,
                'fontSize': 24}),
            html.P('The heroes helped the village rebuild their houses. = 0.84',
            style={
                'marginTop': 50,
                'marginLeft': 200,
                'fontSize': 24}),
		    html.P('(Los héroes ayudaron al pueblo reconstuir las casas.)',
            style={
                'marginTop': 0,
                'marginLeft': 200,
                'fontSize': 24}),
            html.P('The people did a thing in a place. = 0.59',
            style={
                'marginTop': 50,
                'marginLeft': 200,
                'fontSize': 24}),
            html.P('(Las personas hicieron una cosa en un lugar.)',
            style={
                'marginTop': 0,
                'marginLeft': 200,
                'fontSize': 24}),
            ])]),
            dcc.Tab(label='Research Design', id='subtab7', value='subtab7', children=[
                html.Div([dcc.Markdown('''
                ## Media & Murder

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
                'marginLeft': 200,
                'fontSize': 24}),
            ])]),
            dcc.Tab(label='Initial Results', id='subtab8', value='subtab8', children=[
            html.Img(src=app.get_asset_url('media&murder_ir.png'),
            style={
                'marginTop': 50,
                'marginLeft': 100}),
            ]),
            dcc.Tab(label='Next Steps', id='subtab9', value='subtab9', children=[
                html.Div([
            dcc.Markdown('''
            ## Media & Murder

            * End of Deceber: Finish scraping and cleaning text data
            * End of December: Conduct sentiment analysis
            * January: Merging all data
            * February: Statistical analysis
            * March and April: Writing and editing
            ''',
            style={
                'marginTop': 100,
                'marginLeft': 200,
                'fontSize': 24}),
            ])]),
                    ], vertical=True)

@app.callback(Output('tab-3', 'children'),
    [Input('main_tabs', 'value')])

def update_content(value):
    if value == 'tab-3':
        return dcc.Tabs(id='subtabs', value='subtab-3', children=[
            dcc.Tab(label='Organizational Explanations', id='subtab1', value='subtab1', children=[
                html.Div([ 
                    html.H2('Rebels, Resources, & Homicide',
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
                        'marginLeft': 200,
                        'fontSize': 24}),
                    ])], style=subtab_style, selected_style=subtab_style),
            dcc.Tab(label='Post-Conflict', id='subtab2', value='subtab2', children=[
                html.Div([
            html.H2('Rebels, Resources, & Homicide',
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
                'marginLeft': 200,
                'fontSize': 24}),
            ])]),
            dcc.Tab(label='Research Design', id='subtab3', value='subtab3', children=[
                html.Div([ 
                    html.H2('Rebels, Resources, & Homicide',
                    style={'textAlign': 'center'}),
                    dcc.Markdown('''
                    ''',
                    style={
                        'marginTop': 100,
                        'marginLeft': 200,
                        'fontSize': 24}),
                    ])]),
            dcc.Tab(label='Nest Steps', id='subtab4', value='subtab4', children=[
                html.Div([ 
                    html.H2('Rebels, Resources, & Homicide',
                    style={'textAlign': 'center'}),
                    dcc.Markdown('''
                    ''',
                    style={
                        'marginTop': 100,
                        'marginLeft': 200,
                        'fontSize': 24}),
                    ])]),
                    ], vertical=True),
    
@app.callback(Output('tab-4', 'children'),
    [Input('main_tabs', 'value')])

def update_content(value):
    if value == 'tab-4':
        return dcc.Tabs(id='subtabs', value='subtab-4', children=[
            dcc.Tab(label='Social Disorganization', id='subtab1', value='subtab1', children=[
                html.Div([ 
                    html.H2('Prediction & Theories of Violence',
                    style={'textAlign': 'center'}),
                    dcc.Markdown('''
                    ''',
                    style={
                        'marginTop': 100,
                        'marginLeft': 200,
                        'fontSize': 24}),
                    ])], style=subtab_style, selected_style=subtab_style),
            dcc.Tab(label='Political Economy of Crime', id='subtab2', value='subtab2', children=[        
                html.Div([ 
                    html.H2('Prediction & Theories of Violence',
                    style={'textAlign': 'center'}),
                    dcc.Markdown('''
                     ''',
                     style={
                        'marginTop': 100,
                        'marginLeft': 200,
                        'fontSize': 24}),
                    ])]),    
            dcc.Tab(label='Organizational Theory', id='subtab3', value='subtab3', children=[        
                html.Div([ 
                    html.H2('Prediction & Theories of Violence',
                    style={'textAlign': 'center'}),
                    dcc.Markdown('''
                     ''',
                     style={
                        'marginTop': 100,
                        'marginLeft': 200,
                        'fontSize': 24}),
                    ])]),    
            dcc.Tab(label='Machine Learning', id='subtab4', value='subtab4', children=[        
                html.Div([ 
                    html.H2('Prediction & Theories of Violence',
                    style={'textAlign': 'center'}),
                    dcc.Markdown('''
                     ''',
                     style={
                        'marginTop': 100,
                        'marginLeft': 200,
                        'fontSize': 24}),
                    ])]),    
            dcc.Tab(label='Research Design', id='subtab5', value='subtab5', children=[        
                html.Div([ 
                    html.H2('Prediction & Theories of Violence',
                    style={'textAlign': 'center'}),
                    dcc.Markdown('''
                     ''',
                     style={
                        'marginTop': 100,
                        'marginLeft': 200,
                        'fontSize': 24}),
                    ])]),    
            dcc.Tab(label='Initial Results', id='subtab6', value='subtab6', children=[        
                html.Div([ 
                    html.H2('Prediction & Theories of Violence',
                    style={'textAlign': 'center'}),
                    dcc.Markdown('''
                     ''',
                     style={
                        'marginTop': 100,
                        'marginLeft': 200,
                        'fontSize': 24}),
                    ])]),    
            dcc.Tab(label='Next Steps', id='subtab7', value='subtab7', children=[        
                html.Div([ 
                    html.H2('Prediction & Theories of Violence',
                    style={'textAlign': 'center'}),
                    dcc.Markdown('''
                     ''',
                     style={
                        'marginTop': 100,
                        'marginLeft': 200,
                        'fontSize': 24}),
                    ])]),    
                        ], vertical=True),

@app.callback(Output('tab-5', 'children'),
    [Input('main_tabs', 'value')])

def update_content(value):
    if value == 'tab-5':
        return dcc.Tabs(id='subtabs', value='subtab-5', children=[
            dcc.Tab(label='Text Scraping Code', id='subtab1', value='subtab1', children=[
                html.Div([ 
                    html.H2('',
                    style={'textAlign': 'center'}),
                    dcc.Markdown('''
                    ```sh
                    #!/bin/bash -w
                    ```
                    \# directories need to be made in the WD before running
                    
                    \# This first function scrapes the URLS for the relevant section
                    ```sh
                    function getURLS { 
                        curl "$url" |
                        grep "$link_name" |
                        sed "s/$pre_link_find/$pre_link_replace/" |
                        sed "s/$post_link_find/$post_link_replace/" |
                        uniq > $file_name
                    }
                    ```
                    \# This function takes the URLS from the first a get the article text
                    ```sh
                    function getArticles {
                        counter=1
                        for i in $(cat $file_name)
                        do
                            curl $i |
                            html2text |
                            sed -n "$cut_head" |
                            sed -n "$cut_tail" |
                            tail -n+$cut_lines > $article_file$counter.txt
                            counter=$((counter+1))
                        done
                    }
                    ```
                    \#### Scraping ####

                    \## Bogota ##

                    \# El Espectador: www.elespectador.com
                    ```sh
                    url=https://www.elespectador.com/tags/farc/[1-4]/
                    link_name="<a class=\"Card-FullArticle\" href=.*"
                    pre_link_find="^.*href=\""
                    pre_link_replace="https\:\/\/www\.elespectador\.com"
                    post_link_find="\">Ver noticia completa.*"
                    post_link_replace=
                    file_name=test_urls.txt
                    
                    cut_head='/Escuchar este artículo/,$p'
                    cut_tail='/Conecta_con_la_verdad._Suscríbete_a_elpais.com.co/q;p'
                    cut_lines=2
                    article_file=test/text_
                    
                    getURLS
                    getArticles
                    ```
                    ''',
                    style={
                        'marginTop': 10,
                        'marginLeft': 30,
                        'fontSize': 24}),
                    ])], style=subtab_style, selected_style=subtab_style),
            dcc.Tab(label='Regression Trees', id='subtab2', value='subtab2', children=[
                html.Div([
            html.Img(src=app.get_asset_url('reg_tree.png'),
            )])]),
            dcc.Tab(label='Machine Learning Code', id='subtab3', value='subtab3', children=[
                html.Div([ 
                    dcc.Markdown('''
                    ```py
                    from sklearn.model_selection import train_test_split
                    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
                    from sklearn.impute import SimpleImputer
                    from sklearn.linear_model import LinearRegression
                    from sklearn.linear_model import SGDRegressor
                    from sklearn.ensemble import RandomForestRegressor
                    from sklearn.preprocessing import StandardScaler
                    from sklearn.pipeline import Pipeline
                    
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
                    
                    # linear regression
                    def lin_models(X_train, y_train, X_test, y_test, model_name):
                          pipe = Pipeline([('imputer', SimpleImputer(strategy="median")),
                              ('std_scaler', StandardScaler()),
                              ('lin_reg', LinearRegression())])
                          pipe.fit(X_train, y_train)
                          preds = pipe.predict(X_test)
                          print('Linear Regression: %s' % model_name)
                          print('Mean absolute error: %.2f' % mean_absolute_error(y_test, preds))
                          print('Mean squared error: %.2f' % mean_squared_error(y_test, preds))
                          print('Coefficient of determination: %.2f' % r2_score(y_test, preds))
                    
                    lin_models(X_train, y_train, X_test, y_test, "Null Model OLS")
                    
                    # stochastic gradient descent
                    def sgd_models(X_train, y_train, X_test, y_test, max_iter, 
                            penalty, eta0, model_name):
                            """
                            """
                            pipe = Pipeline([('imputer', SimpleImputer(strategy="median")),
                               ('std_scaler', StandardScaler()),
                               ('sgd_reg', SGDRegressor(max_iter=max_iter, 
                                   penalty=penalty, eta0=eta0))])
                            pipe.fit(X_train, y_train)
                            preds = pipe.predict(X_test)
                            print('Stocastic Gradient Descent %s' % model_name) 
                            print('Mean absoulte error: %.2f' % mean_absolute_error(y_test, preds))
                            print('Mean squared error: %.2f' % mean_squared_error(y_test, preds))
                            print('Coefficient of determination: %.2f' % r2_score(y_test, preds))
                    
                    sgd_models(X_train, y_train, X_test, y_test, 
                            500, None, 0.001, "Null Model SGD")
                    
                    # random forest
                    def rf_models(X_train, y_train, X_test, y_test, n_estimators,
                          random_state, model_name):
                          """
                          """
                          pipe = Pipeline([('imputer', SimpleImputer(strategy="median")),
                            ('std_scaler', StandardScaler()),
                            ('rf', RandomForestRegressor(n_estimators-n_estimators,
                                random_state=random_state))])
                          pipe.fit(X_train, y_train)
                          preds = rf_reg.predict(X_test)
                          print('Random Forest: %s' % model_name)
                          print('Mean absolute error: %.2f' % mean_absolute_error(y_test, preds))
                          print('Mean squared error: %.2f' % mean_squared_error(y_test, preds))
                          print('Coefficient of determination: %.2f' % r2_score(y_test, preds))
                    
                    
                    rf_models(X_train, y_train, X_test, y_test, 200, 0, "Null Model")
                    
                    # Social Disorganization Model
                    
                    features = ['con_dur', 'peace_dur', 'totalbrds', 'totalosv', 'mtnest', 
                          'v2x_libdem', 'ef']
                    
                    X = data[features] # imput features
                    
                    # splitting the data
                    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
                    
                    rf_models(X_train, y_train, X_test, y_test, 
                          200, 0, "Social Disorganization Model")
                    
                    # Political Economy Model
                    features = ['con_dur', 'peace_dur', 'totalbrds', 'totalosv', 'mtnest', 
                          'v2x_libdem', 'rgdppc']
                    
                    X = data[features] # imput features
                    
                    # Organizational model
                    
                    features = ['con_dur', 'peace_dur', 'totalbrds', 'totalosv', 'mtnest', 
                          'v2x_libdem', 'drugs_any', 'gems_any', 'agriculture_any', 
                          'minerals_any', 'fuel_any']
                    
                    X = data[features] # imput features
                    
                    # splitting the data
                    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
                    
                    rf_models(X_train, y_train, X_test, y_test, 
                          200, 0, "Organizational Model")
                    ```
                    ''',
                    style={
                        'marginTop': 10,
                        'marginLeft': 30,
                        'fontSize': 24}),
                    ])])], vertical=True)

if __name__ == '__main__':
    app.run_server(debug=True)

