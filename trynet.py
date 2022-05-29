import csv
from re import template
from turtle import width
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import plotly
import seaborn as sns
import plotly.express as px
from plotly_calplot import calplot
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from flask import Flask, render_template , url_for , request
from sklearn.cluster import KMeans
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import json

app = Flask(__name__,template_folder="templates")
@app.route('/')
def home():
 df = pd.read_csv('cars.csv')
 #data clean
 df['car'] = df.Make + ' ' + df.Model
 c = ['Make','Model','car','Variant','Body_Type','Fuel_Type','Fuel_System','Type','Drivetrain','Ex-Showroom_Price','Displacement','Cylinders',
    'ARAI_Certified_Mileage','Power','Torque','Fuel_Tank_Capacity','Height','Length','Width','Doors','Seating_Capacity','Wheelbase','Number_of_Airbags']
 df_full = df.copy()
 df['Ex-Showroom_Price'] = df['Ex-Showroom_Price'].str.replace('Rs. ','',regex=False)
 df['Ex-Showroom_Price'] = df['Ex-Showroom_Price'].str.replace(',','',regex=False)
 df['Ex-Showroom_Price'] = df['Ex-Showroom_Price'].astype(int)
 df = df[c]
 df = df[~df.ARAI_Certified_Mileage.isnull()]
 df = df[~df.Make.isnull()]
 df = df[~df.Width.isnull()]
 df = df[~df.Cylinders.isnull()]
 df = df[~df.Wheelbase.isnull()]
 df = df[~df['Fuel_Tank_Capacity'].isnull()]
 df = df[~df['Seating_Capacity'].isnull()]
 df = df[~df['Torque'].isnull()]
 df['Height'] = df['Height'].str.replace(' mm','',regex=False).astype(float)
 df['Length'] = df['Length'].str.replace(' mm','',regex=False).astype(float)
 df['Width'] = df['Width'].str.replace(' mm','',regex=False).astype(float)
 df['Wheelbase'] = df['Wheelbase'].str.replace(' mm','',regex=False).astype(float)
 df['Fuel_Tank_Capacity'] = df['Fuel_Tank_Capacity'].str.replace(' litres','',regex=False).astype(float)
 df['Displacement'] = df['Displacement'].str.replace(' cc','',regex=False)
 df.loc[df.ARAI_Certified_Mileage == '9.8-10.0 km/litre','ARAI_Certified_Mileage'] = '10'
 df.loc[df.ARAI_Certified_Mileage == '10kmpl km/litre','ARAI_Certified_Mileage'] = '10'
 df['ARAI_Certified_Mileage'] = df['ARAI_Certified_Mileage'].str.replace(' km/litre','',regex=False).astype(float)
 df.Number_of_Airbags.fillna(0,inplace= True)
 df['price'] = df['Ex-Showroom_Price'] 
 df.drop(columns='Ex-Showroom_Price', inplace= True)
 df.price = df.price.astype(int)
 HP = df.Power.str.extract(r'(\d{1,4}).*').astype(int) * 0.98632
 HP = HP.apply(lambda x: round(x,2))
 TQ = df.Torque.str.extract(r'(\d{1,4}).*').astype(int)
 TQ = TQ.apply(lambda x: round(x,2))
 df.Torque = TQ
 df.Power = HP
 df.Doors = df.Doors.astype(int)
 df.Seating_Capacity = df.Seating_Capacity.astype(int)
 df.Number_of_Airbags = df.Number_of_Airbags.astype(int)
 df.Displacement = df.Displacement.astype(int)
 df.Cylinders = df.Cylinders.astype(int)
 df.columns = ['make', 'model','car', 'variant', 'body_type', 'fuel_type', 'fuel_system','type', 'drivetrain', 'displacement', 'cylinders',
            'mileage', 'power', 'torque', 'fuel_tank','height', 'length', 'width', 'doors', 'seats', 'wheelbase','airbags', 'price']

 num_cols = [ i for i in df.columns if df[i].dtype != 'object']
 km = KMeans(n_clusters=8, n_init=20, max_iter=400, random_state=0)
 clusters = km.fit_predict(df[num_cols])
 df['cluster'] = clusters
 df.cluster = (df.cluster + 1).astype('object')

 fig4 = px.pie(df,names=df.body_type.value_counts().index,values=df.body_type.value_counts()
,template='plotly_dark', width=600, height=800)
 fig4.update_traces(hoverinfo='label+percent', textinfo='percent', textfont_size=20 , title="Body Type Market Composition")
 graph4JSON=json.dumps(fig4,cls=plotly.utils.PlotlyJSONEncoder)
 
 fig = px.histogram(df, x="make", color="fuel_type",title="Fuel Type Vs Brand",template='plotly_dark', height=700,width=720).update_xaxes(categoryorder='total descending')
 graph1JSON=json.dumps(fig,cls=plotly.utils.PlotlyJSONEncoder)

 fig8 = px.pie(df, names='make', title="Brand Market Involvement",template='plotly_dark', height=700,width=720)
 fig8.update_traces(textposition='inside', textinfo='percent+label')
 graph8JSON=json.dumps(fig8,cls=plotly.utils.PlotlyJSONEncoder)

 fig2=px.scatter_3d(df,x='power',z='price',y='mileage',color='make',width=700,height=550 , title="Power VS Price VS Mileage Grouped by Brand",template='plotly_dark')
 fig2.update_layout(showlegend=True)
 graph2JSON=json.dumps(fig2,cls=plotly.utils.PlotlyJSONEncoder)

 df4 = pd.read_csv("segment.csv")
 fig79 = px.histogram(df4, x='Price',color='Credit', barmode='group', height=700,width=720,template='plotly_dark')
 graph79JSON=json.dumps(fig79,cls=plotly.utils.PlotlyJSONEncoder)

 fig89 = px.histogram(df4, x='Make',color='No of Dependents'
 , height=350,width=400, barmode='group',template='plotly_dark' ,labels={'x':'Model of Car', 'y':'Number of Purchases'})
 graph89JSON=json.dumps(fig89,cls=plotly.utils.PlotlyJSONEncoder)


 india_states = json.load(open('states_india.geojson'))
 df3 = pd.read_csv("statesales2019.csv")

 state_id_map = {}
 for feature in india_states['features']:
      feature['id'] = feature['properties']['state_code']
      state_id_map[feature['properties']['st_nm']] = feature['id']

 df3['id'] = df3['state'].apply(lambda x: state_id_map[x])

 fig68 = px.choropleth(df3, locations='id',
                              geojson=india_states,
                              color='2019sales',
                              scope='asia',
                              hover_name='state',
                              hover_data=['Q1sales2019',
                                  'Q2sales2019', 'Q3sales2019','Q4sales2019'],
                              template='plotly_dark',
                              height=600)
 fig68.update_layout(margin=dict(l=0, r=0, t=0, b=0))  
 fig68.update_geos(fitbounds="locations", visible=False)
 fig69 = px.bar(
                  df3,
                  x='state',
                  y='2019sales',
                  hover_data=['state', 'Q1sales2019',
                      'Q2sales2019', 'Q3sales2019','Q4sales2019'],
                  template='plotly_dark',
                  height=600,
            )
 graph68JSON=json.dumps(fig68,cls=plotly.utils.PlotlyJSONEncoder)
 graph69JSON=json.dumps(fig69,cls=plotly.utils.PlotlyJSONEncoder)
 
 fig3 = px.scatter(df, x="power", y="price", color="body_type", hover_data=['fuel_type'],width=700, height=800,title="Power & Price Relationship",template='plotly_dark')
 graph3JSON=json.dumps(fig3,cls=plotly.utils.PlotlyJSONEncoder)

 fig5 = make_subplots(rows=2, cols=2,subplot_titles=('Complete dataset count','Vehicles Under 19%','Vehicles Under 21%','Vehicles Under 48%'))
 x=df.displacement
 trace0 = go.Histogram(x=x)
 trace1 = go.Histogram(x=x,
                      xbins=dict(
                      start=0,
                      end=1200,
                      size=100),
                      autobinx=False
                     )
 trace2 = go.Histogram(x=x,
                      xbins=dict(
                      start=1200,
                      end=1500,
                      size=100),
                      autobinx=False
                     )
 trace3 = go.Histogram(x=x,
                      xbins=dict(
                      start=1500,
                      end=max(df.displacement),
                      size=200),
                      autobinx=False
                     )
 fig5.append_trace(trace0, 1, 1)
 fig5.append_trace(trace1, 1, 2)
 fig5.append_trace(trace2, 2, 1)
 fig5.append_trace(trace3, 2, 2)
 fig5.update_layout(height=900,width=600,template='plotly_dark')
 graph5JSON=json.dumps(fig5,cls=plotly.utils.PlotlyJSONEncoder)

 fig6 = px.scatter_matrix(df,dimensions=['displacement', 'mileage', 'power', 'price'],color="fuel_type" , title="Variation of Properties",template='plotly_dark', height=800,width=820)
 fig6.update_layout(
    title='Automobile Technology Variance',
    dragmode='select',
    width=600,
    height=600,
    hovermode='closest',
 )
 graph6JSON=json.dumps(fig6,cls=plotly.utils.PlotlyJSONEncoder)

 fig7 = px.sunburst(df, path=['body_type', 'fuel_type','fuel_tank'] , title="Body Type , Fuel Type based on Fuel Tank Capacity",template='plotly_dark', height=700,width=720)
 graph7JSON=json.dumps(fig7,cls=plotly.utils.PlotlyJSONEncoder)

 choice = "TSLA-USD"
 data = yf.download(tickers=choice, period = '5d', interval = '15m', rounding= True)
 fig99 = go.Figure()
 fig99.add_trace(go.Candlestick(x=data.index,open = data['Open'], high=data['High'], low=data['Low'], close=data['Close'], name = 'market data'))
 fig99.update_layout(title = choice + ' share price', yaxis_title = 'Stock Price (USD)',height=800,width=800)
 fig99.update_xaxes(
 rangeslider_visible=True,
 rangeselector=dict(
 buttons=list([
 dict(count=15, label='15m', step="minute", stepmode="backward"),
 dict(count=45, label='45m', step="minute", stepmode="backward"),
 dict(count=1, label='1h', step="hour", stepmode="backward"),
 dict(count=6, label='6h', step="hour", stepmode="backward"),
 dict(step="all")
 ]) ))
 graph99JSON=json.dumps(fig99,cls=plotly.utils.PlotlyJSONEncoder)

 choice = "UBER-USD"
 data = yf.download(tickers=choice, period = '5d', interval = '15m', rounding= True)
 fig100 = go.Figure()
 fig100.add_trace(go.Candlestick(x=data.index,open = data['Open'], high=data['High'], low=data['Low'], close=data['Close'], name = 'market data'))
 fig100.update_layout(title = choice + ' share price', yaxis_title = 'Stock Price (USD)',height=800,width=800)
 fig100.update_xaxes(
 rangeslider_visible=True,
 rangeselector=dict(
 buttons=list([
 dict(count=15, label='15m', step="minute", stepmode="backward"),
 dict(count=45, label='45m', step="minute", stepmode="backward"),
 dict(count=1, label='1h', step="hour", stepmode="backward"),
 dict(count=6, label='6h', step="hour", stepmode="backward"),
 dict(step="all")
 ]) ))
 graph100JSON=json.dumps(fig100,cls=plotly.utils.PlotlyJSONEncoder)

 fig9 = px.violin(df, y="mileage", x="torque", color="fuel_type", box=True, points="all",hover_data=df.columns , title="Mileage Vs Torque grouped by fuel type",template='plotly_dark', height=700,width=720)
 graph9JSON=json.dumps(fig9,cls=plotly.utils.PlotlyJSONEncoder)

 fig10 = px.scatter_3d(df, x="height", y="width", z="length", color='body_type', title="Dimensional Analysis", height=700,width=720,template='plotly_dark')
 graph10JSON=json.dumps(fig10,cls=plotly.utils.PlotlyJSONEncoder)

 fig88 = px.sunburst(df, path=['type', 'make'] , title="Vehicle Type and Make Composition",template='plotly_dark', height=700,width=720)
 graph88JSON=json.dumps(fig88,cls=plotly.utils.PlotlyJSONEncoder)
 #regression
 df1=df[df['price'] < 1500000]
 df1.head()
 Y = df1.price.values.reshape(-1, 1)
 model = LinearRegression()
 model.fit(Y, df1.mileage)
 x_range = np.linspace(Y.min(), Y.max(), 100)
 y_range = model.predict(x_range.reshape(-1, 1))
 fig111 = px.scatter(df1, x='price', y='mileage', opacity=0.65 , title="Linear Regression : Mileage-Price",template='plotly_dark')
 fig111.add_traces(go.Scatter(x=x_range, y=y_range, name='Regression Fit'))
 fig111.update_layout(height=700,width=720)
 graph111JSON=json.dumps(fig111,cls=plotly.utils.PlotlyJSONEncoder)

 X = df.price.values.reshape(-1, 1)
 model = LinearRegression()
 model.fit(X, df.power)
 x_range = np.linspace(X.min(), X.max(), 100)
 y_range = model.predict(x_range.reshape(-1, 1))
 print(model.predict([[500000]]))
 fig222 = px.scatter(df, x='price', y='power', opacity=0.65 , title="Linear Regression : Power-Price",template='plotly_dark')
 fig222.add_traces(go.Scatter(x=x_range, y=y_range, name='Regression Fit'))
 fig222.update_layout(height=700,width=720)
 graph222JSON=json.dumps(fig222,cls=plotly.utils.PlotlyJSONEncoder)

 df_1 = pd.read_csv('data.csv')
 df_1['date'] = pd.to_datetime(df_1.Date, format="%m/%d/%Y")
 df_1['value']=df_1.Index
 fig_11 = calplot(
         df_1,
         x="date",
         y="value",
         colorscale="spectral",showscale=True,month_lines_width=1,month_lines_color="black",years_title=True,gap=2,total_height=200
         
      )
 fig_11.update_layout(
    xaxis_title="Month",
    yaxis_title="Days",
    legend_title="Preferred Date",
    font=dict(
        family="Arial Bold",size=12,
        color="Black"
    )
      )


 comdata = pd.read_csv("Comm.csv")
 df1=pd.DataFrame(comdata)
 fig101 = px.violin(df1, y="Quantity", x="Make", color="Fuel_Type", box=True, points="all",
          hover_data=df1.columns, title="Sales Data Brand Image",template='plotly_dark',height=600)
 graph101JSON=json.dumps(fig101,cls=plotly.utils.PlotlyJSONEncoder) 

 return render_template('Gp/portfolio-details.html',graph1JSON=graph1JSON,
 graph2JSON=graph2JSON,graph3JSON=graph3JSON,graph4JSON=graph4JSON,graph5JSON=graph5JSON,
 graph6JSON=graph6JSON,graph8JSON=graph8JSON,graph7JSON=graph7JSON,graph9JSON=graph9JSON
 ,graph10JSON=graph10JSON,graph88JSON=graph88JSON,graph99JSON=graph99JSON,graph100JSON=graph100JSON,graph101JSON=graph101JSON
 ,graph111JSON=graph111JSON,graph222JSON=graph222JSON,graph68JSON=graph68JSON,graph69JSON=graph69JSON,graph79JSON=graph79JSON,graph89JSON=graph89JSON)
@app.route('/index',methods = ['POST', 'GET'])
def index():
   df = pd.read_csv('cars.csv')
 #data clean
   df['car'] = df.Make + ' ' + df.Model
   c = ['Make','Model','car','Variant','Body_Type','Fuel_Type','Fuel_System','Type','Drivetrain','Ex-Showroom_Price','Displacement','Cylinders',
    'ARAI_Certified_Mileage','Power','Torque','Fuel_Tank_Capacity','Height','Length','Width','Doors','Seating_Capacity','Wheelbase','Number_of_Airbags']
   df_full = df.copy()
   df['Ex-Showroom_Price'] = df['Ex-Showroom_Price'].str.replace('Rs. ','',regex=False)
   df['Ex-Showroom_Price'] = df['Ex-Showroom_Price'].str.replace(',','',regex=False)
   df['Ex-Showroom_Price'] = df['Ex-Showroom_Price'].astype(int)
   df = df[c]
   df = df[~df.ARAI_Certified_Mileage.isnull()]
   df = df[~df.Make.isnull()]
   df = df[~df.Width.isnull()]
   df = df[~df.Cylinders.isnull()]
   df = df[~df.Wheelbase.isnull()]
   df = df[~df['Fuel_Tank_Capacity'].isnull()]
   df = df[~df['Seating_Capacity'].isnull()]
   df = df[~df['Torque'].isnull()]
   df['Height'] = df['Height'].str.replace(' mm','',regex=False).astype(float)
   df['Length'] = df['Length'].str.replace(' mm','',regex=False).astype(float)
   df['Width'] = df['Width'].str.replace(' mm','',regex=False).astype(float)
   df['Wheelbase'] = df['Wheelbase'].str.replace(' mm','',regex=False).astype(float)
   df['Fuel_Tank_Capacity'] = df['Fuel_Tank_Capacity'].str.replace(' litres','',regex=False).astype(float)
   df['Displacement'] = df['Displacement'].str.replace(' cc','',regex=False)
   df.loc[df.ARAI_Certified_Mileage == '9.8-10.0 km/litre','ARAI_Certified_Mileage'] = '10'
   df.loc[df.ARAI_Certified_Mileage == '10kmpl km/litre','ARAI_Certified_Mileage'] = '10'
   df['ARAI_Certified_Mileage'] = df['ARAI_Certified_Mileage'].str.replace(' km/litre','',regex=False).astype(float)
   df.Number_of_Airbags.fillna(0,inplace= True)
   df['price'] = df['Ex-Showroom_Price']
   df.drop(columns='Ex-Showroom_Price', inplace= True)
   df.price = df.price.astype(int)
   HP = df.Power.str.extract(r'(\d{1,4}).*').astype(int) * 0.98632
   HP = HP.apply(lambda x: round(x,2))
   TQ = df.Torque.str.extract(r'(\d{1,4}).*').astype(int)
   TQ = TQ.apply(lambda x: round(x,2))
   df.Torque = TQ
   df.Power = HP
   df.Doors = df.Doors.astype(int)
   df.Seating_Capacity = df.Seating_Capacity.astype(int)
   df.Number_of_Airbags = df.Number_of_Airbags.astype(int)
   df.Displacement = df.Displacement.astype(int)
   df.Cylinders = df.Cylinders.astype(int)
   df.columns = ['make', 'model','car', 'variant', 'body_type', 'fuel_type', 'fuel_system','type', 'drivetrain', 'displacement', 'cylinders',
            'mileage', 'power', 'torque', 'fuel_tank','height', 'length', 'width', 'doors', 'seats', 'wheelbase','airbags', 'price']
   
   fig11 = px.scatter(df, x="power", y="price", color="body_type", hover_data=['fuel_type'],title="Ptice and Power Relationship",template='plotly_dark')
   graph11JSON=json.dumps(fig11,cls=plotly.utils.PlotlyJSONEncoder)

   rowi=len(df)
   coli=len(df.axes[1])
   toli=rowi*coli

   fig22 = px.histogram(df, x="make", color="fuel_type",template='plotly_dark').update_xaxes(categoryorder='total descending')
   graph22JSON=json.dumps(fig22,cls=plotly.utils.PlotlyJSONEncoder)

   fig33 = px.pie(df,names=df.body_type.value_counts().index,values=df.body_type.value_counts(), width=400, height=400,template='plotly_dark')
   fig33.update_traces(hoverinfo='label+percent', textinfo='percent', textfont_size=20)
   graph33JSON=json.dumps(fig33,cls=plotly.utils.PlotlyJSONEncoder)

   fig44=px.scatter_3d(df,x='power',z='price',y='mileage',color='make',width=550,height=400,template='plotly_dark')
   fig44.update_layout(showlegend=True)
   graph44JSON=json.dumps(fig44,cls=plotly.utils.PlotlyJSONEncoder)

   fig55 = px.sunburst(df, path=['body_type', 'fuel_type','fuel_tank'],width=420, height=420,template='plotly_dark')
   graph55JSON=json.dumps(fig55,cls=plotly.utils.PlotlyJSONEncoder)

   fig66 = px.violin(df, y="mileage", x="torque", color="fuel_type", box=True, points="all",hover_data=df.columns,template='plotly_dark')
   graph66JSON=json.dumps(fig66,cls=plotly.utils.PlotlyJSONEncoder)

   #plot regression 1 
   value4 = request.form.get('vpow')
   value5 = request.form.get('vpri')
   X = df.price.values.reshape(-1, 1)
   model = LinearRegression()
   model.fit(X, df.power)
   x_range = np.linspace(X.min(), X.max(), 100)
   y_range = model.predict(x_range.reshape(-1, 1))
   if value4 != None :
       value6 = int(model.predict([[float(value5)]]))
   elif value4 == None :
      value6= ''
      

   print(model.predict([[500000]]))
   fig_0 = px.scatter(df, x='price', y='power', opacity=0.65,template='plotly_dark')
   fig_0.add_traces(go.Scatter(x=x_range, y=y_range, name='Regression Fit'))
   
   #regression 2
   value1 = request.form.get('vmileage')
   value2 = request.form.get('vprice')
   df1=df[df['price'] < 1500000]
   df1.head()
   Y = df1.price.values.reshape(-1, 1)
   model = LinearRegression()
   model.fit(Y, df1.mileage)
   x_range = np.linspace(Y.min(), Y.max(), 100)
   y_range = model.predict(x_range.reshape(-1, 1))
   if value1 != None :
       value3 = int(model.predict([[float(value2)]]))
   elif value1 == None :
    value3= ''

   
   fig77 = px.scatter_3d(df, x="height", y="width", z="length", color='body_type',template='plotly_dark')
   graph77JSON=json.dumps(fig77,cls=plotly.utils.PlotlyJSONEncoder)

   fig88 = px.sunburst(df, path=['type', 'make'],template='plotly_dark')
   graph88JSON=json.dumps(fig88,cls=plotly.utils.PlotlyJSONEncoder)
   
   df_1 = pd.read_csv('data.csv')
   df_1['date'] = pd.to_datetime(df_1.Date, format="%m/%d/%Y")
   df_1['value']=df_1.Index
   fig_11 = calplot(
         df_1,
         x="date",
         y="value",
         colorscale="spectral",showscale=True,month_lines_width=1,month_lines_color="black",years_title=True,gap=2,total_height=200
         
      )
   fig_11.update_layout(
    xaxis_title="Month",
    yaxis_title="Days",
    legend_title="Preferred Date",
    font=dict(
        family="Arial Bold",size=12,
        color="Black"
    )
      )
   
   graph3JSON=json.dumps(fig_11,cls=plotly.utils.PlotlyJSONEncoder)

   comdata = pd.read_csv("Comm.csv")
   df1=pd.DataFrame(comdata)
   fig101 = px.violin(df1, y="Quantity", x="Make", color="Fuel_Type", box=True, points="all",
          hover_data=df1.columns, title="Sales Data Brand Image",template='plotly_dark')
   graph101JSON=json.dumps(fig101,cls=plotly.utils.PlotlyJSONEncoder)
   return render_template('Gp/index.html',graph3JSON=graph3JSON,
   graph11JSON=graph11JSON,graph22JSON=graph22JSON,graph33JSON=graph33JSON,
   graph44JSON=graph44JSON,graph55JSON=graph55JSON,graph66JSON=graph66JSON
   ,graph77JSON=graph77JSON,graph88JSON=graph88JSON,graph101JSON=graph101JSON, value1=value1,value2=value2,value3=value3,
   value4=value4,value5=value5,value6=value6,rowi=rowi,coli=coli,toli=toli)
   
app.run(host='localhost', port=5000)


 