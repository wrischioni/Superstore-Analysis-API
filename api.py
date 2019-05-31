from datetime import datetime
import fbprophet
import numpy as np
import pandas as pd
import json
import os
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
from api_predict import *


app = Flask(__name__)
CORS(app)

body_var = [
    {'date_time': datetime.now().date()}
]

@app.route("/superstore",methods=['GET', 'POST'])
def default():

  if request.method == 'POST': 
  	# gettig user input variables
    days_in_future = request.form['days']	
    dt_i = body_var[0]['date_time']
    dt_f = np.datetime64(dt_i) + np.timedelta64(days_in_future, 'D')
    # getting date structures
    dt_f_processed = pd.to_datetime(dt_f)
    dt_day = dt_f_processed.day
    dt_week = dt_f_processed.week
    dt_month = dt_f_processed.month
    dt_year = dt_f_processed.year
    # getting product structures
    sku = request.form['sku']

     # model predictor
    megazord = sales_predictor()

    # getting product cluster
    cluster = megazord.get_cluster(sku)

    # getting product sales perc
    '''
    0 - daily
    1 - weekly
    2 - monthly
    '''
    share =  megazord.get_product_share(cluster, sku)

    # ts prediction
    ts_prediction = []
    t_to_predict = ['d', 'w', 'm']
    for tdelta in t_to_predict:
      ts_result = megazord.ts_predict(day=dt_f, cluster=cluster, timedelta=tdelta)
      ts_input = [ts_result['yhat'],
                  ts_result['yhat_lower'],
                  ts_result['yhat_upper']]
      ts_prediction.append(ts_input[:].copy())

    # ensemble prediction
    sales_prediction = megazord.ensemble_predict(cluster, ts_prediction[0][0], 
                                                 ts_prediction[0][1], 
                                                 ts_prediction[0][2], 
                                                 dt_day,
                                                 dt_week,
                                                 dt_month,
                                                 dt_year,
                                                 ts_prediction[1][0], 
                                                 ts_prediction[1][1], 
                                                 ts_prediction[1][2], 
                                                 ts_prediction[2][0], 
                                                 ts_prediction[2][1], 
                                                 ts_prediction[2][2], 
                                                 share[0],
                                                 share[1],
                                                 share[2])
    sales_prediction = np.around(sales_prediction, 3)
    
    return render_template('return.html', body_var=body_var, days=days_in_future, sku=sku, sales=sales_prediction)

  if request.method == 'GET':
    return render_template('body.html', body_var=body_var)

if __name__ == "__main__":
    app.run(debug=True)