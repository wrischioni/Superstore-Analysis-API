import pickle
import pandas as pd
import numpy as np

class sales_predictor():
  def __init__(self):
    pass

  @staticmethod  
  def get_cluster(sku):
    df_dim_cluster = pd.read_csv('./product_cluster.csv', usecols=['Product_ID', 'Cluster'])
    sku_cluster = df_dim_cluster[df_dim_cluster.Product_ID == sku]['Cluster']
    return int(sku_cluster.item())

  # time series model structure  
  def ts_deserialize(self, cluster, timedelta):
    # de-serialize ts_model.pkl file into an object called ts_model using pickle
    if timedelta == 'd':
      ts_pkl = f'ts{cluster}_pkl_D.pkl'
      with open(ts_pkl, 'rb') as handle:
        ts_model = pickle.load(handle)
        return ts_model

    elif timedelta == 'w':
      ts_pkl = f'ts{cluster}_pkl_W.pkl'
      with open(ts_pkl, 'rb') as handle:
        ts_model = pickle.load(handle)
        return ts_model

    elif timedelta == 'm':
      ts_pkl = f'ts{cluster}_pkl_M.pkl'
      with open(ts_pkl, 'rb') as handle:
        ts_model = pickle.load(handle)
        return ts_model

  def ts_predict(self, day, cluster, timedelta):
    day_proc = pd.DataFrame(pd.Series(day))
    day_proc.columns = ['ds']
    ts_model = self.ts_deserialize(cluster, timedelta)
    return ts_model.predict(day_proc)  

  # ensemble model variables pre-processing
  @staticmethod
  def get_product_share(cluster, sku):

    cols = ['Product_ID',
            'Perc_Qtd_Daily',
            'Perc_Qtd_Weekly',
            'Perc_Qtd_Monthly']

    df_dim_perc = pd.read_csv(f'./cluster{cluster}_share.csv', usecols=cols)
    mask = (df_dim_perc.Product_ID == str(sku))
    sku_daily_qty = df_dim_perc[mask]['Perc_Qtd_Daily'].item()
    sku_weekly_qty = df_dim_perc[mask]['Perc_Qtd_Weekly'].item()
    sku_monthly_qty = df_dim_perc[mask]['Perc_Qtd_Monthly'].item()
    return (sku_daily_qty, sku_weekly_qty, sku_monthly_qty)

  # ensemble model structure  
  def ensemble_deserialize(self, cluster):
    # de-serialize ensemble_model.pkl file into an object called ensemble_model using pickle
    ensemble_pkl = f'ensemble{cluster}_pkl.pkl'
    with open(ensemble_pkl, 'rb') as handle:
      ensemble_model = pickle.load(handle)
      return ensemble_model

  def ensemble_predict(self, cluster, d, w, m, y, dp, dq, dql, dqu, wp, wq, wql, wqu, mp, mq, mql, mqu):
    ensemble_model = self.ensemble_deserialize(cluster)
    return ensemble_model.predict(np.array([[
                                    d, w, m, y, dp, dq, dql, dqu, wp, wq, wql, wqu, wp, wq, wql, wqu
                                    ]]))
