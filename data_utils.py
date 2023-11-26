#data_utils.py
#Helper functions to get data 

import pandas as pd
import numpy as np
from mchs_ds.pyrs import RsConnection #database connection
from pathlib import Path

def get_data(
        conn,
        logger, 
        dsn,
        window_start,
        window_stop
    ):
 
    logger.info(f"Loading data for window start date {window_start}:")    

    sql_folder="sql"
    query = Path(sql_folder, "get_features.sql").read_text()
    query = query.replace("${window_start}$", f"'{window_start}'") 
    query = query.replace("${window_stop}$", f"'{window_stop}'") 
    conn.execute_sql_non_query(query)


def get_features(
        logger, 
        dsn,
        window_start,
        window_stop,
        cat_features,
        parse_dates=[],
        
    ):
    
    conn = RsConnection(dsn)
    get_data(conn, logger, dsn, window_start, window_stop)
        
    query = 'SELECT * FROM shpretention_features'
    features = conn.execute_sql_query(
             query,
             # index_col=["shp_member_id"], 
             parse_dates=parse_dates, 
             fix_null_ints=True, 
             parse_strings=cat_features
        )
           
    features.replace(np.nan, None, inplace=True)

    return features

