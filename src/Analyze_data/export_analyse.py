#=======IMPORT DATA=======
import os
import duckdb
import pandas as pd
from ydata_profiling import ProfileReport
#====================================


#=======PATH=======
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DB_PATH = os.path.join(BASE_DIR, 'sql', 'hm_market.db')
#====================================

#=======LOADING=======
def load_features_from_db():
    con = duckdb.connect(DB_PATH, read_only=True)
    query = "SELECT * FROM final_features"
    df = con.query(query).to_df()
    con.close()
    data = df
    return data
#====================================

# if __name__ == "__main__":
#     data = load_features_from_db()
#     profile = ProfileReport(data, title="H&M EDA Report", explorative=True)
#     profile.to_file("hm_data_analysis.html")
#
#     path_to_save = os.path.join(BASE_DIR, 'data', 'report.csv')
#     data.to_csv(path_to_save, index=False)



