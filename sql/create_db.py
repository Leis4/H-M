import os
import duckdb

SQL_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SQL_DIR)
DB_PATH = os.path.join(SQL_DIR, 'hm_market.db')
SQL_FILE = os.path.join(SQL_DIR, 'Make_DB.sql')

os.chdir(PROJECT_ROOT)
con = duckdb.connect(DB_PATH)

with open(SQL_FILE, 'r', encoding='utf-8') as file:
    sql_script = file.read()

con.execute(sql_script)
con.close()
