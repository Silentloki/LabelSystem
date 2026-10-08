from Utils.DataManger import DatabaseSingleton
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
db_path = os.path.join(BASE_DIR, 'AI.db')
sqlite_db = DatabaseSingleton(db_type='SQLITE', db_path=db_path)
#sqlite_db = DatabaseSingleton(db_type='SQLITE', db_path='AI.db')
sqlite_db.connect()
create_table_sql = """
        CREATE TABLE IF NOT EXISTS projects_table (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,          -- 唯一工程名
                    project_type TEXT DEFAULT '检测',   -- 工程类型（默认检测）
                    description TEXT DEFAULT '无',      -- 工程描述（默认无）
                    created_at TEXT NOT NULL            -- 工程创建时间
                    )
   """
sqlite_db.execute_command(create_table_sql)