import os
import logging
from PyQt5.QtSql import QSqlDatabase, QSqlQuery
from PyQt5.QtCore import QMutex, QMutexLocker

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("DatabaseManager")


class DatabaseSingleton:
    """
    数据库操作单例类

    功能：
    - 线程安全的数据库连接管理
    - 支持 SQLite 和 MySQL
    - 事务支持
    - 参数化查询
    - 错误处理和日志记录
    """

    _instance = None
    _mutex = QMutex()

    def __new__(cls, *args, **kwargs):
        """实现单例模式"""
        with QMutexLocker(cls._mutex):
            if not cls._instance:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def __init__(self, db_type='SQLITE', **kwargs):
        """初始化数据库连接"""
        if hasattr(self, '_initialized') and self._initialized:
            return

        self.db_type = db_type.upper()
        self.connection_name = "database_connection"
        self.connection_params = kwargs
        self._transaction_active = False

        # 确保只初始化一次
        self._initialized = True
        logger.info("Database manager initialized")

    def connect(self):
        """创建并打开数据库连接"""
        # 如果连接已存在，先关闭
        if QSqlDatabase.contains(self.connection_name):
            QSqlDatabase.removeDatabase(self.connection_name)

        # 创建数据库连接
        db = QSqlDatabase.addDatabase(self._get_db_driver(), self.connection_name)

        # 设置连接参数
        if self.db_type == 'SQLITE':
            db_path = self.connection_params.get('db_path', 'default_database.sqlite')
            # 如果目录不存在则创建
            db_dir = os.path.dirname(db_path)
            if db_dir and not os.path.exists(db_dir):
                os.makedirs(db_dir)
            db.setDatabaseName(db_path)
            logger.info(f"Connecting to SQLite database: {db_path}")

        elif self.db_type == 'MYSQL':
            db.setHostName(self.connection_params.get('host', 'localhost'))
            db.setPort(self.connection_params.get('port', 3306))
            db.setDatabaseName(self.connection_params.get('db_name', ''))
            db.setUserName(self.connection_params.get('user', ''))
            db.setPassword(self.connection_params.get('password', ''))
            logger.info(f"Connecting to MySQL database: {self.connection_params.get('db_name')}")

        else:
            logger.error(f"Unsupported database type: {self.db_type}")
            return False

        # 打开数据库连接
        if not db.open():
            last_error = db.lastError()
            logger.error(f"Database connection error: {last_error.text()}")
            return False

        logger.info("Database connected successfully")
        return True

    def _get_db_driver(self):
        """获取数据库驱动名称"""
        drivers = {
            'SQLITE': 'QSQLITE',
            'MYSQL': 'QMYSQL'
        }
        return drivers.get(self.db_type, 'QSQLITE')

    def execute_query(self, query_str, params=None, fetch_all=True):
        """
        执行查询操作

        参数:
            query_str (str): SQL查询字符串
            params (dict|list): 查询参数
            fetch_all (bool): 是否获取所有结果

        返回:
            list: 查询结果列表
        """
        if not self._is_connected():
            return []

        query = QSqlQuery(self._get_database())
        query.prepare(query_str)

        # 绑定参数
        if params:
            if isinstance(params, dict):
                for key, value in params.items():
                    query.bindValue(f":{key}", value)
            elif isinstance(params, list) or isinstance(params, tuple):
                for i, value in enumerate(params):
                    query.bindValue(i, value)

        if not query.exec():
            self._log_query_error(query)
            return []

        results = []
        while query.next():
            record = query.record()
            row = {}
            for i in range(record.count()):
                field_name = record.fieldName(i)
                row[field_name] = record.value(i)
            results.append(row)

        # 如果不获取所有结果，只返回第一条
        if not fetch_all and results:
            return results[0]

        return results

    def execute_command(self, command_str, params=None, return_id=False):
        """
        执行数据库命令（插入、更新、删除等）

        参数:
            command_str (str): SQL命令字符串
            params (dict|list): 命令参数
            return_id (bool): 是否返回最后插入的ID

        返回:
            int: 受影响的行数或最后插入的ID
        """
        if not self._is_connected():
            return 0

        query = QSqlQuery(self._get_database())
        query.prepare(command_str)

        # 绑定参数
        if params:
            if isinstance(params, dict):
                for key, value in params.items():
                    query.bindValue(f":{key}", value)
            elif isinstance(params, list) or isinstance(params, tuple):
                for i, value in enumerate(params):
                    query.bindValue(i, value)

        if not query.exec():
            self._log_query_error(query)
            return 0

        # 返回受影响的行数
        affected_rows = query.numRowsAffected()

        if return_id:
            # 获取最后插入的ID
            if self.db_type == 'SQLITE':
                query.exec("SELECT last_insert_rowid()")
            else:  # MySQL
                query.exec("SELECT LAST_INSERT_ID()")

            if query.next():
                return query.value(0)
            return 0

        return affected_rows

    def begin_transaction(self):
        """开始事务"""
        if not self._is_connected():
            return False

        db = self._get_database()
        if db.transaction():
            self._transaction_active = True
            logger.info("Transaction started")
            return True
        return False

    def commit_transaction(self):
        """提交事务"""
        if not self._transaction_active or not self._is_connected():
            return False

        db = self._get_database()
        if db.commit():
            self._transaction_active = False
            logger.info("Transaction committed")
            return True
        return False

    def rollback_transaction(self):
        """回滚事务"""
        if not self._transaction_active or not self._is_connected():
            return False

        db = self._get_database()
        if db.rollback():
            self._transaction_active = False
            logger.info("Transaction rolled back")
            return True
        return False

    def table_exists(self, table_name):
        """检查表是否存在"""
        if self.db_type == 'SQLITE':
            query = "SELECT name FROM sqlite_master WHERE type='table' AND name=?"
            result = self.execute_query(query, [table_name])
            return bool(result)
        else:  # MySQL
            query = "SHOW TABLES LIKE ?"
            result = self.execute_query(query, [table_name])
            return bool(result)

    def close(self):
        """关闭数据库连接"""
        if self._is_connected():
            db = self._get_database()
            db.close()
            QSqlDatabase.removeDatabase(self.connection_name)
            logger.info("Database connection closed")
            return True
        return False

    def _is_connected(self):
        """检查是否已连接数据库"""
        if QSqlDatabase.contains(self.connection_name):
            db = QSqlDatabase.database(self.connection_name, False)
            return db.isOpen()
        return False

    def _get_database(self):
        """获取数据库对象"""
        if self._is_connected():
            return QSqlDatabase.database(self.connection_name)
        return None

    def _log_query_error(self, query):
        """记录查询错误日志"""
        error = query.lastError()
        logger.error(f"Query error: {error.text()}")
        logger.error(f"Executed query: {query.executedQuery()}")

        # 记录绑定参数
        bound_values = query.boundValues()
        logger.error(f"Bound values: {bound_values}")

    def __del__(self):
        """析构函数，自动关闭连接"""
        self.close()