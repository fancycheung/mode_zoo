#!/usr/bin/env python
#coding=UTF-8
'''
    @desc: 数据访问。[!]每次调用重连数据库, 该脚本不适合频繁调用
'''
import sys
import MySQLdb
from MySQLdb.cursors import DictCursor

# MySQL 连接信息
MYSQL_HOST_DEFAULT = 'NULL'
MYSQL_PORT_DEFAULT = 3306
MYSQL_USER_DEFAULT = 'reader'
MYSQL_PWD_DEFAULT = 'miaoji1109'
MYSQL_DB_DEFAULT = 'NULL'

def GetConnection(host, db, user=MYSQL_USER_DEFAULT, passwd=MYSQL_PWD_DEFAULT, port = 3306, charset="utf8"):
    conn = MySQLdb.connect(host=host, user=user, passwd=passwd, db=db, charset=charset)
    return conn

def ExecuteSQL(host, db, sql, args = None):
    '''
        执行SQL语句, 正常执行返回影响的行数，出错返回Flase 
    '''
    ret = 0
    try:
        conn = GetConnection(host, db)
        cur = conn.cursor()

        ret = cur.execute(sql, args)
        conn.commit()
    except MySQLdb.Error, e:
        print 'ExecuteSQL Err: %s'%str(e)
        return None
    finally:
        cur.close()
        conn.close()

    return ret

def ExecuteSQLs(host, db, sql, args = None):
    '''
        执行多条SQL语句, 正常执行返回影响的行数，出错返回Flase 
    '''
    ret = 0
    try:
        conn = GetConnection(host, db)
        cur = conn.cursor()

        ret = cur.executemany(sql, args)
        conn.commit()
    except MySQLdb.Error, e:
        print 'ExecuteSQLs Err: %s'%str(e)
        return None
    finally:
        cur.close()
        conn.close()

    return ret

def QueryBySQL(host, db, sql, args = None, size = None):
    '''
        通过sql查询数据库，正常返回查询结果，否则返回None
    '''
    results = []
    try:
        conn = GetConnection(host, db)
        cur = conn.cursor(cursorclass = DictCursor)
        
        cur.execute(sql, args)
        rs = cur.fetchall()
        for row in rs : 
            results.append(row)
    except MySQLdb.Error, e:
        print 'QueryBySQL Err: %s'%str(e)
        return None
    finally:
        cur.close()
        conn.close()

    return results

if __name__ == "__main__":
    pass
