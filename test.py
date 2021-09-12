from SqlTest import UsingMysql


# 查询数据库数量
def get_count(cursor):
    cursor.execute("select count(time) as total from ScreenSelect")

    # 使用 fetchone() 方法获取单条数据.
    data = cursor.fetchone()

    print("-- 当前数量: %d " % data['total'])


# 新增单条记录
def create_one(params):
    with UsingMysql(log_time=True) as um:
        sql = "insert into ScreenSelect(time,pic,t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12,t13,t14,t15,t16,t17) values(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
        """
        测试数据
        params = (
            '121335', '131331131', '121212', '1212121', '121212', '1212121', '121212', '1212121', '121212', '1212121',
            '121212', '1212121', '121212', '1212121', '121212', '1212121', '121212', '1212121', '121212')
            """
        um.cursor.execute(sql, params)

        # 查看结果
        get_count(um.cursor)


if __name__ == '__main__':
    params = (
        '1213366666', '131331131', '121212', '1212121', '121212', '1212121', '121212', '1212121', '121212', '1212121',
        '121212', '1212121', '121212', '1212121', '121212', '1212121', '121212', '1212121', '121212')
    create_one(params)#插入数据
    # check_it()
