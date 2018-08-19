import pandas as pd
import numpy as np
import os


class generate(object):

    def __init__(self, object):
        self._data = pd.read_csv(object)

    def stati_data(self):
        print('总数据量:', len(self._data))
        print('总session数:', len(self._data.drop_duplicates(['use_ID', 'time'])))
        print('平均session长度:', len(self._data) / len(self._data.drop_duplicates(['use_ID', 'time'])))
        print('总user数:', len(self._data.drop_duplicates('use_ID')))
        print('平均每个用户拥有的session个数:',
              len(self._data.drop_duplicates(['use_ID', 'time'])) / len(self._data.drop_duplicates('use_ID')))
        print('总item数:', len(self._data.drop_duplicates('ite_ID')))

    def reform_u_i_id(self):
        # 将数据中的item和user重新编号，然后再生成session
        user_to_id = {}
        item_to_id = {}
        # 对user进行重新编号
        user_count = 0
        item_count = 0
        for i in range(len(self._data)):
            # 对user 和 item同时进行重新编号
            u_id = self._data.at[i, 'use_ID']
            i_id = self._data.at[i, 'ite_ID']
            if u_id in user_to_id.keys():
                self._data.at[i, 'use_ID'] = user_to_id[u_id]
            else:
                user_to_id[u_id] = user_count
                self._data.at[i, 'use_ID'] = user_count
                user_count += 1
            if i_id in item_to_id.keys():
                self._data.at[i, 'ite_ID'] = item_to_id[i_id]
            else:
                item_to_id[i_id] = item_count
                self._data.at[i, 'ite_ID'] = item_count
                item_count += 1
        self._data.to_csv('../data/data.csv', index=False)
        print('user_count', user_count)
        print('item_count', item_count)

    def generate_session(self):
        session_path = '../data/tallM_dataset.csv'
        if os.path.exists(session_path):
            os.remove(session_path)
        session_file = open('../data/tallM_dataset.csv', 'a')
        # 这里最好使用numpy的格式，最后也按照这样的格式进行保存


        user_num = len(self._data['use_ID'].drop_duplicates())
        item_num = len(self._data['ite_ID'].drop_duplicates())
        session_file.write(str(user_num) + ',' + str(item_num) + '\n')
        last_userid = self._data.at[0, 'use_ID']
        last_time = self._data.at[0, 'time']
        session = str(last_userid) + ',' + str(self._data.at[0, 'ite_ID'])
        for i in range(1, len(self._data)):
            # 文件使用降序打开
            # 最终session的格式为user_id,item_id:item_id...@item_id:item_id...@...
            userid = self._data.at[i, 'use_ID']
            itemid = self._data.at[i, 'ite_ID']
            time = self._data.at[i, 'time']
            if userid == last_userid and time == last_time:
                # 需要将session写入到文件中，然后开始
                session += ":" + str(itemid)
            elif userid != last_userid:
                session_file.write(session + '\n')
                last_userid = userid
                last_time = time
                session = str(userid) + ',' + str(itemid)
            else:
                session += '@' + str(itemid)
                last_time = time


if __name__ == '__main__':
    dataPath = '../data/data.csv'
    object = generate(dataPath)
    object.stati_data()
    # object.reform_u_i_id()
    object.generate_session()
