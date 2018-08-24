import pandas as pd
import numpy as np
import os


# 用于生成datatype_dataset.csv文件
# 即sessions

class generate(object):

    def __init__(self, dataPath, sessPath):
        self._data = pd.read_csv(dataPath)
        self.sessPath = sessPath

    def stati_data(self):
        print('总数据量:', len(self._data))
        print('总session数:', len(self._data.drop_duplicates(['use_ID', 'time'])))
        print('平均session长度:', len(self._data) / len(self._data.drop_duplicates(['use_ID', 'time'])))
        print('总user数:', len(self._data.drop_duplicates('use_ID')))
        print('平均每个用户拥有的session个数:',
              len(self._data.drop_duplicates(['use_ID', 'time'])) / len(self._data.drop_duplicates('use_ID')))
        print('总item数:', len(self._data.drop_duplicates('ite_ID')))
        print('数据集时间跨度：', min(self._data.time), '~', max(self._data.time))

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
        self._data.to_csv('../data/middle_data.csv', index=False)
        print('user_count', user_count)
        print('item_count', item_count)

    # 按照实验设计，test的session是从数据集的最后一个月随机抽取百分之二十的session得到的
    # TallM中使用的test集合是每个用户最后一个session
    def generate_train_test_session(self):
        print('statistics ... ')
        self.stati_data()  # 统计数据集
        print('encode ... ')
        self.reform_u_i_id()  # 重新编码user和item
        print('generate train and test session ... ')
        self._data = pd.read_csv('../data/middle_data.csv')
        self._train_data = self._data[self._data.time < 20100931].reset_index(drop=True)    # 重设索引
        self._test_data = self._data[self._data.time > 20100930].reset_index(drop=True)
        os.remove('../data/middle_data.csv')
        session_train_path = self.sessPath + '_train_dataset.csv'
        session_test_path = self.sessPath + '_test_dataset.csv'
        if os.path.exists(session_train_path):
            os.remove(session_train_path)

        if os.path.exists(session_test_path):
            os.remove(session_test_path)

        # for train session
        # 要考虑最后一个session，目前循环中没有考虑最后一个session
        with open(session_train_path, 'a') as session_train_file:
            user_num = len(self._data['use_ID'].drop_duplicates())
            # users = len(self._train_data['use_ID'].drop_duplicates())
            item_num = len(self._data['ite_ID'].drop_duplicates())
            session_train_file.write(str(user_num) + ',' + str(item_num) + '\n')
            last_userid = self._train_data.at[0, 'use_ID']
            last_time = self._train_data.at[0, 'time']
            session = str(last_userid) + ',' + str(self._train_data.at[0, 'ite_ID'])
            for i in range(1, len(self._train_data)):
                # 文件使用降序打开
                # 最终session的格式为user_id,item_id:item_id...@item_id:item_id...@...
                userid = self._train_data.at[i, 'use_ID']
                itemid = self._train_data.at[i, 'ite_ID']
                time = self._train_data.at[i, 'time']
                if userid == last_userid and time == last_time:
                    # 需要将session写入到文件中，然后开始
                    session += ":" + str(itemid)
                elif userid != last_userid:
                    session_train_file.write(session + '\n')
                    last_userid = userid
                    last_time = time
                    session = str(userid) + ',' + str(itemid)
                else:
                    session += '@' + str(itemid)
                    last_time = time

        # for test session
        # 要考虑最后一个session，目前循环中没有考虑最后一个session
        # 先构建session，一个session一行，然后再随机抽取20%
        with open(session_test_path, 'a') as session_test_file:
            last_userid = self._test_data.at[0, 'use_ID']
            last_time = self._test_data.at[0, 'time']
            session = str(last_userid) + ',' + str(self._test_data.at[0, 'ite_ID'])
            for i in range(1, len(self._test_data)):
                # 最终session的格式为user_id,item_id:item_id...
                userid = self._test_data.at[i, 'use_ID']
                itemid = self._test_data.at[i, 'ite_ID']
                time = self._test_data.at[i, 'time']
                if userid == last_userid and time == last_time:
                    # 需要将session写入到文件中，然后开始
                    session += ":" + str(itemid)
                elif userid != last_userid:
                    session_test_file.write(session + '\n')
                    last_userid = userid
                    last_time = time
                    session = str(userid) + ',' + str(itemid)
                else:
                    session_test_file.write(session + '\n')
                    last_time = time
                    session = str(userid) + ',' + str(itemid)


if __name__ == '__main__':
    datatype = ['tallM', 'gowalla']
    dataPath = '../data/' + datatype[1] + '_data.csv'
    sessPath = '../data/' + datatype[1]
    object = generate(dataPath, sessPath)
    # object.stati_data()
    object.generate_train_test_session()
