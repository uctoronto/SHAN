import pandas as pd
import os


# 用于生成datatype_data.csv文件

class Dataset(object):

    def __init__(self, source_file, data_file):
        self._source_file = source_file
        self._data_file = data_file

    def item_appear(self, appear_time, ds):
        dm = ds[['use_ID', 'ite_ID']].drop_duplicates()
        da = dm.groupby(by=['ite_ID'], as_index=False)['ite_ID'].agg({'cnt': 'count'})
        ite_list = list(da[da['cnt'] >= appear_time]['ite_ID'])
        ds = ds[ds['ite_ID'].isin(ite_list)]
        return ds

    def session_not_single(self, ds):
        ds = ds[ds.duplicated(['use_ID', 'time'], keep=False)]
        return ds

    def user_have_more_session(self, ds, user_sessin):
        dm = ds.drop_duplicates(['use_ID', 'time'])
        da = dm.groupby(by=['use_ID'], as_index=False)['use_ID'].agg({'cnt': 'count'})
        use_list = list(da[da['cnt'] >= user_sessin]['use_ID'])
        ds = ds[ds['use_ID'].isin(use_list)]
        return ds

    def first_step_clean_data(self, months=7, appear_time=20, user_sessin=3):
        data = pd.read_csv(self._source_file, names=['use_ID', 'time', 'latitude', 'longitude', 'ite_ID'], sep='\t')
        ds = data.drop(['latitude', 'longitude'], axis=1)
        s = ds.time
        ds.drop('time', axis=1, inplace=True)
        ds.insert(2, 'time', s)
        ds.to_csv('../data/middle.csv', index=None, header=False)
        f1 = open('../data/middle.csv', 'r')
        path = '../data/gowalla.csv'
        if os.path.exists(path):
            os.remove(path)
        with open(path, 'a') as f2:
            while True:
                lines = f1.readlines(1000)
                if not lines:
                    break
                for line in lines:
                    line = line.strip().split(',')
                    line[2] = line[2].split('T')[0].replace('-', '')
                    str_line = ','.join(str(x) for x in line)
                    f2.write(str_line + '\n')
        os.remove('../data/middle.csv')
        ds = pd.read_csv(path, names=['use_ID', 'ite_ID', 'time'])
        os.remove(path)
        # print('max_time:', max(data.time))  # max_time: 20101023
        # print('min_time:', min(data.time))  # min_time: 20090204
        ds = ds[ds['time'] > 20100331]
        # 去除重复的数据，如果重复在后面的session中会相同的item
        ds = ds.drop_duplicates()
        ds = ds.sort_values(by=['use_ID', 'time'])

        # 统计每个ite_ID被多少个不同的user购买，只保留被至少appear_time个用户购买过的item
        ds = self.item_appear(appear_time, ds)

        # 对每个用户的消费行为，如果一个时间点只有一个item的数据，则去除，即只保留重复数据
        ds = self.session_not_single(ds)

        # 对于每个用户，如果只存在一个session，则去除
        ds = self.user_have_more_session(ds,user_sessin)

        # ds = self.item_appear(appear_time, ds)
        ds.to_csv(self._data_file, index=False)


datatype = 'gowalla'
source_file = '../data/' + datatype + '.txt'
data_file = '../data/' + datatype + '_data.csv'
months = 7
appear_time = 20
user_sessin = 3
data = Dataset(source_file, data_file)
data.first_step_clean_data(months, appear_time, user_sessin)
