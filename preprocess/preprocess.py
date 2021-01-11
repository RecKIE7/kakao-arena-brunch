import os
import pandas as pd
import json

path = os.path.dirname( os.path.abspath( __file__ ))

data_path = path  + '/../res/'
output_path = path + '/../data/'

read_path = data_path + 'read/'
predict_path = data_path + "predict/"

read_file_list = os.listdir(read_path)
exclude_file_list = ['.2019010120_2019010121.un~']
read_df_list = []
for f in read_file_list:
    file_name = os.path.basename(f)
    if file_name in exclude_file_list:
        print(file_name)
    else:
        df_temp = pd.read_csv(read_path+f, header=None, names=['raw'])
        df_temp['dt'] = file_name[:8] # 읽은 날짜
        df_temp['hr'] = file_name[8:10] # 읽은 시간
        df_temp['user_id'] = df_temp['raw'].str.split(' ').str[0] # 유저 id
        df_temp['article_id'] = df_temp['raw'].str.split(' ').str[1:].str.join(' ').str.strip() # 해당 유저가 읽은 글 id
        read_df_list.append(df_temp)      
read = pd.concat(read_df_list)


dev = pd.read_csv(predict_path+'dev.users', names=['id'])
test = pd.read_csv(predict_path+'test.users', names=['id'])

dev.to_csv(output_path + 'dev.csv', index=False)
print("dev preprocessing complete")
test.to_csv(output_path + 'test.csv', index=False)
print("test preprocessing complete")
read.to_csv(output_path + 'read.csv', index=False)
print("read preprocessing complete")
