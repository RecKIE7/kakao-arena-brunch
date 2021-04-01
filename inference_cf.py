import os
import sys
sys.path.append('..')
sys.path.append('../../')
import config as conf


if not os.path.exists(conf.submission + '/recommend_1.txt'):
  print('Cannot find submission/recommend_1.txt')
  sys.exit()

t_users = {}
user_file = conf.data_root + "predict/dev.users"
with open(user_file, "r") as fp:
  for line in fp:
    viewer_id = line.strip()
    t_users[viewer_id] = 1

inferences = {}
with open(conf.submission + '/recommend_1.txt', 'r') as fp:
  for line in fp:
    tokens = line.strip().split()
    inferences[tokens[0]] = tokens[1:]

with open('recommend.txt', 'w') as fp:
  for user in t_users:
    if user in inferences:
      recs = inferences[user]
    else:
      recs = []

    for i in range(len(recs), 100):
      recs.append('@random_' + str(i+1))
  
    fp.write(user + ' ' + ' '.join(recs) + '\n')
