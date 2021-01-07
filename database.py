# -*- coding: utf-8 -*-
import os
import random

import six
import fire
import mmh3
import tqdm

from util import iterate_data_files


def groupby(from_dtm, to_dtm, tmp_dir, out_path, num_chunks=10):
    from_dtm, to_dtm = map(str, [from_dtm, to_dtm])
    fouts = {idx: open(os.path.join(tmp_dir, str(idx)), 'w')
             for idx in range(num_chunks)}
    files = sorted([path for path, _ in iterate_data_files(from_dtm, to_dtm)])
    for path in tqdm.tqdm(files, mininterval=1):
        for line in open(path):
            user = line.strip().split()[0]
            chunk_index = mmh3.hash(user, 17)   % num_chunks # 각 10개의 버켓에 유저 할당
            fouts[chunk_index].write(line)

    map(lambda x: x.close(), fouts.values())
    with open(out_path, 'w') as fout:
        for chunk_idx in fouts.keys():
            _groupby = {}
            chunk_path = os.path.join(tmp_dir, str(chunk_idx))
            for line in open(chunk_path):
                tkns = line.strip().split()
                userid, seen = tkns[0], tkns[1:] # 각 유저가 본 것들
                _groupby.setdefault(userid, []).extend(seen) # _groupby 딕셔너리에 넣어줌
            os.remove(chunk_path) # temp file 삭제

            # 유저별 본 글을 다시 써주기
            for userid, seen in six.iteritems(_groupby): # 다시 _groupby 딕셔너리 불러와서 파일에 쓰기
                fout.write('%s %s\n' % (userid, ' '.join(seen)))

        
def sample_users(data_path, out_path, num_users):
    users = [data.strip().split()[0] for data in open(data_path)]
    random.shuffle(users)
    users = users[:num_users]
    with open(out_path, 'w') as fout:
        fout.write('\n'.join(users))


if __name__ == '__main__':
    fire.Fire({'groupby': groupby,
               'sample_users': sample_users})
