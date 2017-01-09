import sys
import os
import matplotlib.pyplot as plt
import numpy as np
path = sys.argv[1]
cnt = 0

in_files = ['useast1a.txt','useast1c.txt','useast1d.txt','useast1e.txt']
train_files = ['1a-train.txt','1c-train.txt','1d-train.txt','1e-train.txt']
test_files = ['1a-test.txt','1c-test.txt','1d-test.txt','1e-test.txt']

n_train = 30000

for i in range(len(in_files)):
    with open(os.path.join(path,in_files[i])) as fin:
        with open(os.path.join(path, train_files[i]), 'w') as fout1:
            with open(os.path.join(path, test_files[i]), 'w') as fout2:
                cnt = 0
                for line in fin:
                    cnt += 1
                    if cnt==1:
                        continue
                    if cnt <= n_train:
                        fout1.writelines(line)
                    else:
                        fout2.writelines(line)
