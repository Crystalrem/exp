import os
import sys
from generator import *
import numpy as np
import subprocess

process = 3
def eval_face(n):
    def run(rps, iteration):
        print('Test rps %s' % rps)
        threads = []
        outputs = []
        procs = []
        for i in range(process):
	    output = 'logs/{}models/sla_{}ms/googlenet{}_rate{}_iter{}.txt'.format(n, sys.argv[2], i+1, rps, iteration)
            cmd = 'python runtest.py 1 50 %s %s %s'%(float(rps) / process, output, iteration)
	    print cmd + '\n'
            proc = subprocess.Popen(cmd, shell = True)
            procs.append(proc)
            outputs.append(output)

        for proc in procs:
            proc.wait()
	aggr_good = 0
 	aggr_total = 0
	for i, output in enumerate(outputs):
            good, total = parse_result(output)
	    percent = float(good) / total
            print('App %s: %.2f%%' % (i+1, percent*100))
            aggr_good += good
            aggr_total += total
        if float(aggr_good) / aggr_total < .99:
            return False
        return True
    def search(rps, step):
        if step <= 1.5:
            for rps in np.arange(rps - 1, rps, 1):
                for i in range(5):
                    good = run(rps, i)
                    if good:
                        break
                if not good:
                     break
            return
        next_step = int(int(step) / 2)
        flag = False
        for i in range(5):
            good = run(rps - step, i)
            if good:
                search(rps, next_step)
                flag = True
                break
        if not flag:
             search(rps - step, next_step)
         
    duration = 60
    datapath = '/datasets/vgg_face/'
    print(datapath)
    print("Number of models: %s" % n)
    #run(rps)

    rps = 1400

    while True:
    	for i in range(4):
    	    good = run(rps, i)
    	    if good:
    	        break
    	if good:
            rps += 256
    	else:
            break

    search(rps, 128)
#    for rps in np.arange(rps-9.5, rps, 0.5):
#        for i in range(5):
#            good = run(rps, i)
#            if good:
#                break
#        if not good:
#            break

def parse_result(fn):
    total, good = 0, 0
    with open(fn) as f:
        for line in f:
	    total += 1
	    if float(line.strip()) < float(sys.argv[2]):
	        good += 1
    return good, total

def main():
    FORMAT = "[%(asctime)-15s %(levelname)s] %(message)s"
    logging.basicConfig(format=FORMAT)
    logging.getLogger().setLevel(logging.INFO)
    eval_face(int(sys.argv[1]))

if __name__ == "__main__":
    main()
