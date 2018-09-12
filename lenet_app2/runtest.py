import os
import sys
from generator import *


def run_test(datapath, rate, duration, output, app_id):
    dataset = Dataset(datapath)
    gen = Generator(dataset, output, app_id)
    gen.run(rate, duration)
    gen.output_latencies(output)

def eval_face(n):
    rps = float(sys.argv[3])
    output = sys.argv[4]
    iteration = int(sys.argv[5])
    def run(rps, iteration):
        print('Test rps %s' % rps)
        threads = []
        outputs = []
        for i in range(n):
            t = Thread(target=run_test, args=(datapath, rps, duration, output, i+1))
	    t.daemon = True
            threads.append(t)
            outputs.append(output)
        for t in threads:
            t.start()
        for t in threads:
            t.join()
 
         
    duration = 60
    datapath = '/datasets/vgg_face/'
    print(datapath)
    print("Number of models: %s" % n)
    run(rps, iteration)

def main():
    FORMAT = "[%(asctime)-15s %(levelname)s] %(message)s"
    logging.basicConfig(format=FORMAT)
    logging.getLogger().setLevel(logging.INFO)
    eval_face(int(sys.argv[1]))

if __name__ == "__main__":
    main()
