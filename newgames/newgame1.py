import cv2
import os 
import sys
import logging 
import numpy as np 
import random
from multiprocessing import Queue
from threading import Thread
import time
from datetime import datetime
from grpc.beta import implementations
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
sch = [] 
duration = 10
datapath = sys.argv[2] 
ngames = 20 
ngpu = 8
nthread = 6 
class Distribution(object):
  def __init__(self):
    self.resnet_gpu = [0]
    self.resnet_name = [0]
    self.resnet_prob = [0]
    self.lenet_gpu = [0]
    self.lenet_name = [0]
    self.lenet_prob = [0]

dis = Distribution()

def scheduler(): 
#sys.argv[1] is file name of the models' max throughputs 
  models = [] 
  throughputs = [] 
  request_rates = [] 
  gpu_needs = [] 
  
  def rate_ratio_generator(ngames, theta = 0.9): 
    x = np.arange(1., ngames + 1) 
    game_rates = np.power(x, -theta) 
    game_rates = game_rates / max(game_rates) 
    for i in range(0, ngames): 
      request_rates.append(game_rates[i]) 
      request_rates.append(sum(game_rates) * 6) 
  
  def model_attribute(): 
    cnt = 0 
    with open(sys.argv[1]) as files: 
      for line in files: 
        tmp = line 
        throughputs.append(float(tmp)) 
        #print '***********' + line + '*************'
        if cnt < ngames: 
          models.append("resnet" + str(cnt)) 
        else: 
          models.append("lenet" + str(0)) 
        cnt = cnt + 1 
      rate_ratio_generator(ngames) 
      all_gpu_need = 0 
      for i in range(0, ngames + 1): 
        gpu_needs.append(request_rates[i] / throughputs[i]) 
        all_gpu_need += gpu_needs[i] 
      for i in range(0, ngames + 1): 
        gpu_needs[i] = gpu_needs[i] / all_gpu_need * ngpu 
  
  def naive_schedule(): 
    shares = [[models[i], gpu_needs[i]] for i in range(len(models))] 
    for i in range(len(shares)): 
      while shares[i][1] >= 1: 
        sch.append([(shares[i][0], 1)]) 
        shares[i][1] -= 1 
    shares = sorted(shares, key = lambda e: e[1], reverse = True) 
    for name, share in shares: 
      if share == 0: 
        continue 
      best_util = None 
      best_gpu = None 
      for i, gpu in enumerate(sch): 
        s = sum([e[1] for e in gpu]) + share 
        if s > 1.1: 
          continue 
        if best_util is None: 
          bext_util = s + share 
          best_gpu = i 
        elif best_util > 1: 
          if s < best_util: 
            best_util = s 
            best_gpu = i 
          else: 
            #best_util <= 1 
            if s <= 1 and s > best_util: 
              best_util = s 
              best_gpu = i 
      if best_gpu is None: 
        sch.append([(name, share)]) 
      else: 
        sch[best_gpu].append((name, share))
    #print sch
  
  def get_distribution():
    total_resnet = 0.0
    total_lenet = 0.0
    for i, gpu in enumerate(sch):
      for name, share in gpu:
        if 'resnet' in name:
          total_resnet += share
        if 'lenet' in name:
          total_lenet += share
    for i, gpu in enumerate(sch):
      for name, share in gpu:
        if 'resnet' in name:
          prob = share / total_resnet
          prob += dis.resnet_prob[len(dis.resnet_prob) - 1]
          dis.resnet_prob.append(prob)
          dis.resnet_name.append(name)
          dis.resnet_gpu.append(i)
        if 'lenet' in name:
          prob = share / total_lenet
          prob += dis.lenet_prob[len(dis.lenet_prob) - 1]
          dis.lenet_prob.append(prob)
          dis.lenet_name.append(name)
          dis.lenet_gpu.append(i)
  
  model_attribute() 
  naive_schedule() 
  get_distribution()

class Work(Thread):
  def __init__(self, name, img, x1, x2, y1, y2):
     super(Work, self).__init__()
     self.img = img
     self.name = name
     self.scale = [int(y1), int(y2), int(x1), int(x2)]
     self._return = True
  def run(self):
    host = 'localhost'
    tmp = 0
    name = ''
    img = self.img[self.scale[0]:self.scale[1], self.scale[2]:self.scale[3]]
    #print img.shape
    if 'resnet' in self.name:
      rand = random.random()
      for i in range(1, len(dis.resnet_prob)):
        if rand > dis.resnet_prob[i - 1] and rand < dis.resnet_prob[i]:
          tmp = i
      port = 9000 + int(dis.resnet_gpu[tmp])
      #print ('tmp = %s %s', tmp, dis.resnet_name[tmp])
      name = dis.resnet_name[tmp]
    if 'lenet' in self.name:
      rand = random.random()
      for i in range(1, len(dis.lenet_prob)):
        if rand > dis.lenet_prob[i - 1] and rand < dis.lenet_prob[i]:
          tmp = i
      port = 9000 + int(dis.lenet_gpu[tmp])
      name = dis.lenet_name[tmp]
      #print('lalalala %s %s' % (dis.lenet_name[tmp], tmp))
    #print 'check!!!'
    channel = implementations.insecure_channel(host, port)
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
    request = predict_pb2.PredictRequest()
    request.model_spec.name = name
    request.model_spec.signature_name = 'predict_images'
    request.inputs['images'].CopyFrom(
      tf.contrib.util.make_tensor_proto(cv2.imencode('.jpg', img)[1].tostring(), shape=[1]))
    start = datetime.now()
    try:
      result = stub.Predict(request, 5.0)
    except:
      self._return = False
    else:
      self._return = True
      #print 'OMG-----------------------------------------------------\n'
      #print 'OMG\n\n\n\n\n\n\n'

  def join(self):
    Thread.join(self)
    return self._return
    
class Dataset(object): 
  def __init__(self, datapath, max_count = 1000): 
    self.images = os.listdir(datapath) 

class SingleTest(Thread):
  def __init__(self, queue, dataset):
    super(SingleTest, self).__init__()
    self.dataset = dataset  
    self.queue = queue
    self.slas = []
    self.img_idx = random.randint(0, len(self.dataset.images) - 1)

  def run(self):
    while True:
      cmd = self.queue.get()
      if cmd == -1:
        break
      img_name = self.dataset.images[self.img_idx] 
      img = cv2.imread(datapath + img_name)
      clock = datetime.now()
      workers = []
      work = Work('resnet', img, 150, 178, 120, 148)
      workers.append(work)
      work = Work('resnet', img, 180, 208, 120, 148)
      workers.append(work)
      work = Work('resnet', img, 210, 238, 120, 148)
      workers.append(work)
      work = Work('resnet', img, 200, 228, 40, 68)
      workers.append(work)
      work = Work('resnet', img, 230, 258, 40, 68)
      workers.append(work)
      work = Work('resnet', img, 260, 288, 40, 68)
      workers.append(work)
      work = Work('lenet', img, 80, 240, 75, 150)
      workers.append(work)
      sla = 0
      for t in workers:
        t.start()
      for t in workers:
        ret = t.join()
        if ret is False:
          sla = 1000000.0
      if sla == 0:
        sla = (datetime.now() - clock).total_seconds() * 1000.0
      self.slas.append(sla)
      self.img_idx = (self.img_idx + 1) % len(self.dataset.images)

class Generator(object): 
  def __init__(self, dataset, output): 
    self.dataset = dataset 
    self.beg = None 
    self.slas = []
    self.output = output
    self.queue = Queue()
    self.single = []
    for i in range(0, 100):
      single_test = SingleTest(self.queue, self.dataset)
      single_test.start()
      self.single.append(single_test)

  def stop_all(self):
    for _ in range(len(self.single)):
      self.queue.put(-1)
    for t in self.single:
      t.join()
    elapse = time.time() - self.beg 
    logging.info('Finished all requests in {} sec'.format(elapse)) 

  def output_latency(self):
    #print "\n\n=============output lattency========\n\n"
    self.stop_all()
    with open(self.output, 'a') as fout:
      for t in self.single:
        for sla in t.slas:
          fout.write('%s\n' % (sla))

  def run(self, rqs, duration): 
    count = 0 
    gap = 1. / rqs 
    beg = time.time() 
    self.beg = time.time()
    total = duration * rqs 
    while True: 
      now = time.time() 
      while count * gap <= now - beg: 
        self.queue.put(1)
        count += 1 
        now = time.time() 
        if count >= total: 
          break 
      if count >= total or now - beg >= duration: 
        break 
      to_sleep = beg + count * gap - now 
      if to_sleep > 0:
        time.sleep(to_sleep) 
    
def run_test(datapath, rqs, duration, output): 
  dataset = Dataset(datapath) 
  gen = Generator(dataset, output) 
  gen.run(rqs, duration) 
  gen.output_latency()

def parse_result(fn):
  total, good = 0, 0
  with open(fn) as f:
    for line in f:
      total += 1
      if float(line.strip()) < float(50):
        print line.strip()
        good += 1
  return good, total

def run(rqs, epoch): 
  print('Testing rqs %s' % rqs) 
  threads = [] 
  outputs = [] 
  dataset = Dataset(datapath) 
  output = 'logs/request_rate{}_iter{}'.format(rqs, epoch) 
  run_test(datapath, rqs, duration, output) 
  total = 0 
  ngood = 0 
  ngood, total = parse_result(output) 
  percent = float(ngood) / total 
  print('request_rate%s_iter%s: %s ... %s/%s' % (rqs, epoch, percent * 100, ngood, total)) 
  if(percent < .99): 
    return False 
  return True 

def search(rqs, step): 
  if(step < 1.5): 
    for rqs in np.arange(rqs - step - step + 0.5, rqs, 0.5): 
      for i in range(6): 
        achieve = run(rqs, i) 
        if achieve: 
          break 
      if not achieve: 
        break 
    return 
  next_step = int(step / 2) 
  flag = False 
  for i in range(6): 
    achieve = run(rqs - step, i) 
    if achieve: 
      search(rqs, next_step) 
      flag = True 
      break 
  if not flag: 
    search(rqs - step, next_step) 

def generate_confs(): 
  for i, gpu in enumerate(sch): 
    conf = open('tfserv_' + str(i) + '.conf', 'w') 
    conf.write('model_config_list{\n') 
    size = len(gpu) 
    cnt = 0 
    for name, share in gpu: 
      cnt += 1 
      conf.write('config: {\n') 
      conf.write('name: "' + name + '",\n') 
      if 'lenet' in name: 
        conf.write('base_path: "/lenet",\n') 
      else: 
        conf.write('base_path: "/resNet",\n') 
      conf.write('model_platform: "tensorflow"\n}') 
      if cnt == size: 
        continue 
      else: 
        conf.write(',\n') 
    conf.write('\n}')
tfservers = [] 

def call_os(command):
  os.system(command)

def start_tfserving(): 
  for i, gpu in enumerate(sch): 
    gpu_flag = ' CUDA_VISIBLE_DEVICES=' + str(i) 
    port_flag = ' --port=' + str(i + 9000) 
    batch_flag = ' --enable_batching=true --batching_parameters_file='
    model_config = ' --model_config_file=tfserv_' + str(i) + '.conf' 
    contain_resnet = False 
    for name, share in gpu: 
      if "resnet" in name: 
        contain_resnet = True 
    if contain_resnet is True: 
      batch_flag += 'batching_parameter_resnet.txt' 
    else: 
      batch_flag += 'batching_parameter_lenet.txt' 
    command = 'env' + gpu_flag + ' tensorflow_model_server' + port_flag 
    command += batch_flag + model_config 
    #print command
    t = Thread(target = call_os, args = (command, )) 
    tfservers.append(t)
    t.start()

def main(): 
  scheduler() 
  generate_confs() 
  start_tfserving() 
  time.sleep(15)
  search(100, 64) 
  for t in tfservers:
    t.join()
if __name__ == "__main__": 
  main()

