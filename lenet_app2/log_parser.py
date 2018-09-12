import sys
import numpy as np
import matplotlib.pyplot as plt

x = []
y = []
with open(sys.argv[1]) as f:
    for line in f:
        if "tensorflow_serving/batching/batching_session.cc:618" in line:
            batch_size, time = line.strip().split()[-2:]
	    batch_size = int(batch_size.replace(':', ''))
	    print batch_size, time
            x.append(batch_size)
            y.append(time)
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_title('batch_size - time')
plt.xlabel('batch_size')
plt.ylabel('time')
ax1.scatter(x,y,c = 'b', marker = 'o')
plt.show
           
