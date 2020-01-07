from dl_utils import *
import numpy as np
import torch

a = np.random.random_integers(0, 10, size=100)
unique(a)

a = np.arange(10**5)
b = np.arange(10**5)
run_parallel(16, sqr, list(zip(a, b)))
run_parallel(16, sqr, list(zip(a)))

z = torch.randn(20, 3, 32, 32)
net = Flatten()
out = net(z)
print(out.shape)

z = torch.randn(20, 3, 32, 32)
net = Reshape((6, 16, 32))
out = net(z)
print(out.shape)

meters = NamedAverageMeter(['A', 'B'])
for i in range(100):
    x = torch.Tensor([i])
    meters.update({'A': (x, 1), 'B': (x**2, 5)})
avgs = meters.avg()
print(avgs)

metric_tracker = MetricTracker(metrics_to_track=['acc', 'f1'])
for i in range(100):
    op = torch.randn(20, 10)
    y = torch.argmax(torch.randn(20, 10), dim=1)
    metric_tracker.update(op, y)

metrics = metric_tracker.get_metrics()
print(metrics)
