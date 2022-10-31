import torch
import torch.nn as nn
input = 200*4
output = 7
model1 = nn.Sequential(
        nn.BatchNorm1d(input),
        nn.Linear(input,1000),
        nn.ReLU(),
        nn.Linear(1000,500),
        nn.ReLU(),
        nn.Linear(500,250),
        nn.BatchNorm1d(250),
        nn.ReLU(),
        nn.Linear(250,100),
        nn.ReLU(),
        nn.Linear(100,output)
        )
        
# score: 355072
# active_acc: 21439
# active_loss: 3083
# passive_acc: 211971
# passive_loss: 4003

# test score
# score: 140970
# active_acc: 8420
# active_loss: 1059
# passive_acc: 332073
# passive_loss: 1866