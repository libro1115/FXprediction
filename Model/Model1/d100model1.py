import torch
import torch.nn as nn
input = 100*4
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
#20/8model
# score: 331810
# active_acc: 20370
# active_loss: 3485
# passive_acc: 179114
# passive_loss: 3184

#test
# score: 144680
# active_acc: 8720
# active_loss: 1228
# passive_acc: 332004
# passive_loss: 1566

##########################
#look 12*3bar

# score: 1301354
# active_acc: 78648
# active_loss: 11431
# passive_acc: 135861
# passive_loss: 14570

#test
# score: 1205968
# active_acc: 71336
# active_loss: 7808
# passive_acc: 248509
# passive_loss: 15835

#v2
# score: 1368560
# active_acc: 83950
# active_loss: 14254
# passive_acc: 133036
# passive_loss: 9270
#v2test
# score: 1287734
# active_acc: 78488
# active_loss: 12505
# passive_acc: 243805
# passive_loss: 8690
#10/5model

# score: 794287
# active_acc: 97183
# active_loss: 16072
# passive_acc: 108055
# passive_loss: 19200

#test
# score: 1108470
# active_acc: 131655
# active_loss: 15285
# passive_acc: 169297
# passive_loss: 27251
