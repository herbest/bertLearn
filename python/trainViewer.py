import numpy as np
import matplotlib.pyplot as plt
from livelossplot import PlotLosses
import torch
liveloss = PlotLosses()

## =================================================================
## 名称：学習状況の表示
## =================================================================
class trainViewer:
    # ========================
    # コンストラクタ
    # ========================
    def __init__(self):
        self.loss = []
        self.val_loss = []
        pass

    def setLoss(self, loss):
        self.loss.append(loss)
#         print('loss : {0}'.format(loss))

        logs = {}
        logs['loss'] = loss.detach().cpu()
        liveloss.update(logs)
        liveloss.send()

    def setValLoss(self, val_loss):
        self.val_loss.append(val_loss)
#         print('val_loss : {0}'.format(val_loss))

        logs = {}
        logs['val_loss'] = val_loss.detach().cpu()
        liveloss.update(logs)
        liveloss.send()

#     def plot():
#         left = np.array( range(0, len(self.loss)) )
#         height = np.array(self.loss)
#         plt.plot(left, height)

