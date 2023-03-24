import torch
import random
from torch.utils.data import DataLoader

## =================================================================
## 名称：トークンデータをDataLoaderに変換する
## =================================================================
class DataLoaderConverter:
    # ========================
    # コンストラクタ
    # inputLoader      入力データロード用クラスインスタンス
    #                  load()メソッドを実装している
    # ========================
    def __init__(self, batchSize):
        self.batchSize = batchSize

    def convert(self, inputToken):
        tensorList = self.__convertTensor(inputToken)	# torchで扱える形式に変換
        dsTrain, dsVal, dsTest = self.__dataSplit(tensorList)
        print('トレーニングデータ:{0}、検証データ:{1}、テストデータ:{2}、合計:{3}'.format(len(dsTrain), len(dsVal), len(dsTest), len(tensorList)))

        # データセットからデータローダを作成
        # 学習データはshuffle=Trueにする。
        dataloader_train = DataLoader(dsTrain, batch_size=self.batchSize, shuffle=True)
        dataloader_val   = DataLoader(dsVal,   batch_size=256)
        dataloader_test  = DataLoader(dsTest,  batch_size=256)
        return [dataloader_train, dataloader_val, dataloader_test]

    ## データセット分割
    def __dataSplit(self, tensorList):
        random.shuffle(tensorList) # ランダムにシャッフル
        n = len(tensorList)
        n_train = int(0.6*n)
        n_val = int(0.2*n)
        dataset_train = tensorList[:n_train]				# 学習データ(0 ～ MAX*0.6)
        dataset_val = tensorList[n_train:n_train+n_val]		# 検証データ(MAX*0.6 ～ MAX*0.8)
        dataset_test = tensorList[n_train+n_val:]			# テストデータ(MAX*0.8 ～ MAX)
        return [dataset_train, dataset_val, dataset_test]

    ## torchで扱える形式に変換
    def __convertTensor(self, input):
        encodingList = []
        for line in input:
            encoding = { k: torch.tensor(v) for k, v in line.items() }
            encodingList.append(encoding)
        return encodingList

