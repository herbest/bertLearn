import torch
import random
from torch.utils.data import DataLoader

## =================================================================
## ���́F�g�[�N���f�[�^��DataLoader�ɕϊ�����
## =================================================================
class DataLoaderConverter:
    # ========================
    # �R���X�g���N�^
    # inputLoader      ���̓f�[�^���[�h�p�N���X�C���X�^���X
    #                  load()���\�b�h���������Ă���
    # ========================
    def __init__(self, batchSize):
        self.batchSize = batchSize

    def convert(self, inputToken):
        tensorList = self.__convertTensor(inputToken)	# torch�ň�����`���ɕϊ�
        dsTrain, dsVal, dsTest = self.__dataSplit(tensorList)
        print('�g���[�j���O�f�[�^:{0}�A���؃f�[�^:{1}�A�e�X�g�f�[�^:{2}�A���v:{3}'.format(len(dsTrain), len(dsVal), len(dsTest), len(tensorList)))

        # �f�[�^�Z�b�g����f�[�^���[�_���쐬
        # �w�K�f�[�^��shuffle=True�ɂ���B
        dataloader_train = DataLoader(dsTrain, batch_size=self.batchSize, shuffle=True)
        dataloader_val   = DataLoader(dsVal,   batch_size=256)
        dataloader_test  = DataLoader(dsTest,  batch_size=256)
        return [dataloader_train, dataloader_val, dataloader_test]

    ## �f�[�^�Z�b�g����
    def __dataSplit(self, tensorList):
        random.shuffle(tensorList) # �����_���ɃV���b�t��
        n = len(tensorList)
        n_train = int(0.6*n)
        n_val = int(0.2*n)
        dataset_train = tensorList[:n_train]				# �w�K�f�[�^(0 �` MAX*0.6)
        dataset_val = tensorList[n_train:n_train+n_val]		# ���؃f�[�^(MAX*0.6 �` MAX*0.8)
        dataset_test = tensorList[n_train+n_val:]			# �e�X�g�f�[�^(MAX*0.8 �` MAX)
        return [dataset_train, dataset_val, dataset_test]

    ## torch�ň�����`���ɕϊ�
    def __convertTensor(self, input):
        encodingList = []
        for line in input:
            encoding = { k: torch.tensor(v) for k, v in line.items() }
            encodingList.append(encoding)
        return encodingList

