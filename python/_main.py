## =================================================================
## �p�b�P�[�W�C���X�g�[��
## =================================================================
!pip install --upgrade pip
!pip install transformers==4.18.0 fugashi==1.1.0 ipadic==1.0.0 pytorch-lightning==1.6.1
!pip install livelossplot --quiet # acc, loss�O���t�\���p

## =================================================================
## ���C������
## =================================================================
## BERT���O�w�K���f���̃��[�h
from transformers import BertJapaneseTokenizer
MODEL_NAME = 'cl-tohoku/bert-base-japanese-whole-word-masking'
bertTokenizer = BertJapaneseTokenizer.from_pretrained(MODEL_NAME)
plotViewer = trainViewer()
preModel = BertForSequenceClassification_pl(MODEL_NAME, num_labels=2, lr=1e-5, labelName='labels', lossViewer=plotViewer)

## Google�h���C�u���}�E���g
from google.colab import drive
drive.mount('/content/drive')
workDirPath = '/content/drive/MyDrive/SoftmBert'

## =================================================================
## �f�[�^�쐬
## =================================================================
## ���̓f�[�^
import os
import shutil
inputFilePathDir = os.path.join(workDirPath, 'Data')

## ���̓f�[�^���g�[�N����
maxTokenLength = 256
inputLoader = csvLoader(inputFilePathDir, True)
qaTokenizer = QATokenizer(bertTokenizer, inputLoader)
inputData = qaTokenizer.loadData()
inputToken = qaTokenizer.tokenizer(inputData[0:1000], max_length=maxTokenLength)   # ���̓f�[�^�̎g�p��(�f�o�b�O�p)

## BERT�g���[�j���O�f�[�^�쐬
max_epochs = 1
batch_size = 16 # colab�ł��Ȃ�16��GPU�̌��E���ۂ��B���₷�Ƒ����Ȃ邪GPU�g�p���E���z����B
modelDir = os.path.join(workDirPath, 'Model/')
dlTrain, dlVal, dlTest = DataLoaderConverter(batch_size).convert(inputToken)
trainer, checkPoint = TrainerBuilder(max_epochs, modelDir).build()
print('�g���[�j���O�f�[�^:{0}'.format(len(inputToken) *0.6))
print('�o�b�`�T�C�Y�F{0}�A�G�|�b�N���F{1}�AMAX�g�[�N���T�C�Y�F{2}'.format(batch_size, max_epochs, maxTokenLength))
print('�g���[�j���O���s�񐔁F{0}'.format( ((len(inputToken) *0.6) / batch_size) * max_epochs  ))

## =================================================================
## �t�@�C���`���[�j���O
## =================================================================
## �f�B���N�g������ɂ���
if os.path.exists(modelDir):
    shutil.rmtree(modelDir)
os.mkdir(modelDir)

## fit
trainer.fit(preModel, dlTrain, dlVal)
