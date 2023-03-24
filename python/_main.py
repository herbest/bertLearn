## =================================================================
## パッケージインストール
## =================================================================
!pip install --upgrade pip
!pip install transformers==4.18.0 fugashi==1.1.0 ipadic==1.0.0 pytorch-lightning==1.6.1
!pip install livelossplot --quiet # acc, lossグラフ表示用

## =================================================================
## メイン処理
## =================================================================
## BERT事前学習モデルのロード
from transformers import BertJapaneseTokenizer
MODEL_NAME = 'cl-tohoku/bert-base-japanese-whole-word-masking'
bertTokenizer = BertJapaneseTokenizer.from_pretrained(MODEL_NAME)
plotViewer = trainViewer()
preModel = BertForSequenceClassification_pl(MODEL_NAME, num_labels=2, lr=1e-5, labelName='labels', lossViewer=plotViewer)

## Googleドライブをマウント
from google.colab import drive
drive.mount('/content/drive')
workDirPath = '/content/drive/MyDrive/SoftmBert'

## =================================================================
## データ作成
## =================================================================
## 入力データ
import os
import shutil
inputFilePathDir = os.path.join(workDirPath, 'Data')

## 入力データをトークン化
maxTokenLength = 256
inputLoader = csvLoader(inputFilePathDir, True)
qaTokenizer = QATokenizer(bertTokenizer, inputLoader)
inputData = qaTokenizer.loadData()
inputToken = qaTokenizer.tokenizer(inputData[0:1000], max_length=maxTokenLength)   # 入力データの使用数(デバッグ用)

## BERTトレーニングデータ作成
max_epochs = 1
batch_size = 16 # colabでやるなら16がGPUの限界っぽい。増やすと早くなるがGPU使用限界を越える。
modelDir = os.path.join(workDirPath, 'Model/')
dlTrain, dlVal, dlTest = DataLoaderConverter(batch_size).convert(inputToken)
trainer, checkPoint = TrainerBuilder(max_epochs, modelDir).build()
print('トレーニングデータ:{0}'.format(len(inputToken) *0.6))
print('バッチサイズ：{0}、エポック数：{1}、MAXトークンサイズ：{2}'.format(batch_size, max_epochs, maxTokenLength))
print('トレーニング実行回数：{0}'.format( ((len(inputToken) *0.6) / batch_size) * max_epochs  ))

## =================================================================
## ファインチューニング
## =================================================================
## ディレクトリを空にする
if os.path.exists(modelDir):
    shutil.rmtree(modelDir)
os.mkdir(modelDir)

## fit
trainer.fit(preModel, dlTrain, dlVal)
