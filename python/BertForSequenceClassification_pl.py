## =================================================================
## PyTorch Lightningの定義
## =================================================================
import torch
import pytorch_lightning as pl
from transformers import BertForSequenceClassification

class BertForSequenceClassification_pl(pl.LightningModule):
    # ===========================================================
    # 名称：コンストラクタ
    # 引数：model_name		 Transformersのモデルの名前
    #       num_labels		ラベルの数
    #       lr				学習率
    # ===========================================================
    def __init__(self, model_name, num_labels, lr, labelName, lossViewer):
        super().__init__()
        self.save_hyperparameters()		## 以降、self.hparamsでnum_labelsとlrにアクセス出来る
        # BERTのロード
        self.bert_sc = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.bert_sc = self.bert_sc.cuda()
        self.lossViewer = lossViewer

    # 学習データのミニバッチ(`batch`)が与えられた時に損失を出力する関数を書く。
    # batch_idxはミニバッチの番号であるが今回は使わない。
    def training_step(self, batch, batch_idx):
        output = self.bert_sc(**batch)
        loss = output.loss
        self.log('train_loss', loss) # 損失を'train_loss'の名前でログをとる。
        self.lossViewer.setLoss(loss)
        return loss

    # 検証データのミニバッチが与えられた時に、
    # 検証データを評価する指標を計算する関数を書く。
    def validation_step(self, batch, batch_idx):
        output = self.bert_sc(**batch)
        val_loss = output.loss
        self.log('val_loss', val_loss) # 損失を'val_loss'の名前でログをとる。
        self.lossViewer.setValLoss(val_loss)

    # テストデータのミニバッチが与えられた時に、
    # テストデータを評価する指標を計算する関数を書く。
    def test_step(self, batch, batch_idx):
        labels = batch.pop(self.hparams.labelName) # バッチからラベルを取得
        output = self.bert_sc(**batch)
        labels_predicted = output.logits.argmax(-1)
        num_correct = ( labels_predicted == labels ).sum().item()
        accuracy = num_correct/labels.size(0) #精度
        self.log('accuracy', accuracy) # 精度を'accuracy'の名前でログをとる。

    # 学習に用いるオプティマイザを返す関数を書く。
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
