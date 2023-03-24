import pytorch_lightning as pl

## =================================================================
## 名称：トレーニング用のインスタンス作成
## =================================================================
class TrainerBuilder:
    # ========================
    # コンストラクタ
    # ========================
    def __init__(self, epochs, modelTempPath):
        self.epochs = epochs
        self.modelTempPath = modelTempPath
        pass

    def build(self):
        # 学習時にモデルの重みを保存する条件を指定
        checkpoint = pl.callbacks.ModelCheckpoint(
            monitor = 'val_loss',
            mode = 'min',
            save_top_k = 1,
            save_weights_only = True,
            dirpath = self.modelTempPath,
            )

        # 学習の方法を指定
        trainer = pl.Trainer(
            gpus = 1,                    # 使用GPU数 = 1
            val_check_interval = 0.5,    # トレーニング中の検証呼出し率
            max_epochs = self.epochs,    # エポック数
            callbacks = [checkpoint]
            )

        return [trainer, checkpoint]
