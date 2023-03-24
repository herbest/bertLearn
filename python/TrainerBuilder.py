import pytorch_lightning as pl

## =================================================================
## ���́F�g���[�j���O�p�̃C���X�^���X�쐬
## =================================================================
class TrainerBuilder:
    # ========================
    # �R���X�g���N�^
    # ========================
    def __init__(self, epochs, modelTempPath):
        self.epochs = epochs
        self.modelTempPath = modelTempPath
        pass

    def build(self):
        # �w�K���Ƀ��f���̏d�݂�ۑ�����������w��
        checkpoint = pl.callbacks.ModelCheckpoint(
            monitor = 'val_loss',
            mode = 'min',
            save_top_k = 1,
            save_weights_only = True,
            dirpath = self.modelTempPath,
            )

        # �w�K�̕��@���w��
        trainer = pl.Trainer(
            gpus = 1,                    # �g�pGPU�� = 1
            val_check_interval = 0.5,    # �g���[�j���O���̌��،ďo����
            max_epochs = self.epochs,    # �G�|�b�N��
            callbacks = [checkpoint]
            )

        return [trainer, checkpoint]
