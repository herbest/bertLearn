## =================================================================
## ���́F�����@�J�e�S���C�Y�p��tokenizer
## =================================================================
class QATokenizer:
    # ========================
    # �R���X�g���N�^
    # inputLoader      ���̓f�[�^���[�h�p�N���X�C���X�^���X
    #                  load()���\�b�h���������Ă���
    # ========================
    def __init__(self, bertTokenizer, inputLoader):
        self.bertTokenizer = bertTokenizer
        self.inputLoader = inputLoader

    def loadData(self):
        return self.inputLoader.load()

    # ========================
    # �g�[�N����
    # inputDataList    �񎟌����X�g
    # ========================
    def tokenizer(self, inputDataList, max_length=512):
        retList = []
        for line in inputDataList:
            text = line[0]
            label = line[2]

            encoding = self.bertTokenizer(text, max_length = max_length, padding = 'max_length', truncation = True)
            encoding['labels'] = 0 if label == 'IMBT' else 1
            retList.append(encoding)

        return retList
