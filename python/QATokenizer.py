## =================================================================
## 名称：処理機カテゴライズ用のtokenizer
## =================================================================
class QATokenizer:
    # ========================
    # コンストラクタ
    # inputLoader      入力データロード用クラスインスタンス
    #                  load()メソッドを実装している
    # ========================
    def __init__(self, bertTokenizer, inputLoader):
        self.bertTokenizer = bertTokenizer
        self.inputLoader = inputLoader

    def loadData(self):
        return self.inputLoader.load()

    # ========================
    # トークン化
    # inputDataList    二次元リスト
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
