import csv
import glob
import os

## =================================================================
## 名称：CSV読込みクラス
## =================================================================
class csvLoader:
    # ========================
    # コンストラクタ
    # dirPath : 入力CSVが入っているフォルダパス
    # hdr_skip      True :CSVの1行目をスキップする
    #               False:CSVの1行目スキップをしない
    # ========================
    def __init__(self, dirPath, hdr_skip=False):
        self.dirPath = dirPath
        self.hdr_skip = hdr_skip

    # ========================
    # 名称：CSVファイル読込み
    # 説明：CSVを読み込んで二次元リストを返す
    # 戻値：二次元リスト
    # ========================
    def load(self):
        chkFile = os.path.join(self.dirPath, "*.csv")
        schFileList = glob.glob(chkFile)

        retList = []
        for fPath in schFileList:
            print('read file : {0}'.format(fPath))
            with open(fPath, 'r', encoding = 'shift_jis', errors = 'ignore') as f:
                reader = csv.reader(f)
                if self.hdr_skip == True : next(reader) # ヘッダースキップ
                for line in reader:
                    retList.append(line)
        return retList
