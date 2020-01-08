# -*- coding:utf-8 -*-
import os
from data_helper import PaperDataset, Dictionary
from torch.utils.data import DataLoader


class PaperTitleExtractor(object):
    def __init__(self, arguments):
        self.arguments = arguments
        self.wsp_dict = Dictionary(
            src_path=self.arguments.train_path,
            trg_path=self.arguments.dict_path
        )

    def train(self):
        train_data_loader = self.train_data_loader()
        for _ in range(self.arguments.epochs):
            for data in train_data_loader:
                enc_input, dec_input, dec_output = data
                print(enc_input.shape)
                print(dec_input.shape)
                print(dec_output.shape)
                exit()

    def val(self):
        pass

    def make_dictionary(self):
        # 구글 Word Sentence Piece 모델 사용을 위한 딕셔너리 생성
        self.wsp_dict.train(self.arguments.vocab_size)

    def train_data_loader(self):
        # Train 데이터셋 입력을 위한 DataLoader
        dataset = PaperDataset(
            path=self.arguments.train_path,
            sp=self.wsp_dict.model(),
            encoder_sentence_size=self.arguments.encoder_sentence_size,
            decoder_sentence_size=self.arguments.decoder_sentence_size
        )
        data_loader = DataLoader(dataset, batch_size=self.arguments.batch_size, shuffle=True)
        return data_loader

    def val_data_loader(self):
        # Validation 데이터셋 입력을 위한 DataLoader
        pass

    def test_data_loader(self):
        # Test 데이터셋 입력을 위한 DataLoader
        pass
