# -*- coding:utf-8 -*-
import os
import torch
import shutil
import random
import argparse
import sentencepiece as spm
from torch.utils.data import Dataset


DATA_BASE_DIR = './data'


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--collect_dataset', action='store_true')
    return parser.parse_args()


class Dictionary(object):
    def __init__(self, src_path, trg_path):
        self.src_path = src_path
        self.trg_path = trg_path
        self.model_prefix = 'paper'

    def model(self):
        # Word Sentence Piece 모델 Get
        sp = spm.SentencePieceProcessor()
        path = os.path.join(self.trg_path, self.model_prefix+'.model')
        if sp.load(path):
            return sp
        else:
            raise FileExistsError()

    def train(self, vocab_size):
        # Word Sentence Piece 학습
        spm.SentencePieceTrainer.Train(
            '--input={} --model_prefix={} --vocab_size={} '
            '--bos_id=0 --eos_id=1 --unk_id=2 --pad_id=3'.format(
                self.src_path, self.model_prefix, vocab_size
            )
        )

        # 딕셔너리 저장 파일 생성
        if not os.path.exists(self.trg_path):
            os.mkdir(self.trg_path)

        # Word Sentence Piece 모델 저장
        cur_dir = os.curdir
        model_name = os.path.join(cur_dir, self.model_prefix+'.model')
        vocab_name = os.path.join(cur_dir, self.model_prefix+'.vocab')
        shutil.move(model_name, self.trg_path)
        shutil.move(vocab_name, self.trg_path)


class PaperDataset(Dataset):
    def __init__(self, path, sp, encoder_sentence_size, decoder_sentence_size):
        self.lines = open(path, encoding='utf-8').readlines()
        self.encoder_sentence_size = encoder_sentence_size
        self.decoder_sentence_size = decoder_sentence_size
        self.sp = sp
        self.bos_id = self.sp.PieceToId('<s>')
        self.eos_id = self.sp.PieceToId('</s>')
        self.pad_id = self.sp.PieceToId('<pad>')

    def __getitem__(self, item):
        line = self.lines[item]
        title, contents = line.split('\t')
        enc_input = self.enc_input_to_vector(sentence=contents, size=self.encoder_sentence_size)
        dec_input = self.dec_input_to_vector(sentence=title, size=self.decoder_sentence_size)
        dec_output = self.dec_output_to_vector(sentence=title, size=self.decoder_sentence_size)
        return enc_input, dec_input, dec_output

    def enc_input_to_vector(self, sentence, size):
        # 초록(Abstract)을 벡터로 Convert
        vector = self.sp.EncodeAsIds(sentence)
        vector.insert(0, self.bos_id)
        vector.insert(-1, self.eos_id)
        vector = self.padding(vector, size)
        return torch.tensor(vector)

    def dec_input_to_vector(self, sentence, size):
        # 제목(Title)을 벡터로 Convert
        vector = self.sp.EncodeAsIds(sentence)
        vector.insert(0, self.bos_id)
        vector = self.padding(vector, size)
        return torch.tensor(vector)

    def dec_output_to_vector(self, sentence, size):
        # 제목(Title)을 벡터로 Convert
        vector = self.sp.EncodeAsIds(sentence)
        vector.insert(-1, self.eos_id)
        vector = self.padding(vector, size)
        return torch.tensor(vector)

    def padding(self, vector, size):
        vector_size = len(vector)
        if vector_size >= size:
            vector = vector[:size]
        remainder_size = size - vector_size
        remainder_vector = [self.pad_id for _ in range(remainder_size)]
        vector = vector + remainder_vector
        return vector

    def __len__(self):
        return len(self.lines)


def collect_paper_dataset():
    # 아카이브 논문 데이터셋 수집
    import arxiv
    keywords = ['machine learning', 'deep learning', 'support vector machine',
                'gradient descent', 'convolution neural network', 'recurrent neural network']

    # Collecting Dataset
    total_content = []
    for keyword in keywords:
        keyword = keyword.strip()
        temp = arxiv.query(query=keyword, max_results=10000)
        for t in temp:
            title = ' '.join(t['title'].split())
            summary = ' '.join(t['summary'].split())
            line = title + '\t' + summary
            total_content.append(line)

    # Shuffling
    random.shuffle(total_content)

    # Split Train dataset / Test dataset / Validation dataset
    train_ratio = 0.80
    val_ratio = 0.10

    total_size = len(total_content)
    with open(os.path.join(DATA_BASE_DIR, 'train.txt'), 'w', encoding='utf-8') as f:
        point = int(total_size * train_ratio)

        for line in total_content[:point]:
            f.write(line+'\n')

    with open(os.path.join(DATA_BASE_DIR, 'val.txt'), 'w', encoding='utf-8') as f:
        start_point = int(total_size * train_ratio)
        end_point = int(total_size * (train_ratio + val_ratio))

        for line in total_content[start_point:end_point]:
            f.write(line+'\n')

    with open(os.path.join(DATA_BASE_DIR, 'test.txt'), 'w', encoding='utf-8') as f:
        point = int(total_size * (train_ratio + val_ratio))

        for line in total_content[point:]:
            f.write(line+'\n')


if __name__ == '__main__':
    args = get_args()

    if args.collect_dataset:
        # 아카이브 논문 데이터셋 수집
        collect_paper_dataset()
