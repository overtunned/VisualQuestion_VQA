import sys
sys.path.insert(0, '/home/ok_sikha/abhishek/VisualQuestion_VQA/Visual_All')
import numpy as np 
import torch
import torchvision.transforms as transforms
import json
import _pickle as cPickle
from torch.utils.data import Dataset, DataLoader
import os
import utils
from PIL import Image
from dataset_vqa import Dictionary, VQAFeatureDataset
import glob
import matplotlib.pyplot as plt
import cv2
import torchvision.transforms.functional as F
from models import *
from tqdm import tqdm
import time
import h5py
from flair.embeddings import BertEmbeddings,DocumentPoolEmbeddings
from flair.data import Sentence
import pickle 

def _create_entry(question, answer):
    answer.pop('image_id')
    answer.pop('question_id')
    entry = {
        'question_id' : question['question_id'],
        'image_id'    : question['image_id'],
        'question'    : question['question'],
        'answer'      : answer}
    return entry
def _load_dataset(dataroot,name):
    question_path = os.path.join(
        dataroot, 'vgenome_%s2021_2_questions.json' % name)
    questions = sorted(json.load(open(question_path)),
                       key=lambda x: x['question_id'])
    answer_path = os.path.join(dataroot,'cache', '%s_target_top_3000_ans.pkl' % name)
    answers = cPickle.load(open(answer_path, 'rb'))
    answers = sorted(answers, key=lambda x: x['question_id'])
    # print(answers[0])
    utils.assert_eq(len(questions), len(answers))
    entries = []
    print("No. of Questions:", len(questions))
    print("No. of answers:", len(answers))
    for question, answer in zip(questions, answers):
        utils.assert_eq(question['question_id'], answer['question_id'])
        utils.assert_eq(question['image_id'], answer['image_id'])
        img_id = question['image_id']
        if(len(answer['scores'])>0):
            entries.append(_create_entry(question, answer))
    return(entries)

class Dataset_VQA(Dataset):
    """Dataset for VQA applied to the attention case with image features in .hdf5 file 
    """
    def __init__(self,img_root_dir,feats_data_path,dictionary,dataroot,bert_option=False,rcnn_pkl_path=None,num_classes=3344,filename_len=12,choice='train',arch_choice='resnet152',layer_option='pool',transform_set=None):

        #initializations
        self.data_root=dataroot
        self.feats_data_path=feats_data_path
        self.img_dir=img_root_dir
        self.choice=choice
        self.transform=transform_set
        self.num_classes=num_classes
        self.dictionary=dictionary
        self.arch=arch_choice 
        self.layer_option=layer_option 
        self.filename_len=filename_len
        self.num_ans_candidates=num_classes
        self.rcnn_pkl_path=rcnn_pkl_path
        self.bert_option=bert_option
        #self.bert=BertEmbeddings('bert-base-uncased')
        #self.doc_bert=DocumentPoolEmbeddings([self.bert])
        start_time=time.time()
        if(self.rcnn_pkl_path is not None):
            print('Loading the hdf5 features from rcnn')
            self.pkl_data=pickle.load(open(self.rcnn_pkl_path,'rb'))
            h5_path=os.path.join(feats_data_path,self.choice+'36.hdf5')
            hf=h5py.File(h5_path, 'r')
            self.features=hf.get('image_features')
        # else:
        #     print('Loading the hdf5 features from resnet152')
            
        #     h5_path = os.path.join(feats_data_path,self.choice+'_feats_'+self.arch+'_'+self.layer_option+'.hdf5')
        #     file_path=os.path.join(feats_data_path,self.choice+'_filenames_'+self.arch+'.txt')
        #     fl_p=open(file_path)
        #     self.file_list=list(fl_p.readlines())
        #     self.file_list=[filename.split("\n")[0] for filename in self.file_list]
        #     hf=h5py.File(h5_path)
        #     self.features=hf.get('feats')

        #loading bert features 
        if(self.bert_option is True):
            #load bert features from .hdf5 file 
            h5_path_bert=os.path.join(self.data_root,self.choice+'_bert.hdf5')
            #print(h5_path_bert)
            hf_bert=h5py.File(h5_path_bert)
            self.bert_features=hf_bert.get('bert_embeddings')
            print('Bert shape')
            print(self.bert_features.shape)
            print('Loading question ids')
            self.quest_ids=list(hf_bert.get('question_ids'))
            print('Question ids loaded')
        #with h5py.File(h5_path, 'r') as hf:
        #    self.features = np.array(hf.get('feats'))
        end_time=time.time()
        elapsed_time=end_time-start_time
        print('Total elapsed time: %f' %(elapsed_time))
        
        #self.v_dim = 7
        self.entries=_load_dataset(self.data_root,self.choice)
        self.tokenize()
        self.tensorize()
        self.v_dim=2048

    def tokenize(self, max_length=14):
        """Tokenizes the questions.

        This will add q_token in each entry of the dataset.
        -1 represent nil, and should be treated as padding_idx in embedding
        """
        for entry in self.entries:
            tokens = self.dictionary.tokenize(entry['question'], False)
            tokens = tokens[:max_length]
            if len(tokens) < max_length:
                # Note here we pad in front of the sentence
                padding = [self.dictionary.padding_idx] * (max_length - len(tokens))
                tokens = padding + tokens
            utils.assert_eq(len(tokens), max_length)
            entry['q_token'] = tokens

    def tensorize(self):
        #self.features = torch.from_numpy(self.features)
        #self.spatials = torch.from_numpy(self.spatials)

        for entry in self.entries:
            question = torch.from_numpy(np.array(entry['q_token']))
            entry['q_token'] = question
            answer = entry['answer']
            class_labels = np.array(answer['Class_Label'])
            labels=np.array(answer['labels'])
            scores = np.array(answer['scores'], dtype=np.float32)
            if len(labels):
                labels = torch.from_numpy(labels)
                scores = torch.from_numpy(scores)
                class_labels=torch.from_numpy(class_labels)
                entry['answer']['labels'] = labels
                entry['answer']['scores'] = scores
                entry['answer']['Class_Label']=class_labels
            else:
                entry['answer']['labels'] = None
                entry['answer']['scores'] = None
                entry['answer']['Class_Label']=None


    def __getitem__(self,index):
        entry=self.entries[index]
        #filtering of the labels based on the score
        question=entry['q_token']
        answer_data=entry['answer']
        label=answer_data['Class_Label']
        image_id=entry['image_id']
        question_sent=entry['question']
        question_id=entry['question_id']

        #print(label)
        #print(type(question_sent))
        #sentence=Sentence(question_sent)
        #print('Here')
        #self.bert.embed(sentence)
        #print(sentence[0].embedding.shape)
        if(self.rcnn_pkl_path is not None):
            if(image_id in self.pkl_data):
                idx=self.pkl_data[image_id]
                feat=torch.from_numpy(self.features[idx])
                # print(feat.size())
        else:
            filename=str(image_id)+'.jpg'
            idx=self.file_list.index(os.path.join(self.img_dir,filename))
            
            feat=torch.from_numpy(self.features[idx])
            feat=feat.view(feat.size(0),feat.size(1)*feat.size(2))
            feat=feat.transpose(1,0)
        target = torch.zeros(self.num_classes)
        if label is not None:
            target.scatter_(0,label,1)

        if(self.bert_option is True):
            index_question=self.quest_ids.index(question_id)
            quest_feats=torch.from_numpy(self.bert_features[index_question])
            quest_feats=quest_feats.float()
            return(feat,quest_feats,question_sent,question_id,target)
        else:

            return(feat,question,question_sent,question_id,target)

    def __len__(self):
        return(len(self.entries))


if __name__ == "__main__":
    image_root_dir="/home/ok_sikha/abhishek/VG_100K"
    dictionary=Dictionary.load_from_file('/home/ok_sikha/abhishek/VisualQuestion_VQA/data/dictionary.pkl')
    feats_data_path="/home/ok_sikha/abhishek/VisualQuestion_VQA/data"
    data_root="/home/ok_sikha/abhishek/VisualQuestion_VQA/data"
    train_rcnn_pickle_file="/home/ok_sikha/abhishek/VisualQuestion_VQA/data/train36_imgid2idx.pkl"
    train_dataset=Dataset_VQA(img_root_dir=image_root_dir,
                                feats_data_path=feats_data_path,
                                dictionary=dictionary,
                                dataroot=data_root,
                                rcnn_pkl_path=train_rcnn_pickle_file,
                                choice='train',
                                arch_choice="rcnn",
                                layer_option="pool")
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=1)

    feat,question,question_sent,question_id,target=next(iter(train_loader))

    print(feat.shape)
    print(question)
    print(question_sent)
    print(question_id)
    print(target)