import torch 
import json 
from flair.embeddings import BertEmbeddings,DocumentPoolEmbeddings
import numpy as np 
from tqdm import tqdm 
from flair.data import Sentence
import os 
import h5py

def extract_bert_features(json_file,dataroot_folder,split="train"):
    questions=json.load(open(json_file))

    question_ids=[quest['question_id'] for quest in questions]
    #questions=questions[0:10]
    bert=BertEmbeddings('bert-base-multilingual-cased')
    doc_bert=DocumentPoolEmbeddings([bert])
    bert_embed_matrix=np.zeros((len(questions),3072))
    print('Extracting bert features')
    
    for index,quest in tqdm(enumerate(questions)):
        sentence=Sentence(quest['question'])
        doc_bert.embed(sentence)
        bert_embed_matrix[index]=sentence.embedding.numpy()
    
    hdf5_file_path=os.path.join(dataroot_folder,split+'_bert.hdf5')
    h5f = h5py.File(hdf5_file_path, 'w')
    h5f.create_dataset('bert_embeddings', data=bert_embed_matrix)
    h5f.create_dataset('question_ids', data=question_ids)
    h5f.close()

if __name__ == "__main__":
    # json_file="/proj/digbose92/VQA/VisualQuestion_VQA/Visual_All/data/v2_OpenEnded_mscoco_train2014_yes_no_questions.json"
    # dataroot_folder="/data/digbose92/VQA/COCO/train_hdf5_COCO"
    dataroot_folder="/home/ok_sikha/abhishek/VisualQuestion_VQA/"
    json_file=dataroot_folder+"data/vgenome_train2021_questions.json"
#    json_file="/proj/digbose92/VQA/VisualQuestion_VQA/Visual_All/data/v2_OpenEnded_mscoco_val2014_yes_no_questions.json"
#   dataroot_folder="/proj/digbose92/VQA/VisualQuestion_VQA/common_resources"
    extract_bert_features(json_file,dataroot_folder)
