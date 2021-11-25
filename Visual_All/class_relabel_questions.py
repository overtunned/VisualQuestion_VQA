import json
import pandas as pd 
import numpy as np 
from tqdm import tqdm
import pickle

def add_entry(map,label_list):
    relabel_list=[map[str(lab)] for lab in label_list]
    return(relabel_list)


train_label_file='/home/ok_sikha/abhishek/VisualQuestion_VQA/data/train_target_top_3000_ans.pkl'
validation_label_file='/home/ok_sikha/abhishek/VisualQuestion_VQA/data/val_target_top_3000_ans.pkl'
csv_file="/home/ok_sikha/abhishek/VisualQuestion_VQA/data/Train_3000_Class_Distribution.csv"
train_dataset=pickle.load(open(train_label_file,'rb'))
valid_dataset=pickle.load(open(validation_label_file,'rb'))

class_distribution=pd.read_csv(csv_file)
keys_set=class_distribution['Label_indices'].tolist()
keys_set=[str(key) for key in keys_set]

label2indices=dict(zip(keys_set,class_distribution['Relabel_class'].tolist()))
# print(label2indices)

print('RELABELLING TRAINING DATA')
for data_sample in tqdm(train_dataset):
    label=data_sample['labels']
    score=data_sample['scores']
    if(len(score)>0):
        max_id=score.index(max(score))
        label_question=label[max_id]
        relabel_data=label2indices[str(label_question)]
    data_sample['Class_Label']=relabel_data
#train_questions_dict={}
#train_questions_dict['questions']=train_questions

with open('/home/ok_sikha/abhishek/VisualQuestion_VQA/data/cache/train_target_top_3000_ans.pkl', 'wb') as fp:
    pickle.dump(train_dataset, fp)

print(train_dataset[0]['Class_Label'])


print('RELABELLING VALIDATION DATA')
for data_sample in tqdm(valid_dataset):
    label=data_sample['labels']
    score=data_sample['scores']
    if(len(score)>0):
        max_id=score.index(max(score))
        label_question=label[max_id]
        relabel_data=label2indices[str(label_question)]
    #relabel_data=add_entry(label2indices,label)
    data_sample['Class_Label']=relabel_data

#valid_questions_dict={}
#valid_questions_dict['questions']=val_questions
with open('/home/ok_sikha/abhishek/VisualQuestion_VQA/data/cache/validation_target_top_3000_ans.pkl','wb') as fp:
    pickle.dump(valid_dataset, fp)





