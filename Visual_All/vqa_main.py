#repo for running the main part of VQA
#assume that vqa_utils is already run 
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pickle
import sys
sys.path.append('/content/VisualQuestion_VQA/Visual_All')
from dataset_vqa import Dictionary, VQAFeatureDataset
from models import EncoderLSTM, FusionModule, EncoderCNN
from dataset_image_vqa import VQADataset
import torchvision.transforms as transforms
from tqdm import tqdm
from torch.autograd import Variable
import pdb
import os
from torch.utils.tensorboard import SummaryWriter

def question_parse(token_list):
    data=pickle.load(open('/content/drive/MyDrive/College_paper/VisualQuestion_VQA/data1/dictionary.pkl','rb'))
    index2word_map=data[1]
    word_list=[]

    for idval in token_list.tolist():
        if(idval==56478):
            word_list.append(index2word_map[idval-1])
        else:
            word_list.append(index2word_map[idval])
    #word_list=[index2word_map[id] for id in token_list.tolist()]
    print(word_list)

def preproc_question_tokens(question_array):

    num_questions,seq_length=question_array.shape
    for i in np.arange(num_questions):
        index=np.where(question_array==56478)
        question_array[index]=56477
    return(question_array)

def convert_one_hot2int(one_hot):
    one_hot=one_hot.astype(int)
    class_ind=np.argmax(one_hot,axis=1)
    return(class_ind)

def main(args):
    writer = SummaryWriter()
    #defining torch configurations
    #torch.manual_seed(args.seed)
    #torch.cuda.manual_seed(args.seed)
    #torch.backends.cudnn.benchmark = True


    #extract weights from the weight matrices
    weights=np.load(args.file_name)

    # CUDA for PyTorch
    #if cuda:
    device=0
    torch.cuda.set_device(device)

    #device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    #use_cuda = torch.cuda.is_available()
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #defining dictionary and VQAFeatureDataset
    #transforms for pretrained network(transform for resnet now)
    train_transform = transforms.Compose([ 
        transforms.Resize((args.crop_size,args.crop_size)),
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])

    validate_transform=transforms.Compose([ 
        transforms.Resize((args.crop_size,args.crop_size)),
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])

    
    dictionary = Dictionary.load_from_file('/content/drive/MyDrive/College_paper/VisualQuestion_VQA/data1/dictionary.pkl')
    train_dataset = VQADataset(image_root_dir=args.img_root_dir,dictionary=dictionary,dataroot=args.data_root_dir,choice='train',transform_set=train_transform)
    eval_dataset = VQADataset(image_root_dir=args.img_root_dir,dictionary=dictionary,dataroot=args.data_root_dir,choice='val',transform_set=validate_transform)
    

    #model definition 
    print('Loading the models')
    image_encoder=EncoderCNN(embed_size=args.img_feats).to(device)
    question_encoder=EncoderLSTM(hidden_size=args.num_hid,weights_matrix=weights,fc_size=args.q_embed,max_seq_length=args.max_sequence_length,batch_size=args.batch_size).to(device)
    fusion_network=FusionModule(qnetwork=question_encoder,img_network=image_encoder,fuse_embed_size=args.fuse_embed,input_fc_size=args.img_feats,class_size=args.num_class).to(device)
    #print(list(fusion_network.parameters()))
    print(fusion_network)
    #input()
    

    #Dataloader initialization
    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=4)
    eval_loader =  DataLoader(eval_dataset, args.batch_size, shuffle=True, num_workers=2)

    # Loss and optimizer
    criterion = nn.NLLLoss()
    #params=lis
    #params = list(image_encoder.linear.parameters())+list(image_encoder.bn.parameters())+list(question_encoder.parameters()) + list(fusion_network.parameters()) 
    optimizer = torch.optim.Adam(fusion_network.parameters(), lr=args.learning_rate)

    # Train the models
    total_step = len(train_loader)
    step=0
    #Training starts
    print('Training Starting ......................')
    if os.path.exists(args.model_checkpoint):
      checkpoint = torch.load(PATH)
      fusion_network.load_state_dict(checkpoint['model_state_dict'])
      optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
      epochs = checkpoint['epoch']
      loss = checkpoint['loss']
      print('Modeled loaded from ', epochs)
    else:
      epochs = 0
    
    model_path = args.model_save_path
    # fusion_network = torch.load('/content/drive/MyDrive/College_paper/VisualQuestion_VQA/data1/vgg_ft19.pth')


    def evaluate_val(model,loader,criterion,device):
        loss=0
        accuracy=0
        print('Evaluation started')
        with torch.no_grad():
            for image_sample,question_token,labels in tqdm(loader):
                image_sample,question_token,labels = image_sample.to(device),question_token.to(device),labels.to(device)
                output=model.forward(question_token,image_sample)
                loss+= criterion(output,labels).item()
                ps = torch.exp(output)
                equality= (labels.data == ps.max(dim=1)[1])
                accuracy+=equality.type(torch.FloatTensor).mean()

        return loss,accuracy

    file_train=open(os.path.join(args.data_root_dir,'train_loss_log_vgg.txt'),'a+')
    loss_save=[]

    for epoch in range(args.epochs):

        running_loss = 0.0
        running_corrects = 0
        step=0
        for data in tqdm(train_loader):
            image_samp,question_toks,labels=data
            image_samp=image_samp.to(device)
            question_toks=question_toks.to(device)
            labels=labels.to(device)
            
            class_outputs=fusion_network(question_toks,image_samp)
            _, preds = torch.max(class_outputs, 1)
            loss = criterion(class_outputs, labels)
            #question_encoder.zero_grad()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #print('Enter some key')
            #input()
            # statistics
            running_loss += loss.item() * image_samp.size(0)
            running_corrects += torch.sum(preds == labels.data)
            if(step%500==0):
            #optimizer.zero_grad()
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                    .format(epoch, args.epochs, step, total_step, loss.item()))
            step=step+1
        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects.double() / len(train_dataset)
        writer.add_scalar('Loss/train', epoch_loss, epoch)
        writer.add_scalar('Accuracy/train', epoch_acc, epoch)
        print(epoch_loss)
        # loss_save.append(val_loss)
        
        val_loss,accuracy = evaluate_val(fusion_network,eval_loader,criterion,device)
        writer.add_scalar('Accuracy/val', accuracy, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)

        string='Epoch {}:{} loss: {} Accuracy : {} \t'.format(epoch,args.epochs,running_loss,epoch_acc)
        string+='Eval Accuracy : {}\n'.format(accuracy)
        file_train.write(string)
        print('{} Loss: {:.4f} Acc: {:.4f}'.format('train', epoch_loss, epoch_acc))
        torch.save({
                    'epoch': epoch,
                    'model_state_dict': fusion_network.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': epoch_loss,
                    }, PATH)
        if (epoch +1)% 5 ==0:
          save_path = model_path+'/vgg_ft_2048{}.pth'.format(epoch+1)
          torch.save(fusion_network, save_path)
          print ("model saved")
        print("checkpoint saved")
    writer.flush() 
    file_train.close()
    writer.close()

def test(args):
    #extract weights from the weight matrices
    weights=np.load(args.file_name)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #defining dictionary and VQAFeatureDataset
    #transforms for pretrained network(transform for resnet now)
    test_transform = transforms.Compose([ 
        transforms.Resize((args.crop_size,args.crop_size)),
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])

    
    dictionary = Dictionary.load_from_file('/home/ok_sikha/abhishek/VisualQuestion_VQA/data/dictionary.pkl')
    # train_rcnn_pickle_file="/home/ok_sikha/abhishek/VisualQuestion_VQA/data/train36_imgid2idx.pkl"
    # val_rcnn_pickle_file="/home/ok_sikha/abhishek/VisualQuestion_VQA/data/val36_imgid2idx.pkl"
    test_dataset = VQADataset(image_root_dir=args.img_root_dir,feats_data_path = args.feat_root_dir,dictionary=dictionary,dataroot=args.data_root_dir,choice='test',transform_set=test_transform)

    #model definition 
    print('Loading the models')
    # image_encoder=EncoderCNN(embed_size=args.img_feats).to(device)
    # question_encoder=EncoderLSTM(hidden_size=args.num_hid,weights_matrix=weights,fc_size=args.q_embed,max_seq_length=args.max_sequence_length,batch_size=args.batch_size).to(device)
    # fusion_network=FusionModule(qnetwork=question_encoder,img_network=image_encoder,fuse_embed_size=args.fuse_embed,input_fc_size=args.img_feats,class_size=args.num_class).to(device)
    # print(list(fusion_network.parameters()))
    # print(fusion_network)
    # input()
    fusion_network = torch.load(args.test_model_path)

    

    #Dataloader initialization
    test_loader = DataLoader(test_dataset, args.batch_size, shuffle=True, num_workers=4)

    # Train the models
    total_step = len(test_loader)
    step=0
    #Training starts
    print('testing Starting ......................')
    # PATH = "/home/ok_sikha/abhishek/VisualQuestion_VQA/model_saved/model_frcnn_vgg_without_resnet.pt"
    # model_path = args.data_root_dir + '/models'
    # if os.path.exists(PATH):
    #   checkpoint = torch.load(PATH)
    #   fusion_network.load_state_dict(checkpoint['model_state_dict'])
    #   optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #   epochs = checkpoint['epoch']
    #   loss = checkpoint['loss']
    #   print('Modeled loaded from ', epochs)
    # else:
    #   epochs = 0

    loss=0
    accuracy=0
    criterion = nn.NLLLoss()

    with torch.no_grad():
        for feat, image_sample,question_token,labels in tqdm(test_loader):
            feat,image_sample,question_token,labels = feat.to(device),image_sample.to(device),question_token.to(device),labels.to(device)
            output=fusion_network.forward(question_token,image_sample, feat)
            loss+= criterion(output,labels).item()
            ps = torch.exp(output)
            equality= (labels.data == ps.max(dim=1)[1])
            accuracy+=equality.type(torch.FloatTensor).mean()
    #split path and take the last directory name as the model name 
    model_name=args.test_model_path.split('/')[-1]
    model_name = model_name.split('.')[0]
    file_train=open(os.path.join(args.data_root_dir, args.test_output_file),'a+')
    string='model : {}\tloss: {}\tAccuracy : {} \n'.format(args.test_model_name, loss,accuracy)
    file_train.write(string)
    file_train.close()
    return loss,accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--num_hid', type=int, default=1024)
    parser.add_argument('--crop_size', type=int, default=224 , help='size for randomly cropping images')
    parser.add_argument('--img_root_dir', type=str, default="/content/drive/MyDrive/College_paper/Dataset", help='location of the visual genome images')
    parser.add_argument('--data_root_dir', type=str, default="/content/drive/MyDrive/College_paper/VisualQuestion_VQA/data1", help='location of the associated data')
    parser.add_argument('--model_checkpoint', type=str, default="/content/drive/MyDrive/College_paper/VisualQuestion_VQA/model_saved/model_vgg16_ft_2048.pt")
    parser.add_argument('--model_save_path', type=str, default= "/content/drive/MyDrive/College_paper/VisualQuestion_VQA/data1/models")
    parser.add_argument('--file_name', type=str, default="/content/drive/MyDrive/College_paper/VisualQuestion_VQA/data1/ft_init_300d.npy")
    parser.add_argument('--output', type=str, default='saved_models')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--max_sequence_length', type=int, default=14)
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    parser.add_argument('--q_embed',type=int, default=2048, help='embedding output of the encoder RNN')
    parser.add_argument('--img_feats',type=int, default=2048, help='input feature size of the image space')
    parser.add_argument('--fuse_embed',type=int, default=2048, help='Overall embedding size of the fused network')
    parser.add_argument('--num_class',type=int, default=3344, help='Number of output classes')
    parser.add_argument('--learning_rate',type=float,default=0.0001,help='Learning rate')

    
    # Testing parameters
    # parser.add_argument('--test_batch_size', type=int, default=128)
    # parser.add_argument('--test_max_sequence_length', type=int, default=14)
    # parser.add_argument('--test_img_root_dir', type=str, default="/home/ok_sikha/abhishek/VG_100K", help='location of the visual genome images')
    # parser.add_argument('--test_feat_root_dir', type=str, default="/home/ok_sikha/abhishek/VisualQuestion_VQA/data", help='location of the frcnn features visual genome images')
    parser.add_argument('--test_output_file', type=str, default="frcnn_vgg_without_resnet.txt")
    parser.add_argument('--test_model_path', type=str, default='/content/drive/MyDrive/College_paper/VisualQuestion_VQA/data1/models/frcnn_vgg_without_resnet75.pth')
    parser.add_argument('--test_seed', type=int, default=1111, help='random seed')

    args = parser.parse_args()
    main(args)
    test(args)