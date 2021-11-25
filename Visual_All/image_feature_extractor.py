import csv
import sys
import h5py
import numpy as np
import utils
import _pickle as cPickle
import os
import base64
from tqdm import tqdm
FIELDNAMES = ['image_id', 'image_h', 'image_w', "objects_id", "objects_conf", "attrs_id", "attrs_conf", 'num_boxes', 'boxes', 'features']
feature_length = 2048
num_fixed_boxes = 36


csv.field_size_limit(sys.maxsize)

def image_feats_converter(filenames):


    h_train = h5py.File(filenames['train_data_file'], "w")
    h_val = h5py.File(filenames['val_data_file'], "w")


    if os.path.exists(filenames['train_ids_file']) and os.path.exists(filenames['val_ids_file']):
        print(filenames['train_ids_file'])
        print(filenames['val_ids_file'])
        train_imgids = cPickle.load(open(filenames['train_ids_file'],'rb')).tolist()
        val_imgids = cPickle.load(open(filenames['val_ids_file'],'rb')).tolist()
    else:
        with open('/home/ok_sikha/abhishek/VisualQuestion_VQA/data/train_imgids.pkl', 'rb') as f:
            train_imgids = cPickle.load(f).tolist()
        with open('/home/ok_sikha/abhishek/VisualQuestion_VQA/data/val_imgids.pkl', 'rb') as f:
            val_imgids = cPickle.load(f).tolist()
        cPickle.dump(train_imgids, open(filenames['train_ids_file'], 'wb'))
        cPickle.dump(val_imgids, open(filenames['val_ids_file'], 'wb'))

    train_indices = {}
    val_indices = {}

    train_img_features = h_train.create_dataset(
        'image_features', (len(train_imgids), num_fixed_boxes, feature_length), 'f')
    train_img_bb = h_train.create_dataset(
        'image_bb', (len(train_imgids), num_fixed_boxes, 4), 'f')
    train_spatial_img_features = h_train.create_dataset(
        'spatial_features', (len(train_imgids), num_fixed_boxes, 6), 'f')

    val_img_bb = h_val.create_dataset(
        'image_bb', (len(val_imgids), num_fixed_boxes, 4), 'f')
    val_img_features = h_val.create_dataset(
        'image_features', (len(val_imgids), num_fixed_boxes, feature_length), 'f')
    val_spatial_img_features = h_val.create_dataset(
        'spatial_features', (len(val_imgids), num_fixed_boxes, 6), 'f')

    train_counter = 0
    val_counter = 0
    notfound = []
    print("reading tsv...")
    with open(filenames['infile'], "r") as tsv_in_file:
        reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames=FIELDNAMES)
        for item in tqdm(reader):
            item['num_boxes'] = int(item['num_boxes'])
            try:
                image_id = int(item['image_id'])
            except:
                continue
            image_w = float(item['image_w'])
            image_h = float(item['image_h'])
            bboxes = np.frombuffer(
                base64.b64decode(item['boxes']),
                dtype=np.float32).reshape((item['num_boxes'], -1))

            box_width = bboxes[:, 2] - bboxes[:, 0]
            box_height = bboxes[:, 3] - bboxes[:, 1]
            scaled_width = box_width / image_w
            scaled_height = box_height / image_h
            scaled_x = bboxes[:, 0] / image_w
            scaled_y = bboxes[:, 1] / image_h

            box_width = box_width[..., np.newaxis]
            box_height = box_height[..., np.newaxis]
            scaled_width = scaled_width[..., np.newaxis]
            scaled_height = scaled_height[..., np.newaxis]
            scaled_x = scaled_x[..., np.newaxis]
            scaled_y = scaled_y[..., np.newaxis]

            spatial_features = np.concatenate(
                    (scaled_x,
                    scaled_y,
                    scaled_x + scaled_width,
                    scaled_y + scaled_height,
                    scaled_width,
                    scaled_height),
                    axis=1)
            flag= False
            if image_id in train_imgids:
                train_imgids.remove(image_id)
                train_indices[image_id] = train_counter
                train_img_bb[train_counter, :, :] = bboxes
                train_img_features[train_counter, :, :] = np.frombuffer(
                    base64.b64decode(item['features']),
                    dtype=np.float32).reshape((item['num_boxes'], -1))
                train_spatial_img_features[train_counter, :, :] = spatial_features
                train_counter += 1
                flag = True
            if image_id in val_imgids:
                val_imgids.remove(image_id)
                val_indices[image_id] = val_counter
                val_img_bb[val_counter, :, :] = bboxes
                val_img_features[val_counter, :, :] = np.frombuffer(
                    base64.b64decode(item['features']),
                    dtype=np.float32).reshape((item['num_boxes'], -1))
                val_spatial_img_features[val_counter, :, :] = spatial_features
                val_counter += 1
                flag = True
            if not flag:
                notfound.append(image_id)
                continue

    if len(train_imgids) != 0:
        print('Warning: train_image_ids is not empty')

    if len(val_imgids) != 0:
        print('Warning: val_image_ids is not empty')

    if len(notfound) != 0:
        print(notfound)
        print('Warning: %d images not found' % len(notfound))

    cPickle.dump(train_indices, open(filenames['train_indices_file'], 'wb'))
    cPickle.dump(val_indices, open(filenames['val_indices_file'], 'wb'))
    h_train.close()
    h_val.close()
    print("done!")




