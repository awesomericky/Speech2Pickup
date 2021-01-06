### Import Dependencies
import json
import numpy as np
from os import listdir, remove, makedirs
from os.path import isfile, join, isdir
from scipy.stats import multivariate_normal as mulnorm
from utils import load_w2v

def preprocess_text2pickup(start_point, only_test_data=False):

    # Generate heatmap and instruction vectors

    ## Google Colab

    json_train_path = '/content/drive/MyDrive/Speech2Pickup/dataset2/en.train.jsonl'
    json_test_path = '/content/drive/MyDrive/Speech2Pickup/dataset2/en.validation.jsonl'
    word2vec_path = '/content/drive/MyDrive/Speech2Pickup/GoogleNews-vectors-negative300.bin'
    total_word_path = '/content/drive/MyDrive/Speech2Pickup/dataset2/total_words.npz'
    heatmap_train_path = '/content/drive/MyDrive/Speech2Pickup/dataset2/heatmap/train'
    heatmap_test_path = '/content/drive/MyDrive/Speech2Pickup/dataset2/heatmap/test'
    total_data_train_path = '/content/drive/MyDrive/Speech2Pickup/dataset2/text2pickup_npz_train'
    total_data_test_path = '/content/drive/MyDrive/Speech2Pickup/dataset2/text2pickup_npz_test/preprocessed4text2pickup_test.npz'


    ## Local PC

    # json_train_path = './data/dataset2/en.train.jsonl'
    # json_test_path = './data/dataset2/en.validation.jsonl'
    # word2vec_path = './data/GoogleNews-vectors-negative300.bin'
    # total_word_path = './data/dataset2/total_words.npz'
    # heatmap_train_path = './data/dataset2/heatmap/train'
    # heatmap_test_path = './data/dataset2/heatmap/test'
    # total_data_train_path = './data/dataset2'
    # total_data_test_path = './data/dataset2/preprocessed4text2pickup_test.npz'


    en_train_objs = []
    en_val_objs = []

    with open(json_train_path) as f:
        for jsonobj in f:
            en_train_obj = json.loads(jsonobj)
            en_train_objs.append(en_train_obj)

    with open(json_test_path) as f:
        for jsonobj in f:
            en_val_obj = json.loads(jsonobj)
            en_val_objs.append(en_val_obj)

    save_freq = len(en_train_objs)//232
    save_point = [save_freq*i for i in range(232)]
    save_point.append(len(en_train_objs))
    save_point_data = [0]

    ### Import word2vec model
    w2v_path = word2vec_path
    w2v_model = load_w2v(w2v_path)

    dim_embed = w2v_model['woman'].shape[0]

    ### Calculate Maximum length of the language command
    total_words = set()
    max_len = 0
    num_train_data = 0
    num_val_data = 0
    num_data = 0
    checkpoint = 1

    for en_train_obj in en_train_objs:
        des_objects = en_train_obj["objects"]
        for des_object in des_objects:
            des_instructions = des_object["instructions"]
            for des_instruction in des_instructions:
                words = des_instruction.split()
                for word in words:
                    total_words.add(word)
                if max_len < len(words):
                    max_len = len(words)
            num_train_data += len(des_instructions)
        if en_train_objs.index(en_train_obj)+1 == save_point[checkpoint]:
            save_point_data.append(num_train_data)
            checkpoint += 1
    
    for en_val_obj in en_val_objs:
        des_objects = en_val_obj["objects"]
        for des_object in des_objects:
            des_instructions = des_object["instructions"]
            for des_instruction in des_instructions:
                words = des_instruction.split()
                for word in words:
                    total_words.add(word)
                if max_len < len(words):
                    max_len = len(words)
            num_val_data += len(des_instructions)
    
    num_data = num_train_data + num_val_data
    total_words = list(total_words)

    # Save total words
    save_file_name = total_word_path
    np.savez_compressed(save_file_name, words=total_words)

    max_len = max_len + 5

    print('Total data number: %d' % num_data)
    print('Total train data number: %d' % num_train_data)
    print('Total test data number: %d' % num_val_data)

    print('Now processing Train data ..')
    start_point = start_point//5
    for i in range(start_point, len(save_point_data)-1):
        num_save_data = save_point_data[i+1] - save_point_data[i]

        ### Set Empty data arrays
        inputs = np.zeros((num_save_data, dim_embed, max_len), dtype=np.float32)
        outputs = np.zeros((num_save_data, 4), dtype=np.float32)  # (x, y, width, height)
        img_idx = np.zeros((num_save_data, 1), dtype=np.float32)
        seq_len = np.zeros((num_save_data, 1), dtype=np.float32)
        tmp_num = 0

        for en_train_obj in en_train_objs[save_point[i]:save_point[i+1]]:
            print('Processing {}/{}'.format(en_train_objs.index(en_train_obj)+1, len(en_train_objs)))
            des_objects = en_train_obj["objects"]
            for des_object in des_objects:
                prev_output = [0, 0, 0, 0]
                des_instructions = des_object["instructions"]
                for des_instruction in des_instructions:
                    words = des_instruction.split()
                    for w, word in enumerate(words):
                        if word not in w2v_model.vocab.keys():
                            inputs[tmp_num, :, w] = np.zeros((dim_embed,))
                        else:
                            inputs[tmp_num, :, w] = w2v_model[word]

                    img_idx[tmp_num, 0] = float('%04d' % int(en_train_obj["image_file"].split('.')[0]))
                    outputs[tmp_num, 0] = float(round(des_object["bbox"]["x"]))
                    outputs[tmp_num, 1] = float(round(des_object["bbox"]["y"]))
                    outputs[tmp_num, 2] = float(round(des_object["bbox"]["width"]))
                    outputs[tmp_num, 3] = float(round(des_object["bbox"]["height"]))
                    seq_len[tmp_num, 0] = len(words)

                    if prev_output[0] != outputs[tmp_num, 0] or prev_output[1] != outputs[tmp_num, 1] or prev_output[2] != outputs[tmp_num, 2] or prev_output[3] != outputs[tmp_num, 3]:
                        x = np.array(range(1024))
                        y = np.array(range(1280))
                        xx, yy = np.meshgrid(x, y)
                        xxyy = np.c_[xx.T.ravel(), yy.T.ravel()]

                        tmp_output = outputs[tmp_num, :].astype(int)
                        tmp_cov = [[1000, 0], [0, 1000]]
                        mvn = mulnorm([tmp_output[1], tmp_output[0]], tmp_cov)

                        tmp_heatmap = mvn.pdf(xxyy)
                        tmp_heatmap = tmp_heatmap / np.max(tmp_heatmap)
                        tmp_heatmap = tmp_heatmap.reshape((len(x), len(y)))

                        indi_name = ('/%04d_%04d_%04d_%04d_%04d.npz')% (img_idx[tmp_num, 0], outputs[tmp_num, 0], outputs[tmp_num, 1], outputs[tmp_num, 2], outputs[tmp_num, 3])
                        heatmap_train_indi_path = heatmap_train_path + indi_name
                        np.savez_compressed(heatmap_train_indi_path, heatmap=tmp_heatmap)

                    prev_output = outputs[tmp_num, :]
                    tmp_num += 1
                    
        ### Save preprocessed data
        total_data_train_indi_path = total_data_train_path + '/preprocessed4text2pickup_train_{}.npz'.format(save_point[i+1])
        save_file_name = total_data_train_indi_path
        np.savez_compressed(save_file_name, img_idx=img_idx, seq_len=seq_len, inputs=inputs, outputs=outputs)
    

    ### Set Empty data arrays
    inputs = np.zeros((num_val_data, dim_embed, max_len), dtype=np.float32)
    outputs = np.zeros((num_val_data, 4), dtype=np.float32)  # (x, y, width, height)
    img_idx = np.zeros((num_val_data, 1), dtype=np.float32)
    seq_len = np.zeros((num_val_data, 1), dtype=np.float32)
    tmp_num = 0

    print('Now processing Test data ..')
    for en_val_obj in en_val_objs:
        print('Processing {}/{}'.format(en_val_objs.index(en_val_obj)+1, len(en_val_objs)))
        des_objects = en_val_obj["objects"]
        for des_object in des_objects:
            prev_output = [0, 0, 0, 0]
            des_instructions = des_object["instructions"]
            for des_instruction in des_instructions:
                words = des_instruction.split()
                for w, word in enumerate(words):
                    if word not in w2v_model.vocab.keys():
                        inputs[tmp_num, :, w] = np.zeros((dim_embed,))
                    else:
                        inputs[tmp_num, :, w] = w2v_model[word]

                img_idx[tmp_num, 0] = float('%04d' % int(en_val_obj["image_file"].split('.')[0]))
                outputs[tmp_num, 0] = float(round(des_object["bbox"]["x"]))
                outputs[tmp_num, 1] = float(round(des_object["bbox"]["y"]))
                outputs[tmp_num, 2] = float(round(des_object["bbox"]["width"]))
                outputs[tmp_num, 3] = float(round(des_object["bbox"]["height"]))
                seq_len[tmp_num, 0] = len(words)

                if prev_output[0] != outputs[tmp_num, 0] or prev_output[1] != outputs[tmp_num, 1] or prev_output[2] != outputs[tmp_num, 2] or prev_output[3] != outputs[tmp_num, 3]:
                    x = np.array(range(1024))
                    y = np.array(range(1280))
                    xx, yy = np.meshgrid(x, y)
                    xxyy = np.c_[xx.T.ravel(), yy.T.ravel()]

                    tmp_output = outputs[tmp_num, :].astype(int)
                    tmp_cov = [[1000, 0], [0, 1000]]
                    mvn = mulnorm([tmp_output[1], tmp_output[0]], tmp_cov)

                    tmp_heatmap = mvn.pdf(xxyy)
                    tmp_heatmap = tmp_heatmap / np.max(tmp_heatmap)
                    tmp_heatmap = tmp_heatmap.reshape((len(x), len(y)))

                    indi_name = ('/%04d_%04d_%04d_%04d_%04d.npz')% (img_idx[tmp_num, 0], outputs[tmp_num, 0], outputs[tmp_num, 1], outputs[tmp_num, 2], outputs[tmp_num, 3])
                    heatmap_test_indi_path = heatmap_test_path + indi_name
                    np.savez_compressed(heatmap_test_indi_path, heatmap=tmp_heatmap)

                prev_output = outputs[tmp_num, :]
                tmp_num += 1
    
    ### Save preprocessed data
    save_file_name = total_data_test_path
    np.savez_compressed(save_file_name, img_idx=img_idx, seq_len=seq_len, inputs=inputs, outputs=outputs)

# preprocess_text2pickup(start_point=610, only_test_data=False)

# ## Local PC

# json_train_path = './data/dataset2/en.train.jsonl'
# json_test_path = './data/dataset2/en.validation.jsonl'
# word2vec_path = './data/GoogleNews-vectors-negative300.bin'
# total_word_path = './data/dataset2/total_words.npz'
# heatmap_train_path = './data/dataset2/heatmap/train'
# heatmap_test_path = './data/dataset2/heatmap/test'
# total_data_train_path = './data/dataset2'
# total_data_test_path = './data/dataset2/preprocessed4text2pickup_test.npz'


# en_train_objs = []
# en_val_objs = []

# with open(json_train_path) as f:
#     for jsonobj in f:
#         en_train_obj = json.loads(jsonobj)
#         en_train_objs.append(en_train_obj)

# with open(json_test_path) as f:
#     for jsonobj in f:
#         en_val_obj = json.loads(jsonobj)
#         en_val_objs.append(en_val_obj)

# ### Calculate Maximum length of the language command
# train_words = set()
# test_words = set()
# max_len = 0
# num_train_data = 0
# num_val_data = 0
# num_data = 0
# checkpoint = 1

# for en_train_obj in en_train_objs:
#     des_objects = en_train_obj["objects"]
#     for des_object in des_objects:
#         des_instructions = des_object["instructions"]
#         for des_instruction in des_instructions:
#             words = des_instruction.split()
#             for word in words:
#                 train_words.add(word)
#             if max_len < len(words):
#                 max_len = len(words)
#         num_train_data += len(des_instructions)

# for en_val_obj in en_val_objs:
#     des_objects = en_val_obj["objects"]
#     for des_object in des_objects:
#         des_instructions = des_object["instructions"]
#         for des_instruction in des_instructions:
#             words = des_instruction.split()
#             for word in words:
#                 test_words.add(word)
#             if max_len < len(words):
#                 max_len = len(words)
#         num_val_data += len(des_instructions)

# num_data = num_train_data + num_val_data
# train_words = list(train_words)
# test_words = list(test_words)
# n = 0

# for word in test_words:
#     if word not in train_words:
#         print(word)
#         n += 1
# print(len(train_words))
# print(len(test_words))
# print(n)
# print(test_words)