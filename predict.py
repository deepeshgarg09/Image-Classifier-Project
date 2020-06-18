#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow_hub as hub
from PIL import Image
import tensorflow as tf

import numpy as np
import argparse
import json

image_size = 224
image_shape = (image_size,image_size, 3)

def class__names(json_file):
    with open(json_file, 'r') as k:
        class_names = json.load(k)
    #remapping
    revised_class_names = dict()
    for key in class_names:
        revised_class_names[str(int(key)-1)] = class_names[key]
    return revised_class_names


def load_saved_model(model_path):
    model = tf.keras.models.load_model(model_path, custom_objects={'KerasLayer':hub.KerasLayer})
    print(model.summary())
    return model

def process_image(np_image):
    ts_img = tf.image.convert_image_dtype(np_image, dtype=tf.int16, saturate=False)
    rs_img = tf.image.resize(np_image,(image_size,image_size)).numpy()
    normal_img = rs_img/255
    return normal_img

def predict(image_path, model_path, top_k, all_class_names):
    top_k = int(top_k)
    model = load_saved_model(model_path)
    img = Image.open(image_path)
    test_image = np.asarray(img)
    p_t_image = process_image(test_image)
    prob_preds = model.predict(np.expand_dims(p_t_image,axis=0))
    prob_preds = prob_preds[0].tolist()
    top_pred_class_id = model.predict_classes(np.expand_dims(p_t_image,axis=0))
    top_pred_class_prob = prob_preds[top_pred_class_id[0]]
    pred_class = all_class_names[str(top_pred_class_id[0])]
    print("\nTop Predicted class image and it's probability is :\n","class_id is:",top_pred_class_id, "class_name is :", pred_class, "; class_probability :",top_pred_class_prob)
    values, indices= tf.math.top_k(prob_preds, k=top_k)
    probs_topk = values.numpy().tolist()
    classes_topk = indices.numpy().tolist()
    print("Top k classes are:",classes_topk)
    print("Top k probabilities are:",probs_topk)
    class_labels = [all_class_names[str(i)] for i in classes_topk]
    print('Top k class labels are:',class_labels)
    class_prob_dict = dict(zip(class_labels, probs_topk))       
    print("\nTop K classes along with associated probabilities are :\n\n",class_prob_dict)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = "Description for my parser")
    parser.add_argument("image_path",help="Image Path", default="")
    parser.add_argument("saved_model",help="Model Path", default="")
    parser.add_argument("--top_k", help="Fetch top k predictions", required = False, default = 3)
    parser.add_argument("--category_names", help="Class map json file", required = False, default = "label_map.json")
    args = parser.parse_args()
    all_class_names = get_class_names(args.category_names)
    predict(args.image_path, args.saved_model, args.top_k, all_class_names)

