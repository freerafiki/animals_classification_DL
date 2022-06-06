# the code we added for our dataset
import os, random
import numpy as np
import cv2
import pdb

def load_random_animals(dataset_folder, max_classes=2, img_size=(64,64), train_test_split=0.8):

    classes = []
    train_set_x_orig = []
    train_set_y_orig = []
    test_set_x_orig = []
    test_set_y_orig = []

    # images will be read from the folders
    categories_folder = os.listdir(dataset_folder)
    random.shuffle(categories_folder)
    cat_index = 0
    for category in categories_folder:
        if os.path.isdir(os.path.join(dataset_folder, category)) and cat_index < max_classes:
            classes.append(category)
            image_names = os.listdir(os.path.join(dataset_folder, category))
            random.shuffle(image_names) # shuffle to keep it random

            # divide into training and test images
            split_num = np.round(len(image_names) * train_test_split).astype(int)
            training_images_names = image_names[:split_num]
            test_images_names = image_names[split_num:]

            # read and append training images
            for training_image_name in training_images_names:
                img = cv2.imread(os.path.join(dataset_folder, category, training_image_name))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                data_img = cv2.resize(img, img_size)
                train_set_x_orig.append((data_img))
                train_set_y_orig.append(cat_index) # cat_index is the 'label', the correct category

            # read and append test images
            for test_image_name in test_images_names:
                img = cv2.imread(os.path.join(dataset_folder, category, test_image_name))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                data_img = cv2.resize(img, img_size)
                test_set_x_orig.append((data_img))
                test_set_y_orig.append(cat_index) # cat_index is the 'label', the correct category

            print(f'found {len(image_names)} images of {category}')
            cat_index += 1

    print(f"finished with {len(train_set_x_orig)} training and {len(test_set_x_orig)} images of {cat_index} categories")
    train_set_x_orig = np.array(train_set_x_orig)
    train_set_y_orig = np.array(train_set_y_orig)
    test_set_x_orig = np.array(test_set_x_orig)
    test_set_y_orig = np.array(test_set_y_orig)
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes
