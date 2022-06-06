import os, pdb, random, shutil
import numpy as np
folder = '/home/palma/Unive/RePAIR/Datasets/RePAIR_dataset/group_1/raw/frames'
training_folder = '/home/palma/Unive/RePAIR/Datasets/RePAIR_dataset/classification/group_1/train'
test_folder = '/home/palma/Unive/RePAIR/Datasets/RePAIR_dataset/classification/group_1/test'
train_test_split = 0.85
for frame_folder in os.listdir(folder):
    print(f'fragment {frame_folder}')
    frames = os.listdir(os.path.join(folder, frame_folder))
    random.shuffle(frames)
    split = np.round(len(frames) * train_test_split).astype(int)
    fragment_training_folder = os.path.join(training_folder, frame_folder)
    if not os.path.exists(fragment_training_folder):
        os.mkdir(fragment_training_folder)
    training_frames = frames[:split]
    for training_frame in training_frames:
        shutil.copy(os.path.join(folder, frame_folder, training_frame), fragment_training_folder)
    fragment_test_folder = os.path.join(test_folder, frame_folder)
    if not os.path.exists(fragment_test_folder):
        os.mkdir(fragment_test_folder)
        test_frames = frames[split:]
    for test_frame in test_frames:
        shutil.copy(os.path.join(folder, frame_folder, test_frame), fragment_test_folder)
