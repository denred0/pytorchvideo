import os
from pathlib import Path
import shutil
from tqdm import tqdm
import pickle

from sklearn.preprocessing import LabelEncoder


def train_val_split_videos(root_directory, exstention, val_part):
    """create folder dataset/train with all videos need to split"""

    val_folder = 'val'
    val_root = os.path.join(root_directory, val_folder)
    Path(val_root).mkdir(parents=True, exist_ok=True)

    train_folder = 'train'
    train_root = os.path.join(root_directory, train_folder)

    train_list = []
    val_list = []

    folder_names = os.listdir(train_root)

    label_encoder = LabelEncoder()
    encoded = label_encoder.fit_transform(folder_names)

    # save labels dict to file
    with open(Path(root_directory).joinpath('label_encoder.pkl'), 'wb') as le_dump_file:
        pickle.dump(label_encoder, le_dump_file)

    # save labels annotations in understandable format
    with open(Path(root_directory).joinpath('labels_annotations.txt'), "w") as text_file:
        for id, label in enumerate(label_encoder.classes_):
            record = str(id) + ' ' + label
            print(record, file=text_file)

    for i, (subdir, dirs, files) in tqdm(enumerate(os.walk(train_root))):
        for folder in dirs:

            # create folder for val with folder class
            Path(os.path.join(val_root, folder)).mkdir(parents=True, exist_ok=True)

            label = label_encoder.transform([folder]).item()

            for i, filename in enumerate(os.listdir(os.path.join(subdir, folder))):
                name, file_extension = os.path.splitext(filename)
                if file_extension in exstention:
                    if i % int(1 / val_part) == 0:
                        shutil.move(os.path.join(subdir, folder, filename),
                                    os.path.join(val_root, folder, filename))
                        val_list.append(os.path.join(val_root, folder, filename) + ' ' + str(label))
                    else:
                        train_list.append(os.path.join(train_root, folder, filename) + ' ' + str(label))

    print(f"Train files: {len(train_list)}")
    print(f"Val files: {len(val_list)}")

    with open(os.path.join(root_directory, 'train.csv'), 'w') as f:
        # f.write('image_name,PredString,domain\n')
        for item in train_list:
            f.write("%s\n" % item)

    with open(os.path.join(root_directory, 'val.csv'), 'w') as f:
        # f.write('image_name,PredString,domain\n')
        for item in val_list:
            f.write("%s\n" % item)


def train_val_split_images(root_directory, val_part):
    """create folder dataset/train with all videos need to split"""

    val_folder = 'val'
    val_root = os.path.join(root_directory, val_folder)
    Path(val_root).mkdir(parents=True, exist_ok=True)

    train_folder = 'train'
    train_root = os.path.join(root_directory, train_folder)
    Path(train_root).mkdir(parents=True, exist_ok=True)

    data_root = os.path.join(root_directory, 'data')

    train_list = []
    val_list = []

    folder_names = os.listdir(data_root)

    label_encoder = LabelEncoder()
    encoded = label_encoder.fit_transform(folder_names)

    # save labels dict to file
    with open(Path(root_directory).joinpath('label_encoder.pkl'), 'wb') as le_dump_file:
        pickle.dump(label_encoder, le_dump_file)

    # save labels annotations in understandable format
    with open(Path(root_directory).joinpath('labels_annotations.txt'), "w") as text_file:
        for id, label in enumerate(label_encoder.classes_):
            record = str(id) + ' ' + label
            print(record, file=text_file)

    for i, (subdir, dirs, files) in tqdm(enumerate(os.walk(data_root))):
        for folder in dirs:

            # create folder for val with folder class
            Path(os.path.join(val_root, folder)).mkdir(parents=True, exist_ok=True)
            Path(os.path.join(train_root, folder)).mkdir(parents=True, exist_ok=True)

            label = label_encoder.transform([folder]).item()

            for i, fold in enumerate(os.listdir(os.path.join(subdir, folder))):
                if i % int(1 / val_part) == 0:
                    shutil.move(os.path.join(data_root, folder, fold), os.path.join(val_root, folder, fold))
                    val_list.append(os.path.join(val_root, folder, fold) + ' ' + str(label))
                else:
                    shutil.move(os.path.join(data_root, folder, fold), os.path.join(train_root, folder, fold))
                    train_list.append(os.path.join(train_root, folder, fold) + ' ' + str(label))

    print(f"Train files: {len(train_list)}")
    print(f"Val files: {len(val_list)}")

    with open(os.path.join(root_directory, 'train.csv'), 'w') as f:
        # f.write('image_name,PredString,domain\n')
        for item in train_list:
            f.write("%s\n" % item)

    with open(os.path.join(root_directory, 'val.csv'), 'w') as f:
        # f.write('image_name,PredString,domain\n')
        for item in val_list:
            f.write("%s\n" % item)


# def delete_previous_dataset(root_dir):
#     shutil.rmtree(root_dir)
#     Path(root_dir).mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    dataset_folder = 'dataset'
    root_dir = os.path.join('data', dataset_folder)
    video_ext = ['.avi']
    val_part = 0.2

    mode = "videos"

    if mode == 'videos':
        train_val_split_videos(root_directory=root_dir, exstention=video_ext, val_part=val_part)
    else:
        train_val_split_images(root_directory=root_dir, val_part=val_part)
