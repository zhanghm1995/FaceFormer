'''
Copyright (c) 2022 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2022-02-12 17:29:02
Email: haimingzhang@link.cuhk.edu.cn
Description: Test the VOCADataset class
'''
import os
import shutil
import configparser
from dataset.voca_dataset import DataHandler, Batcher
from config_parser import read_config, create_default_config

def test_voca_dataset(config):
    data_handler = DataHandler(config)
    batcher = Batcher(data_handler)

    num_train_batches = batcher.get_num_batches(config['batch_size'])

    processed_audio, face_vertices, face_templates, subject_idx = batcher.get_training_batch(config['batch_size'])
    print(processed_audio.shape, face_vertices.shape, face_templates.shape, subject_idx.shape)


def main():
    pkg_path, _ = os.path.split(os.path.realpath(__file__))
    init_config_fname = os.path.join(pkg_path, 'training_config.cfg')
    if not os.path.exists(init_config_fname):
        print('Config not found %s' % init_config_fname)
        create_default_config(init_config_fname)

    config = configparser.RawConfigParser()
    config.read(init_config_fname)

    # Path to cache the processed audio
    config.set('Input Output', 'processed_audio_path',
               './training_data/processed_audio_%s.pkl' % config.get('Audio Parameters', 'audio_feature_type'))

    checkpoint_dir = config.get('Input Output', 'checkpoint_dir')
    if os.path.exists(checkpoint_dir):
        print('Checkpoint dir already exists %s' % checkpoint_dir)
        key = input('Press "q" to quit, "x" to erase existing folder, and any other key to continue training: ')
        if key.lower() == 'q':
            return
        elif key.lower() == 'x':
            try:
                shutil.rmtree(checkpoint_dir, ignore_errors=True)
            except:
                print('Failed deleting checkpoint directory')

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    config_fname = os.path.join(checkpoint_dir, 'config.pkl')
    if os.path.exists(config_fname):
        print('Use existing config %s' % config_fname)
    else:
        with open(config_fname, 'w') as fp:
            config.write(fp)
            fp.close()

    config = read_config(config_fname)

    test_voca_dataset(config)


if __name__ == "__main__":
    main()
