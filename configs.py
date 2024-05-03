#!/usr/bin/env python3

import os

SPECIAL_TOKENS = ['<|triple|>', '<|target|>', '<|endoftext|>']

LEARNING_RATE = 1e-6
EPOCHS = 50
BATCH_SIZE = 8

# en OR ru
LANG = 'en'

# train, validation, test,
# challenge_train_sample, challenge_validation_sample
# challenge_test_scramble, challenge_test_numbers
TRAIN_SPLIT = 'train'
VALID_SPLIT = 'validation'
TEST_SPLIT = 'test'

# Set None for full dataset
DATASET_SIZE = None

MODEL_NAME = "model_state.pt"
MODEL_PATH = os.path.join(os.curdir, 'model_10_49', 'gpt2_webnlg_epoch49')
MODEL_PATH_FOLDER = 'model_'

