import os
import glob
import torch
import numpy as np 

from model import Captcha
from sklearn import preprocessing
from sklearn import model_selection
from sklearn import metrics

import config
import dataset
import glob
import engine


def run_training():

    image_files =  glob.glob(os.path.join(config.DATA_DIR, "*.png"))
    targets = [x.split("/")[-1][:-4] for x in image_files]
    targets = [[c for c in x] for x in targets]
    targets_flat = [c for clist in targets for c in clist]

    labl_enc = preprocessing.LabelEncoder()
    labl_enc.fit(targets_flat)
    
    #print(labl_enc)
    #print(labl_enc.classes_)
    #print(len(labl_enc.classes_))

    targets_enc = [labl_enc.transform(x) for x in targets]
    targets_enc = np.array(targets_enc)+1
    print(targets_enc)

    train_imgs, test_imgs, train_targets, test_targets, train_orig_targets, test_orig_targets = model_selection.train_test_split(image_files, targets_enc, targets, test_size=0.1, random_state=42)

    train_dataset = dataset.CaptchaClassification(train_imgs, train_targets, resize=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH))

    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)

    test_dataset = dataset.CaptchaClassification(test_imgs, test_targets, resize=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH))

    test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS)

    model = Captcha(num_classes=len(labl_enc.classes_))

    model.to(config.DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.8, patience=5, verbose=True
    )
    for epoch in range(config.EPOCHS):
        train_loss = engine.train_fn(model, train_loader, optimizer)
        valid_preds, valid_loss = engine.eval_fn(model, test_loader)
        print("Epoch:{} , train_loss = {}, valid_loss= {}".format(epoch, train_loss, valid_loss))


if __name__=="__main__":
    run_training()