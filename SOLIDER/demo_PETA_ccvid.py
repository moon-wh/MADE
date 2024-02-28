import argparse
import glob
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from dataset.augmentation import get_transform
from models.backbone.swin_transformer import *
from models.model_factory import build_backbone, build_classifier

import torch
from torch.utils.data import Dataset, DataLoader
import os.path as osp
from PIL import Image
from configs import cfg, update_config

from models.base_block import FeatClassifier
from tools.function import get_model_log_path, get_reload_weight
from tools.utils import set_seed, str2bool
from tqdm import tqdm
import random
set_seed(605)

class CCVIDLoader(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.img_dirs = []
        for session in ['session1', 'session2', 'session3']:
            pdirs = glob.glob(osp.join(root_dir,session,'*'))
            pdirs.sort()
            for pdir in pdirs:
                self.img_dirs.append(pdir)
                # pid = int(osp.basename(pdir))
                # img_dirs = glob.glob(osp.join(pdir, '*.jpg'))
                # for img_dir in img_dirs:
                #     self.imgs.append(img_dir)
    
    def __len__(self):
        return len(self.img_dirs)
    
    def __getitem__(self, idx):
        tracklet_dir = self.img_dirs[idx]
        img_path = random.choice(glob.glob(osp.join(tracklet_dir, '*.jpg')))
        img = Image.open(img_path)
        img = img.convert("RGB")
        if self.transform:
            img = self.transform(img)
        return (img, tracklet_dir)
        

clas_name = ['accessoryHat','accessoryMuffler','accessoryNothing','accessorySunglasses','hairLong' ,
             'upperBodyCasual', 'upperBodyFormal', 'upperBodyJacket', 'upperBodyLogo', 'upperBodyPlaid', 'upperBodyShortSleeve',
             'upperBodyThinStripes', 'upperBodyTshirt','upperBodyOther','upperBodyVNeck',
             'lowerBodyCasual', 'lowerBodyFormal', 'lowerBodyJeans', 'lowerBodyShorts', 'lowerBodyShortSkirt','lowerBodyTrousers',
             'footwearLeatherShoes', 'footwearSandals', 'footwearShoes', 'footwearSneaker' ,
             'carryingBackpack', 'carryingOther', 'carryingMessengerBag', 'carryingNothing', 'carryingPlasticBags' ,
             'personalLess30','personalLess45','personalLess60','personalLarger60','personalMale',
             'upperBodyBlack', 'upperBodyBlue', 'upperBodyBrown', 'upperBodyGreen', 'upperBodyGrey', 'upperBodyOrange',
             'upperBodyPink', 'upperBodyPurple', 'upperBodyRed', 'upperBodyWhite', 'upperBodyYellow', 'lowerBodyBlack',
             'lowerBodyBlue', 'lowerBodyBrown', 'lowerBodyGreen', 'lowerBodyGrey', 'lowerBodyOrange', 'lowerBodyPink',
             'lowerBodyPurple', 'lowerBodyRed', 'lowerBodyWhite', 'lowerBodyYellow', 'hairBlack', 'hairBlue', 'hairBrown',
             'hairGreen', 'hairGrey', 'hairOrange', 'hairPink', 'hairPurple', 'hairRed', 'hairWhite', 'hairYellow', 'footwearBlack',
             'footwearBlue', 'footwearBrown', 'footwearGreen', 'footwearGrey', 'footwearOrange', 'footwearPink', 'footwearPurple',
             'footwearRed', 'footwearWhite', 'footwearYellow','accessoryHeadphone', 'personalLess15', 'carryingBabyBuggy', 'hairBald',
             'footwearBoots', 'lowerBodyCapri', 'carryingShoppingTro', 'carryingUmbrella', 'personalFemale', 'carryingFolder',
             'accessoryHairBand', 'lowerBodyHotPants', 'accessoryKerchief', 'lowerBodyLongSkirt', 'upperBodyLongSleeve',
             'lowerBodyPlaid', 'lowerBodyThinStripes', 'carryingLuggageCase', 'upperBodyNoSleeve', 'hairShort', 'footwearStocking',
             'upperBodySuit', 'carryingSuitcase', 'lowerBodySuits', 'upperBodySweater', 'upperBodyThickStripes']


def main(cfg, args):
    exp_dir = os.path.join('exp_result', cfg.DATASET.NAME)
    print("--------------- exp_dir", exp_dir)
    model_dir, log_dir = get_model_log_path(exp_dir, cfg.NAME)

    train_tsfm, valid_tsfm = get_transform(cfg)
    print(valid_tsfm)

    backbone, c_output = build_backbone(cfg.BACKBONE.TYPE, cfg.BACKBONE.MULTISCALE)


    classifier = build_classifier(cfg.CLASSIFIER.NAME)(
        # nattr=79,
        nattr=105,
        c_in=c_output,
        bn=cfg.CLASSIFIER.BN,
        pool=cfg.CLASSIFIER.POOLING,
        scale =cfg.CLASSIFIER.SCALE
    )

    model = FeatClassifier(backbone, classifier)

    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()

    model = get_reload_weight(model_dir, model, pth='ckpt_max_2024-02-25_14:41:35.pth')   # Change weights path
    model.eval()

    test_dataset = CCVIDLoader(args.test_img, valid_tsfm)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=1)

    f = open('PAR_PETA_105_ccvid.txt', 'a', encoding='utf-8')

    with torch.no_grad():
        for batch_idx, (inputs, imgpaths) in enumerate(tqdm(test_loader)):
            inputs = inputs.cuda()
            # print("---------------------- inputs",inputs.shape)
            valid_logits, attns = model(inputs)
            # print("---------------------- valid_logits",valid_logits[0].cpu().numpy().shape)
            for i in range(len(imgpaths)):
                valid_probs = torch.sigmoid(valid_logits[0]).cpu().numpy()
                valid_probs = valid_probs[0]>0.5
                for j, val in enumerate(valid_probs):
                    if val:
                        f.write(str(imgpaths[i]) + ' ' + str(j) + ' ' + str(1) + '\n')
                    else:
                        f.write(str(imgpaths[i]) + ' ' + str(j) + ' ' + str(0) + '\n')

def argument_parser():
    parser = argparse.ArgumentParser(description="attribute recognition",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--test_img", help="test images", type=str,
        default="/home/c3-0/datasets/CCVID/CCVID/",
    )
    parser.add_argument(
        "--cfg", help="decide which cfg to use", type=str,default='configs/peta_zs.yaml'
    )
    parser.add_argument("--debug", type=str2bool, default="true")


    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = argument_parser()
    update_config(cfg, args)

    main(cfg, args)