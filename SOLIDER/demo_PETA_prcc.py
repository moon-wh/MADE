import argparse
import glob
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from dataset.augmentation import get_transform
from models.model_factory import build_backbone, build_classifier

import torch
import os.path as osp
from PIL import Image
from configs import cfg, update_config

from models.base_block import FeatClassifier
from tools.function import get_model_log_path, get_reload_weight
from tools.utils import set_seed, str2bool
set_seed(605)

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

    model = get_reload_weight(model_dir, model, pth='')   # Change weights path
    model.eval()

    with torch.no_grad():
        for name in os.listdir(args.test_img+'/rgb'):
            if name == 'train' or name == 'val':
                pdirs = glob.glob(osp.join(args.test_img+'/rgb/'+name, '*'))
                pdirs.sort()
                for pdir in pdirs:
                    pid = int(osp.basename(pdir))
                    img_dirs = glob.glob(osp.join(pdir, '*.jpg'))
                    for img_dir in img_dirs:
                        img = Image.open(img_dir)
                        img = img.convert("RGB")
                        img = valid_tsfm(img).cuda()
                        img = img.view(1, *img.size())
                        valid_logits, attns = model(img)
                        valid_probs = torch.sigmoid(valid_logits[0]).cpu().numpy()
                        valid_probs = valid_probs[0]>0.5
                        for i, val in enumerate(valid_probs):
                            if val:
                                with open(args.test_img + '/' + 'PAR_PETA_105.txt', 'a',
                                          encoding='utf-8') as f:
                                    f.write(img_dir+' '+ str(i)+' '+ str(1) + '\n')
                            else:
                                with open(args.test_img + '/' + 'PAR_PETA_105.txt', 'a',
                                          encoding='utf-8') as f:
                                    f.write(img_dir+' '+ str(i)+' '+ str(0) + '\n')
            else:
                for cam in ['A', 'B', 'C']:
                    pdirs = glob.glob(osp.join(args.test_img+'/rgb/test', cam, '*'))
                    for pdir in pdirs:
                        pid = int(osp.basename(pdir))
                        img_dirs = glob.glob(osp.join(pdir, '*.jpg'))
                        for img_dir in img_dirs:
                            img = Image.open(img_dir)
                            img = img.convert("RGB")
                            img = valid_tsfm(img).cuda()
                            img = img.view(1, *img.size())
                            valid_logits, attns = model(img)
                            valid_probs = torch.sigmoid(valid_logits[0]).cpu().numpy()
                            valid_probs = valid_probs[0]>0.5
                            for i, val in enumerate(valid_probs):
                                if val:
                                    with open(args.test_img + '/' + 'PAR_PETA_105.txt', 'a',
                                              encoding='utf-8') as f:
                                        f.write(img_dir +' '+  str(i) +' '+  str(1) + '\n')
                                else:
                                    with open(args.test_img + '/' + 'PAR_PETA_105.txt', 'a',
                                              encoding='utf-8') as f:
                                        f.write(img_dir +' '+ str(i) +' '+ str(0) + '\n')

def argument_parser():
    parser = argparse.ArgumentParser(description="attribute recognition",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--test_img", help="test images", type=str,
        default="../prcc",
    )
    parser.add_argument(
        "--cfg", help="decide which cfg to use", type=str,default='../MADE/SOLIDER/configs/peta_zs.yaml'
    )
    parser.add_argument("--debug", type=str2bool, default="true")


    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = argument_parser()
    update_config(cfg, args)

    main(cfg, args)