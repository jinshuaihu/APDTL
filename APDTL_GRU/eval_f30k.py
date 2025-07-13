import os
import time
import logging
import argparse
from lib import evaluation


logging.basicConfig()


logger = logging.getLogger()
logger.setLevel(logging.INFO)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='f30k', help='coco or f30k')
    parser.add_argument('--data_path', default='./data/f30k')
    parser.add_argument('--save_results', action='store_true')
    parser.add_argument('--evaluate_cxc', action='store_true')
    opt = parser.parse_args()

    if opt.dataset == 'coco':
        weights_bases = ['runs/coco_butd_region_bigru',]
    elif opt.dataset == 'f30k':
        weights_bases = ['./runs/f30k_butd_region_bigru',]
    else:
        raise ValueError('Invalid dataset argument {}'.format(opt.dataset))

    for base in weights_bases:
        logger.info('Evaluating {}...'.format(base))
        model_path = os.path.join(base, 'model_best.pth')
        if opt.save_results:
            save_path = os.path.join(base, 'results_{}.npy'.format(opt.dataset))
        else:
            save_path = None

        if opt.dataset == 'coco':
            if not opt.evaluate_cxc:
                # Evaluate COCO 5-fold 1K
                evaluation.evalrank(model_path, data_path=opt.data_path, split='testall', fold5=True)
                # Evaluate COCO 5K
                evaluation.evalrank(model_path, data_path=opt.data_path, split='testall', fold5=False, save_path=save_path)
            else:
                # Evaluate COCO-trained models on CxC
                evaluation.evalrank(model_path, data_path=opt.data_path, split='testall', fold5=True, cxc=True)
        elif opt.dataset == 'f30k':
            # Evaluate Flickr30K
            stime = time.time()
            evaluation.evalrank(model_path, data_path=opt.data_path, split='test', fold5=False, save_path=save_path)
            print('eval total time: ', time.time() - stime)


if __name__ == '__main__':

    main()
