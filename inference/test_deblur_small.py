import os
import torch
import glob
import numpy as np
import imageio
import cv2
import math
import time
import argparse
from basicsr.models.archs.gshift_deblur2 import GShiftNet
from skimage.metrics import structural_similarity as SSIM_
from skimage.metrics import peak_signal_noise_ratio as PSNR_
from scipy.ndimage import gaussian_filter

class Traverse_Logger:
    def __init__(self, result_dir, filename='inference_log.txt'):
        self.log_file_path = os.path.join(result_dir, filename)
        open_type = 'a' if os.path.exists(self.log_file_path) else 'w'
        self.log_file = open(self.log_file_path, open_type)

    def write_log(self, log):
        print(log)
        self.log_file.write(log + '\n')

def ssim_calculate(img1, img2, sd=1.5, C1=0.01**2, C2=0.03**2):
    # Processing input image
    img1 = np.array(img1, dtype=np.float32) / 255
    img1 = img1.transpose((2, 0, 1))

    # Processing gt image
    img2 = np.array(img2, dtype=np.float32) / 255
    img2 = img2.transpose((2, 0, 1))


    mu1 = gaussian_filter(img1, sd)
    mu2 = gaussian_filter(img2, sd)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = gaussian_filter(img1 * img1, sd) - mu1_sq
    sigma2_sq = gaussian_filter(img2 * img2, sd) - mu2_sq
    sigma12 = gaussian_filter(img1 * img2, sd) - mu1_mu2

    ssim_num = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2))

    ssim_den = ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    ssim_map = ssim_num / ssim_den
    return np.mean(ssim_map)

class Inference:
    def __init__(self, args):

        self.save_image = args.save_image
        self.border = args.border
        self.model_path = args.model_path
        self.data_path = args.data_path
        self.result_path = args.result_path
        self.n_seq = 5
        self.size_must_mode = 4
        self.device = 'cuda'
        self.one_len = args.one_len

        if not os.path.exists(self.result_path):
            os.mkdir(self.result_path)
            print('mkdir: {}'.format(self.result_path))

        self.input_path = os.path.join(self.data_path, "blur")
        self.GT_path = os.path.join(self.data_path, "gt")

        now_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        self.logger = Traverse_Logger(self.result_path, 'inference_log_{}.txt'.format(now_time))

        self.logger.write_log('Inference - {}'.format(now_time))
        self.logger.write_log('save_image: {}'.format(self.save_image))
        self.logger.write_log('border: {}'.format(self.border))
        self.logger.write_log('model_path: {}'.format(self.model_path))
        self.logger.write_log('data_path: {}'.format(self.data_path))
        self.logger.write_log('result_path: {}'.format(self.result_path))
        self.logger.write_log('n_seq: {}'.format(self.n_seq))
        self.logger.write_log('size_must_mode: {}'.format(self.size_must_mode))
        self.logger.write_log('device: {}'.format(self.device))

        self.net = GShiftNet(future_frames=2, past_frames=2)
        self.net.load_state_dict(torch.load(self.model_path)['params'])
        self.net.half()
        self.net = self.net.to(self.device)
        self.logger.write_log('Loading model from {}'.format(self.model_path))
        self.net.eval()

    def infer(self):
        with torch.no_grad():
            total_psnr = {}
            total_ssim = {}
            videos = sorted(os.listdir(self.input_path))
            # print(len(videos))
            # exit(0)
            for v in videos:
                video_psnr = []
                video_ssim = []
                input_frames = sorted(glob.glob(os.path.join(self.input_path, v, "*")))
                gt_frames = sorted(glob.glob(os.path.join(self.GT_path, v, "*")))
                # print(len(input_frames), len(gt_frames))
                # print(input_frames[0:4])
                # exit(0)

                # input_seqs = self.gene_seq(input_frames, n_seq=self.n_seq)
                # gt_seqs = self.gene_seq(gt_frames, n_seq=self.n_seq)
                # for in_seq, gt_seq in zip(input_seqs, gt_seqs):
                # 
                begin_frames = 2
                end_frames = 2
                one_len = self.one_len # len(input_frames) - begin_frames - end_frames

                k_len = (len(input_frames) - begin_frames - end_frames) // one_len
                index = 0
                for kk in range(k_len):
                    start_time = time.time()
                    in_seq = input_frames[kk*one_len+0:kk*one_len+one_len+begin_frames+end_frames]
                    gt_seq = gt_frames[kk*one_len+begin_frames:kk*one_len+begin_frames+one_len]
                    filename = os.path.basename(in_seq[self.n_seq // 2]).split('.')[0]
                    inputs = [imageio.imread(p) for p in in_seq]
                    gts = [imageio.imread(p) for p in gt_seq]
                    h, w, c = inputs[self.n_seq // 2].shape
                    new_h, new_w = h - h % self.size_must_mode, w - w % self.size_must_mode
                    inputs = [im[:new_h, :new_w, :] for im in inputs]
                    gts = [im[:new_h, :new_w, :] for im in gts]
                    in_tensor = self.numpy2tensor(inputs).to(self.device)
                    preprocess_time = time.time()
                    # print(in_tensor.shape)
                    _, _, _, H, W = in_tensor.shape
                    # print(in_tensor.shape)

                    output = self.net(in_tensor.half())
                    #print(output.shape)
                    # exit(0)
                    output = output.float()
                    forward_time = time.time()
                    for ele in range(one_len):
                        output_img = output[ele].clamp(0,1.0).permute(1,2,0).cpu().numpy()
                        output_img = output_img * 255
                        psnr = PSNR_(output_img, gts[ele], data_range=255)
                        ssim = ssim_calculate(output_img, gts[ele])
                        # print(psnr)
                        video_psnr.append(psnr)
                        video_ssim.append(ssim)
                        total_psnr[v] = video_psnr
                        total_ssim[v] = video_ssim
                        if self.save_image:
                            if not os.path.exists(os.path.join(self.result_path, v)):
                                os.mkdir(os.path.join(self.result_path, v))
                            cv2.imwrite(os.path.join(self.result_path, v, '%03d.png'%ele), output_img[...,::-1])
                    postprocess_time = time.time()
                    del output; del in_tensor
                    torch.cuda.empty_cache()

                    self.logger.write_log(
                        '> {}-{} PSNR={:.5}, SSIM={:.4} pre_time:{:.3}s, forward_time:{:.3}s, post_time:{:.3}s, total_time:{:.3}s'
                            .format(v, filename, psnr, ssim,
                                    preprocess_time - start_time,
                                    forward_time - preprocess_time,
                                    postprocess_time - forward_time,
                                    postprocess_time - start_time))
                    # exit(0)


            sum_psnr = 0.
            sum_ssim = 0.
            n_img = 0
            for k in total_psnr.keys():
                self.logger.write_log("# Video:{} AVG-PSNR={:.5}, AVG-SSIM={:.4}".format(
                    k, sum(total_psnr[k]) / len(total_psnr[k]), sum(total_ssim[k]) / len(total_ssim[k])))
                sum_psnr += sum(total_psnr[k])
                sum_ssim += sum(total_ssim[k])
                n_img += len(total_psnr[k])
            self.logger.write_log("# Total AVG-PSNR={:.5}, AVG-SSIM={:.4}".format(sum_psnr / n_img, sum_ssim / n_img))

    def gene_seq(self, img_list, n_seq):
        if self.border:
            half = n_seq // 2
            img_list_temp = img_list[:half]
            img_list_temp.extend(img_list)
            img_list_temp.extend(img_list[-half:])
            img_list = img_list_temp
        seq_list = []
        for i in range(len(img_list) - 2 * (n_seq // 2)):
            seq_list.append(img_list[i:i + n_seq])
        return seq_list

    def numpy2tensor(self, input_seq, rgb_range=1.):
        tensor_list = []
        for img in input_seq:
            img = np.array(img).astype('float64')
            np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))  # HWC -> CHW
            tensor = torch.from_numpy(np_transpose).float()  # numpy -> tensor
            tensor.mul_(rgb_range / 255)  # (0,255) -> (0,1)
            tensor_list.append(tensor)
        stacked = torch.stack(tensor_list).unsqueeze(0)
        return stacked

    def tensor2numpy(self, tensor, rgb_range=1.):
        rgb_coefficient = 255 / rgb_range
        img = tensor.mul(rgb_coefficient).clamp(0, 255).round()
        img = img[0].data
        img = np.transpose(img.cpu().numpy(), (1, 2, 0)).astype(np.uint8)
        return img

    def get_PSNR_SSIM(self, output, gt, crop_border=4):
        cropped_output = output[crop_border:-crop_border, crop_border:-crop_border, :]
        cropped_GT = gt[crop_border:-crop_border, crop_border:-crop_border, :]
        psnr = self.calc_PSNR(cropped_GT, cropped_output)
        ssim = self.calc_SSIM(cropped_GT, cropped_output)
        return psnr, ssim

    def calc_PSNR(self, img1, img2):
        '''
        img1 and img2 have range [0, 255]
        '''
        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        mse = np.mean((img1 - img2) ** 2)
        if mse == 0:
            return float('inf')
        return 20 * math.log10(255.0 / math.sqrt(mse))

    def calc_SSIM(self, img1, img2):
        '''calculate SSIM
        the same outputs as MATLAB's
        img1, img2: [0, 255]
        '''

        def ssim(img1, img2):
            C1 = (0.01 * 255) ** 2
            C2 = (0.03 * 255) ** 2

            img1 = img1.astype(np.float64)
            img2 = img2.astype(np.float64)
            kernel = cv2.getGaussianKernel(11, 1.5)
            window = np.outer(kernel, kernel.transpose())

            mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
            mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
            mu1_sq = mu1 ** 2
            mu2_sq = mu2 ** 2
            mu1_mu2 = mu1 * mu2
            sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
            sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
            sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

            ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                                    (sigma1_sq + sigma2_sq + C2))
            return ssim_map.mean()

        if not img1.shape == img2.shape:
            raise ValueError('Input images must have the same dimensions.')
        if img1.ndim == 2:
            return ssim(img1, img2)
        elif img1.ndim == 3:
            if img1.shape[2] == 3:
                ssims = []
                for i in range(3):
                    ssims.append(ssim(img1, img2))
                return np.array(ssims).mean()
            elif img1.shape[2] == 1:
                return ssim(np.squeeze(img1), np.squeeze(img2))
        else:
            raise ValueError('Wrong input image dimensions.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CDVD-TSP-Inference')

    parser.add_argument('--save_image', action='store_true', default=False, help='save image if true')
    parser.add_argument('--border', action='store_true', help='restore border images of video if true')

    parser.add_argument('--default_data', type=str, default='.',
                        help='quick test, optional: DVD, GOPRO')
    parser.add_argument('--one_len', type=int, default=96)
    args = parser.parse_args()
    if args.default_data == 'DVD':
        args.data_path = './dataset/DVD/test/'
        args.model_path = 'pretrained_models/net_dvd_deblur_small.pth'
        args.result_path = 'infer_results/DVD'
    elif args.default_data == 'GOPRO':
        args.data_path = './dataset/GOPRO/test/'
        args.model_path = 'pretrained_models/net_gopro_deblur_small.pth'
        args.result_path = 'infer_results/gopro'

    Infer = Inference(args)
    Infer.infer()
