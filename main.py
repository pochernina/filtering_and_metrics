import cv2
import argparse
import numpy as np
import math

def mse(im1, im2):
    err = np.sum((im1.astype('float') - im2.astype('float')) ** 2)
    err /= im1.shape[0] * im1.shape[1]
    return err

def psnr(im1, im2):
    return 20 * math.log10(255 / math.sqrt(mse(im1, im2)))

def ssim(im1, im2):
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    im1 = im1.astype('float')
    im2 = im2.astype('float')
    mu_1 = np.mean(im1)
    mu_2 = np.mean(im2)
    sigma_1_sq = np.var(im1)
    sigma_2_sq = np.var(im2)
    sigma_12 = np.mean((im1 - mu_1) * (im2 - mu_2))
    return ((2 * mu_1 * mu_2 + c1) * (2 * sigma_12 + c2)) / ((mu_1 ** 2 + mu_2 ** 2 + c1) * (sigma_1_sq + sigma_2_sq + c2))

def median_filter(im, rad):
    temp = []
    new_im = np.zeros(im.shape)
    for i in range(len(im)):
        for j in range(len(im[0])):
            for newi in range(-rad, rad + 1):
                for newj in range(-rad, rad + 1):
                    ii = 0 if i + newi < 0 else (len(im) - 1 if i + newi > len(im) - 1 else i + newi)
                    jj = 0 if j + newj < 0 else (len(im[0]) - 1 if j + newj > len(im[0]) - 1 else j + newj)
                    temp.append(im[ii][jj])
            temp.sort()
            new_im[i][j] = temp[(len(temp) - 1) // 2]
            temp = []
    return new_im

def gaussian_kernel(rad, sigma):
    kernel = np.zeros((2 * rad + 1, 2 * rad + 1))
    for i in range(-rad, rad + 1):
        for j in range(-rad, rad + 1):
            kernel[i + rad][j + rad] = 1 / (2 * math.pi * (sigma ** 2)) * np.exp(-(i ** 2 + j ** 2) / (2 * (sigma ** 2)))
    return kernel

def gaussian_filter(im, rad, sigma):
    new_im = np.zeros(im.shape)
    kernel = gaussian_kernel(rad, sigma)
    imb = cv2.copyMakeBorder(im, rad, rad, rad, rad, cv2.BORDER_REPLICATE)
    for i in range(rad, rad + len(im)):
        for j in range(rad, rad + len(im[0])):
            g = imb[i - rad: i + rad + 1, j - rad: j + rad + 1]
            new_im[i - rad][j - rad] = np.sum(kernel * g)
    return new_im

def bilateral_filter(im, rad, sigma_d, sigma_r):
    im = im.astype('int')
    new_im = np.zeros(im.shape)
    imb = cv2.copyMakeBorder(im, rad, rad, rad, rad, cv2.BORDER_REPLICATE)
    gauss = gaussian_kernel(rad, sigma_d) * (2 * math.pi * (sigma_d ** 2))
    for i in range(rad, rad + len(im)):
        for j in range(rad, rad + len(im[0])):
            g = imb[i - rad: i + rad + 1, j - rad: j + rad + 1]
            bilateral = np.exp(-((g - imb[i, j]) ** 2 / (2 * (sigma_r ** 2))))
            kernel = bilateral * gauss
            new_im[i - rad][j - rad] = np.sum(g * kernel) / np.sum(kernel)
    return new_im


parser = argparse.ArgumentParser(description = 'Filtering and metrics')
parser.add_argument('command', type = str, choices = ['mse', 'psnr', 'ssim', 'median', 'gauss', 'bilateral'])
parser.add_argument('parameters', nargs = '+')
args = parser.parse_args()

if args.command == 'mse' or args.command == 'psnr' or args.command == 'ssim':
    input_file1 = args.parameters[0]
    input_file2 = args.parameters[1]
    im1 = cv2.imread(input_file1, cv2.IMREAD_GRAYSCALE)
    im2 = cv2.imread(input_file2, cv2.IMREAD_GRAYSCALE)
    if args.command == 'mse':
        print(mse(im1, im2))
    elif args.command == 'psnr':
        print(psnr(im1, im2))
    else:
        print(ssim(im1, im2))
    
elif args.command == 'median':
    rad = int(args.parameters[0])
    input_file = args.parameters[1]
    output_file = args.parameters[2]
    im = cv2.imread(input_file, cv2.IMREAD_GRAYSCALE)
    new_im = median_filter(im, rad)
    cv2.imwrite(output_file, new_im)

elif args.command == 'gauss':
    sigma_d = float(args.parameters[0])
    input_file = args.parameters[1]
    output_file = args.parameters[2]
    rad = int(3 * sigma_d)
    im = cv2.imread(input_file, cv2.IMREAD_GRAYSCALE)
    new_im = gaussian_filter(im, rad, sigma_d)
    cv2.imwrite(output_file, new_im)

else:
    sigma_d = float(args.parameters[0])
    sigma_r = float(args.parameters[1])
    input_file = args.parameters[2]
    output_file = args.parameters[3]
    rad = int(3 * sigma_d)
    im = cv2.imread(input_file, cv2.IMREAD_GRAYSCALE)
    new_im = bilateral_filter(im, rad, sigma_d, sigma_r)
    cv2.imwrite(output_file, new_im)