import cv2
import argparse
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity

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
        print(mean_squared_error(im1, im2))
    elif args.command == 'psnr':
        print(peak_signal_noise_ratio(im1, im2))
    else:
        height, width = im1.shape[:2]
        rad = min(height, width)
        if rad % 2 == 0: rad -= 1
        print(structural_similarity(im1, im2, win_size = rad))
    
elif args.command == 'median':
    rad = int(args.parameters[0])
    input_file = args.parameters[1]
    output_file = args.parameters[2]
    im = cv2.imread(input_file)
    new_im = cv2.medianBlur(im, 2 * rad + 1)
    cv2.imwrite(output_file, new_im)

elif args.command == 'gauss':
    sigma_d = float(args.parameters[0])
    input_file = args.parameters[1]
    output_file = args.parameters[2]
    rad = int(3 * sigma_d)
    im = cv2.imread(input_file)
    new_im = cv2.GaussianBlur(im, (2 * rad + 1, 2 * rad + 1), sigma_d, sigma_d)
    cv2.imwrite(output_file, new_im)

else:
    sigma_d = float(args.parameters[0])
    sigma_r = float(args.parameters[1])
    input_file = args.parameters[2]
    output_file = args.parameters[3]
    rad = int(3 * sigma_d)
    im = cv2.imread(input_file, cv2.IMREAD_GRAYSCALE)
    new_im = cv2.bilateralFilter(im, 2 * rad + 1, sigma_r, sigma_d)
    cv2.imwrite(output_file, new_im)