"""
Written by Vamseekrishna Kattika
Feb 16, 2019
"""
import sys
import cv2
import numpy as np

#pylint: disable=all


def linear_scaling_xyY_domain(input_image, tmp, W_1, H_1, W_2, H_2):
    """
    Performs linear scaling in xyY domain
    """
    linear_rgb_to_xyz_mat = np.array([0.412453,
                                      0.35758,
                                      0.180423,
                                      0.212671,
                                      0.71516,
                                      0.072169,
                                      0.019334,
                                      0.119193,
                                      0.950227]).reshape(3,
                                                         3)
    xyz_to_linear_rgb_mat = np.array([3.240479, -
                                      1.53715, -
                                      0.498535, -
                                      0.969256, 1.875991, 0.041556, 0.055648, -
                                      0.204043, 1.057311]).reshape(3, 3)

    def inv_gamma(v):
        if v < 0.03928:
            return v / 12.92
        return ((v + 0.055) / 1.055)**2.4

    def gamma(D):
        if D < 0.00304:
            return 12.92 * D
        return 1.055 * (D ** (1 / 2.4)) - 0.055

    def clipping(x):
        if x > 1:
            return 1
        if x < 0:
            return 0
        return x

    def srgb_to_non_linear_rgb(b, g, r):
        return (b / 255, g / 255, r / 255)

    def non_linear_to_linear_rgb(nlb, nlg, nlr):
        return (inv_gamma(nlb), inv_gamma(nlg), inv_gamma(nlr))

    def linear_rgb_to_xyz(lb, lg, lr):
        linear_rgb_mat = np.array([lr, lg, lb]).reshape(3, 1)
        xyz_arr = np.dot(linear_rgb_to_xyz_mat, linear_rgb_mat).flatten()
        return (xyz_arr[0], xyz_arr[1], xyz_arr[2])

    def xyz_to_xyY(X, Y, Z):
        if X + Y + Z == 0:
            return (0, 0, Y)
        x = X / (X + Y + Z)
        y = Y / (X + Y + Z)
        return (x, y, Y)

    def srgb_to_xyY(b, g, r):
        nlb, nlg, nlr = srgb_to_non_linear_rgb(b, g, r)
        lb, lg, lr = non_linear_to_linear_rgb(nlb, nlg, nlr)
        x, y, z = linear_rgb_to_xyz(lb, lg, lr)
        x_dash, y_dash, Y = xyz_to_xyY(x, y, z)
        return (x_dash, y_dash, Y)

    def xyY_to_xyz(x, y, Y):
        if y == 0:
            return (0, Y, 0)
        X = Y * x / y
        Z = Y * (1 - x - y) / y

        return (X, Y, Z)

    def xyz_to_linear_rgb(x, y, z):
        xyz_mat = np.array([x, y, z]).reshape(3, 1)
        rgb_arr = np.dot(xyz_to_linear_rgb_mat, xyz_mat).flatten()
        return (rgb_arr[2], rgb_arr[1], rgb_arr[0])

    def linear_to_non_linear_rgb(lb, lg, lr):
        nlb, nlg, nlr = (clipping(lb), clipping(lg), clipping(lr))
        return (gamma(nlb), gamma(nlg), gamma(nlr))

    def non_linear_to_srgb(nlb, nlg, nlr):
        return (int(round(nlb * 255)),
                int(round(nlg * 255)),
                int(round(nlr * 255)))

    def xyY_to_srgb(x, y, Y):
        X, Y_dash, Z = xyY_to_xyz(x, y, Y)
        lb, lg, lr = xyz_to_linear_rgb(X, Y_dash, Z)
        nlb, nlg, nlr = linear_to_non_linear_rgb(lb, lg, lr)
        b, g, r = non_linear_to_srgb(nlb, nlg, nlr)
        return (b, g, r)

    def linear_scaling(b, g, r, A, B):
        x, y, Y = srgb_to_xyY(b, g, r)
        new_y = (Y - A) / (B - A)
        b, g, r = xyY_to_srgb(x, y, new_y)
        return b, g, r

    A = 1
    B = 0
    for i in range(H_1, H_2 + 1):
        for j in range(W_1, W_2 + 1):
            b, g, r = input_image[i, j]
            tup = srgb_to_xyY(b, g, r)
            if tup[2] < A:
                A = tup[2]
            if tup[2] > B:
                B = tup[2]

    for i in range(H_1, H_2 + 1):
        for j in range(W_1, W_2 + 1):
            b, g, r = input_image[i, j]
            b, g, r = linear_scaling(b, g, r, A, B)
            tmp[i, j] = b, g, r


def main():
    """
    Main function
    """

    if len(sys.argv) != 7:
        print(sys.argv[0], ': takes 6 arguments not', len(sys.argv) - 1)
        print('Expecting arguments w1 h1 w2 h2 ImageIn ImageOut.')
        print('Example:', sys.argv[0], "0.2 0.1 0.8 0.5 fruits.jpg out.png")
        sys.exit()

    w_1 = float(sys.argv[1])
    h_1 = float(sys.argv[2])
    w_2 = float(sys.argv[3])
    h_2 = float(sys.argv[4])
    image_in = sys.argv[5]
    image_out = sys.argv[6]

    if w_1 < 0 or h_1 < 0 or w_2 <= w_1 or h_2 <= h_1 or w_2 > 1 or h_2 > 1:
        print('Arguments must satisfy 0 <= w1 < w2 <= 1, 0 <=h1 < h2 <= 1 ')
        sys.exit()

    input_image = cv2.imread(image_in, cv2.IMREAD_COLOR)
    if input_image is None:
        print(sys.argv[0], ': Failed to read image from: ', image_in)
        sys.exit()

    cv2.imshow("input image: " + image_in, input_image)
    rows, cols, bands = input_image.shape
    W_1 = round(w_1 * (cols - 1))
    H_1 = round(h_1 * (rows - 1))
    W_2 = round(w_2 * (cols - 1))
    H_2 = round(h_2 * (rows - 1))

    tmp = np.copy(input_image)

    linear_scaling_xyY_domain(input_image, tmp, W_1, H_1, W_2, H_2)

    cv2.imshow('Temporary Image:', tmp)

    output_image = np.zeros([rows, cols, bands], dtype=np.uint8)

    for i in range(0, rows):
        for j in range(0, cols):
            b, g, r = tmp[i, j]
            output_image[i, j] = [b, g, r]

    cv2.imshow("Output Image:" + image_out, output_image)
    cv2.imwrite(image_out, output_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
