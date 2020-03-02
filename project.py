from PIL import Image, ImageDraw, ImageFont
import numpy as np
import math
import tkinter as tk
from tkinter import messagebox
from PIL import ImageTk, Image
from tkinter import filedialog

# Kaan Mert Koç 041801122 02.03.2020
# This program takes an image that consist of numbers from the user and after making eight connection it calculates the
# moments for that numbers and compare it with original image that also consist numbers with k nearest neighbors
# algorithm then saves prediction into txt files.

def main():

    img = Image.open('/Users/kaankoc/Desktop/all.jpeg')
    hu_moment, r_moment, z_moments = main_setter(img)
    img2 = ButtonEvent()
    hu_moment2, r_moment2, zernik_moments2 = main_setter(img2)

    predictions = make_predictions(z_moments, zernik_moments2)
    z = open("/Users/kaankoc/Desktop/z.txt", "w")
    index = 1
    for elements in predictions:
        z.write("Prediction for number " + str(index) + str(elements) + ", ")
        if index != 9:
            index += 1
        else:
            index = 0
    index = 0

    predictions = make_predictions(hu_moment, hu_moment2)
    hu = open("/Users/kaankoc/Desktop/hu.txt", "w")
    for elements in predictions:
        hu.write("Prediction for number " + str(index) + " " +  str(elements) + ", ")
        if index != 9:
            index += 1
        else:
            index = 0
    index = 0

    predictions = make_predictions(r_moment, r_moment2)
    r = open("/Users/kaankoc/Desktop/r.txt", "w")
    for elements in predictions:
        r.write("Prediction for number " + str(index) + str(elements) + ", ")
        if index != 9:
            index += 1
        else:
            index = 0

# This method makes predictions depending on distances values from the moments.
# First input is the original image which we take as an input, second is the moment values that predictions wanted to
# make. Return is a list of predictions.
def make_predictions(database_moments, target_moments):
    predictions = []

    a = 0
    del database_moments[0]
    del target_moments[0]
    for elements in target_moments:
        distances = get_neighbors(database_moments, elements)
        del distances[0]
        min = distances[0][1]
        index = 0
        for i in range(len(distances)):
            if distances[i][1] < min:
                min = distances[i][1]
                index += 1
        predictions.insert(a, index + 1)
        a += 1
    return predictions


# This method does set the program to take input from the user and does necessary jobs in order to calculate moments.
# This method does this in order, takes the image as an input converts it to gray, makes them eight connected,
# find the min x,y and max x,y values in order the draw rectangles aronud them, crops the image then calculates the all
# moments.
# First input is the img. Return is the moments of that image.
def main_setter(img):
    img_gray = img.convert('L')  # converts the image to grayscale image
    img_gray.show()
    ONE = 150
    a = np.asarray(img_gray)  # from PIL to np array

    a_bin = threshold(a, 100, ONE, 0)
    im = Image.fromarray(a_bin)  # from np array to PIL format
    im.show()

    im_label, label, max_k = blob_coloring_8_connected(a_bin, ONE)

    new_img2 = np2PIL_color(label)

    coordinates = find_coordinates(im_label, max_k)
    draw_rectangle(coordinates, new_img2)
    hu_moments, r_moments, z_moments = crop_images_calculates_moments(coordinates, im, ONE)

    new_img2.show()
    return hu_moments, r_moments, z_moments


# This method implemented by our instructor, sets the gui for taken image from the user.
# Return is the image that is taken from the user.
def ButtonEvent():
    messagebox.showinfo( "COMP204", "Programming Studio")
    top = tk.Tk()  # creates window

    top.filename =filedialog.askopenfilename(initialdir = "/",title = "Select file",
                                         filetypes =(("jpeg files","*.jpg"),("all files","*.*"),("png files", "*.png")))
    img = Image.open(top.filename) # image

    return img


# This method calculates the euclidean distance with given formula square root of num1^2 - num2^2
# inputs are two numbers the return is the float value of the equation
def euclidean_distance(arr1, arr2):
    distance = 0.0
    for i in range(len(arr1) - 1):
        distance += pow((arr1[i] - arr2[i]), 2)
    return math.sqrt(distance)


# This method uses the euclidean_distance method and returns the distance to the moments stored in each numbers respectively
# The first input is a 2d array stores the moments for each number in the database, second input is an array of size seven
# where it stores the moments of the number that wanted to be recognized. Return is a 2d array, first index is the index
# of the array in the database of moments array that distances is calculated.
def get_neighbors(database_of_moments, target_moments):
    index = 0
    distances = [[]]
    for moments in database_of_moments:
            dist = euclidean_distance(target_moments, moments)

            distances.append([index, dist])
            index += 1
    return distances


# This method is for cropping and resizing the image then calculating the hu, r and zernike moments.
# First input is a 2d array which stores the coordinates for low x,y and max x,y values for the images
# Second input is the binarized array of the image. Third one is the constant number for moments.
def crop_images_calculates_moments(coordinates, im, ONE):
    row, col = coordinates.shape[0], coordinates.shape[1]
    # calculate hu moments then store it in 2d row to 7 array
    hu_moments = [[]]
    r_moments = [[]]
    z_moments = [[]]
    for i in range(row):
        if coordinates[i][0] == 1:
            # This line stores the minimum x and y values from the array.
            x1, y1, x2, y2 = coordinates[i][1], coordinates[i][2], coordinates[i][3], coordinates[i][4]
            # This line creates a box from the stores x and y values.
            box = (x1, y1, x2, y2)
            image_cropped = im.crop(box)

            image_resize = image_cropped.resize((21,21))
            # We have to binarized the array again because we resized the image and use it for calculating moments.
            arr = np.asarray(image_resize)
            a_bin = threshold(arr, ONE/3, 1, 0)
            hu_list = calculate_hu(a_bin, ONE)
            r_list = calculate_r_moment(hu_list)
            z11, z22, z31 = calculate_zernike(a_bin, 1, 1, ONE), calculate_zernike(a_bin, 2, 2, ONE), calculate_zernike(a_bin, 3, 1, ONE)
            z33, z42, z44 = calculate_zernike(a_bin, 3, 3, ONE), calculate_zernike(a_bin, 4, 2, ONE), calculate_zernike(a_bin, 4, 4, ONE)
            z51, z53, z55 = calculate_zernike(a_bin, 5, 1, ONE), calculate_zernike(a_bin, 5, 3, ONE), calculate_zernike(a_bin, 5, 5, ONE)
            z62, z64, z66 = calculate_zernike(a_bin, 6, 2, ONE), calculate_zernike(a_bin, 6, 4, ONE), calculate_zernike(a_bin, 6, 6, ONE)
            z_moments.append([z11, z22, z31, z33, z42, z44, z51, z53, z55, z62, z64, z66])
            # I added this two if blocks for precaution because i wasn't sure the results would be not zero and i still should
            # feel like it should stay like that.
            if len(hu_list) != 0:
                hu_moments.append(hu_list)
            if len(r_list) != 0:
                r_moments.append(r_list)

            image_resize.show()

    return hu_moments, r_moments, z_moments


# This method calculates the moment for each values of binarized array.
# First input is the array, second one is the constant value.
# Return is a list that consist seven moment values.
def calculate_hu(arr, ONE):
    row, col = len(arr), len(arr[0])
    m = np.zeros(shape=(row, col), dtype=float)
    mu = np.zeros(shape=(row, col), dtype=float)
    n = np.zeros(shape=(row, col), dtype=float)
    # I calculate the m values for every row and col of the array, but the k and l values goes to one only.
    for k in range(1):
        for l in range(1):
            for i in range(row):
                for j in range(col):
                    m[k][l] += pow(i, k) * pow(j, l) * (arr[i][j] / ONE)
    # Here i calculate the x0 and y0 values depending on m values.
    x0 = m[1][0] / m[0][0]
    y0 = m[0][1] / m[0][0]
    # Here i calculate mu and n values depending on row and col again and i use x0 and y0 to do so.
    for k in range(3):
        for l in range(3):
            for i in range(row):
                for j in range(col):
                    mu[k][l] += pow(i - x0, k) * pow(j - y0, l) * (arr[i][j] / ONE)
                    y = ((k + l) / 2) + 1
                    # In this line i included the exp, because the value was too small for some values it was rounding
                    # into zero and that was causing an error.
                    n[k][l] += mu[k][l] / np.exp(mu[0][0]**y)
    # I calculate all the h moments depending on the values of n.
    h1 = n[2][0] + n[0][2]
    h2 = pow(n[2][0] - n[0][2], 2) + 4 * pow(n[1][1], 2)
    h3 = pow(n[3][0] - 3 * n[1][2], 2) + pow(3 * n[2][1] - n[0][3], 2)
    h4 = pow(n[3][0] + n[1][2], 2) + pow(n[2][1] + n[0][3], 2)
    h5 = (n[3][0] - (3 * n[1][2])) * (pow(n[3][0] + n[1][2], 2) - 3 * pow(n[2][1] + n[0][3], 2)) + ((3 * n[2][1]) - n[0][3] * (n[2][1] + n[0][3] * (3 * pow(n[3][0] + n[1][2], 2)) - pow(n[2][1] + n[0][3], 2)))
    h6 = (n[2][0] - n[0][3]) * (pow(n[3][0] + n[1][2], 2) - (pow(n[2][1] + n[0][3], 2))) + (4 * n[1][1] * (n[3][0] + n[1][2]) * (n[2][1] + n[0][3]))
    h7 = ((3 * n[2][1] - n[0][3]) * (n[3][0] + n[1][2]) * ((pow(n[3][0] + n[1][2], 2)) - 3 * pow(n[2][1] + n[0][3], 2))) + (3 * n[1][2] - n[3][0] * (n[2][1] + n[0][3]) * (3 * pow(n[3][0] + n[1][2], 2)) - pow(n[2][1] + n[0][3], 2))

    return [h1, h2, h3, h4, h5, h6, h7]


# This method calculates the zernike with respect to formula.
# First input is the array which stores the connected arrays, second input and third input is the integers n and m
# Which zernike values that wanted to be calculated. I added that because i couldn't figure out how to do it in loop
# because zernike moments indexes doesn't increasing by one. Third input is the constant value.
# Return is the zernike moment of nm.
def calculate_zernike(arr, n, m, ONE):

    row, col = len(arr), len(arr[0])
    p = np.zeros(shape=(row, col), dtype=float)
    teta = np.zeros(shape=(row, col), dtype=float)
    delta_xi = np.zeros(shape=(row, col), dtype=float)
    delta_yj = np.zeros(shape=(row, col), dtype=float)

    for i in range(row - 1):
        n = row
        xi = (pow(2, 0.5) / (n - 1)) * i - 1 / pow(1, 0.5)
        for j in range(col - 1):
            n = col
            yj = (pow(2, 0.5) / (n - 1)) * j - 1 / pow(1, 0.5)
            p[i][j] = pow(pow(xi, 2) + pow(yj, 2), 0.5)
            teta[i][j] = math.atan(yj/xi)
            delta_xi[i][j] = 2 / n * pow(2, 0.5)
            delta_yj[i][j] = 2 / n * pow(2, 0.5)
    r = 0
    # In the below three lines i needed to cast it into integer because when i divide some value which is not dividible
    # python considers it like a float value, since i can't use float value to be upper limit.
    for k in range(int((n - abs(m)) / 2)):
        for l in range(int((n - abs(m)) / 2)):
            r += (pow(-1, l) * pow(p[k][l], n - 2 * l) * math.factorial(n - l)) / math.factorial(l) * math.factorial(int((n + abs(m) / 2)) - 2) * math.factorial(int((n - abs(m) / 2)) - 2)
    zr = 0
    zı = 0
    for i in range(row - 1):
        for j in range(col - 1):
            zr += ((n+1) / math.pi) * r * p[i][j] * math.cos(m * teta[i][j]) * (arr[i][j] / ONE) * delta_yj[i][j] * delta_xi[i][j]
            zı += (-1 * (n+1) / math.pi) * r * p[i][j] * math.cos(m * teta[i][j]) * (arr[i][j] / ONE) * delta_xi[i][j] * delta_yj[i][j]

    return math.sqrt(pow(zr, 2) + pow(zı, 2))


# This method calculates the r moment depending on hu moment values.
# Input is the array of moments. Return is the array that consists all the r moments.
def calculate_r_moment(feature_vectors):
    r1 = (feature_vectors[1] ** (1 / 2)) / feature_vectors[0]
    r2 = (feature_vectors[0] + (feature_vectors[1] ** (1 / 2))) / (feature_vectors[0] - (feature_vectors[1] ** (1 / 2)))
    r3 = (feature_vectors[2] ** (1 / 2)) / (feature_vectors[3] ** (1 / 2))
    r4 = (feature_vectors[2] ** (1 / 2)) / (abs(feature_vectors[4]) ** (1 / 2))
    r5 = (feature_vectors[3] ** (1 / 2)) / (abs(feature_vectors[4]) ** (1 / 2))
    r6 = abs(feature_vectors[5]) / (feature_vectors[0] * feature_vectors[2])
    r7 = abs(feature_vectors[5]) / (feature_vectors[0] * (abs(feature_vectors[4]) ** (1 / 2)))
    r8 = abs(feature_vectors[5]) / (feature_vectors[2] * (feature_vectors[1] ** (1 / 2)))
    r9 = abs(feature_vectors[5]) / ((feature_vectors[1] * abs(feature_vectors[4])) ** (1 / 2))
    r10 = abs(feature_vectors[4]) / (feature_vectors[2] * feature_vectors[3])
    return [r1, r2, r3, r4, r5, r6, r7, r8, r9, r10]


# This method draws the rectangle around the numbers depending on values of min x,y and max x,y values.
# First input is an array consists of array that stores the values. Second input is the image that rectangle will be drawn for.
# Return void, since it draws to the image.
def draw_rectangle(coordinates, new_img2):
    row, col = coordinates.shape[0], coordinates.shape[1]
    for i in range(row):
        for j in range(col):
            if coordinates[i][0] == 1:
                img3 = ImageDraw.Draw(new_img2)
                img3.rectangle(((coordinates[i][1], coordinates[i][2]), (coordinates[i][3], coordinates[i][4])))


# This method implemented by our instructor, which binarizes the arrays.
# First input is an integer that is the size of row, second input is also an integer that size of the collumn.
# Third input is the constant. Return is the image array that is binarized.
def binary_image(nrow,ncol,Value):
    x, y = np.indices((nrow, ncol))
    mask_lines = np.zeros(shape=(nrow,ncol))

    x0, y0, r0 = 30, 30, 10
    x1, y1, r1 = 70, 30, 10

    for i in range(50, 70):
        mask_lines[i][i] = 1
        mask_lines[i][i + 1] = 1
        mask_lines[i][i + 2] = 1
        mask_lines[i][i + 3] = 1
        mask_lines[i][i + 6] = 1
        mask_lines[i-20][90-i+1] = 1
        mask_lines[i-20][90-i+2] = 1
        mask_lines[i-20][90-i+3] = 1

    mask_square1 = np.fmax(np.absolute(x - x1), np.absolute(y - y1)) <= r1
    imge = np.logical_or(mask_lines, mask_square1) * Value

    return imge


# This method implemented by our instructor, which converts numpy array into the pil array.
# First input is the image's numpy array. Return is the pil array.
def np2PIL(im):
    print("size of arr: ", im.shape)
    img = Image.fromarray(im, 'RGB')
    return img

# This method implemented by our instructor, which adds color into pil array.
# The first input is the pil array format. Return is the colored pil array image.
def np2PIL_color(im):
    print("size of arr: ", im.shape)
    img = Image.fromarray(np.uint8(im))
    return img


# This method implemented by our instructor, converts numpy array into binary array. If the value is lower than t value
# It assigns the index of the array to low value, if not it assigns to the high value.
# The first input is the numpy array, t is the compare value of the array. Low value is our constant value. High is the
# upper boundary.
def threshold(im,T, LOW, HIGH):
    (nrows, ncols) = im.shape
    im_out = np.zeros(shape=im.shape)
    for i in range(nrows):
        for j in range(ncols):
            if abs(im[i][j]) < T:
                im_out[i][j] = LOW
            else:
                im_out[i][j] = HIGH
    return im_out


# This is the blob coloring eight connected algorithm mostly implemented by our instructor. To be precise he did implement
# The four connected and we added two extra if blocks to be eight connected. First input is the binarized image array
# Second input is the constant value. There are three returns, first return is the image array that is eight connected
# second return is the colored eight connected image array, last one is the k value that we used to connect.
def blob_coloring_8_connected(bim, ONE):
    # Here is the necessary variables and numpy arrays that needed to be declared.
    max_label = int(10000)
    nrow = bim.shape[0]
    ncol = bim.shape[1]
    print("nrow, ncol", nrow, ncol)
    im = np.zeros(shape=(nrow,ncol), dtype=int)
    a = np.arange(0, max_label, dtype=int)
    color_map = np.zeros(shape=(max_label, 3), dtype=np.uint8)
    color_im = np.zeros(shape=(nrow, ncol, 3), dtype=np.uint8)

    for i in range(max_label):
        np.random.seed(i)
        color_map[i][0] = np.random.randint(0, 255, 1, dtype=np.uint8)
        color_map[i][1] = np.random.randint(0, 255, 1, dtype=np.uint8)
        color_map[i][2] = np.random.randint(0, 255, 1, dtype=np.uint8)

    k = 0
    for i in range(nrow):
        for j in range(ncol):
            im[i][j] = max_label
    for i in range(1, nrow - 1):
        for j in range(1, ncol - 1):
                c = bim[i][j]
                l = bim[i][j - 1]
                u = bim[i - 1][j]
                # This four line stands for upper, left, upper left and upper right, we only need to check the upper ones
                # because when we increase the row by one in the loop it will check the upper ones too, so we are not
                # considering the below ones.
                label_u = im[i - 1][j]
                label_l = im[i][j - 1]
                label_u_l = im[i - 1][j - 1]
                label_u_r = im[i - 1][j + 1]
                # Here we are setting the all elements to the max label value in order to differentiate from the connects.
                im[i][j] = max_label
                if c == ONE:
                    min_label = min(label_u, label_l, label_u_l, label_u_r)
                    # Checking whether it has another value that needs to connected or not, if not we are increasing the k.
                    if min_label == max_label:
                        k += 1
                        im[i][j] = k
                    # If it has another value that needs to be connected we call update array function to do that.
                    else:
                        im[i][j] = min_label
                        if min_label != label_u and label_u != max_label:
                            update_array(a, min_label, label_u)
                        if min_label != label_u_l and label_u_l != max_label:
                            update_array(a, min_label, label_u_l)
                        if min_label != label_u_r and label_u_r != max_label:
                            update_array(a, min_label, label_u_r)
                        if min_label != label_l and label_l != max_label:
                            update_array(a, min_label, label_l)

                else:
                    im[i][j] = max_label
    # We are doing this to determine how many values that k can present and store it into numpy array.
    for i in range(k+1):
        index = i
        while a[index] != index:
            index = a[index]
        a[i] = a[index]
    # We are adding colors to the array so that our color_im numpy array becomes 3d array where third dimension
    # represents colors.
    for i in range(nrow):
        for j in range(ncol):

            if bim[i][j] == ONE:
                im[i][j] = a[im[i][j]]
                if im[i][j] == max_label:
                    im[i][j] == 0
                    color_im[i][j][0] = 0
                    color_im[i][j][1] = 0
                    color_im[i][j][2] = 0
                color_im[i][j][0] = color_map[im[i][j], 0]
                color_im[i][j][1] = color_map[im[i][j], 1]
                color_im[i][j][2] = color_map[im[i][j], 2]
    return im, color_im, k


# This method finds the min x,y and max x,y values that is connected by in order to calculate moments and draw rectangles.
# First input is the image array that consist of eight connected values, second one is the max k used, this is needed
# in order to store it into another array called coordinates with respect to k. Return is the coordinates which holds
# the x,y values and indexes indices that the k values.
def find_coordinates(im_label, max_k):
    coordinates = np.zeros(shape=(max_k,5), dtype=int)
    k = 1
    nrow = im_label.shape[0]
    ncol = im_label.shape[1]
    min_i, min_j, max_i, max_j = nrow, ncol, 0, 0
    while k < max_k:
        for i in range(1, nrow - 1):
            for j in range(1, ncol - 1):
                if im_label[i][j] == k:
                    if i < min_i:
                        min_i = i
                    if j < min_j:
                        min_j = j
                    if i > max_i:
                        max_i = i
                    if j > max_j:
                        max_j = j
                    coordinates[k][0] = 1
                    coordinates[k][1] = min_j
                    coordinates[k][2] = min_i
                    coordinates[k][3] = max_j
                    coordinates[k][4] = max_i
        min_i, min_j, max_i, max_j = nrow, ncol, 0, 0

        k = k + 1
    return coordinates


# This method updates the array in order to reduce k values. First input is the array that holds values.
# Second and third input which are label1 and label2 are the integer values that needs to be compared.
def update_array(a, label1, label2):
    index = lab_small = lab_large = 0
    if label1 < label2:
        lab_small = label1
        lab_large = label2
    else:
        lab_small = label2
        lab_large = label1
    index = lab_large
    while index > 1 and a[index] != lab_small:
        if a[index] < lab_small:
            temp = index
            index = lab_small
            lab_small = a[temp]
        elif a[index] > lab_small:
            temp = a[index]
            a[index] = lab_small
            index = temp
        else:
            break

    return


if __name__=='__main__':
    main()
