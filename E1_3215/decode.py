import cv2
# import cv
import sys
import numpy as np
from myqrcode import demask, getContours, find
import x
import struct
from CRC import CRC_Decoding

global temp
first = 0
end = 0


def JReduce(image, m, n):
    """
    对输入图像进行下采样。
    """
    H = int(image.shape[0] * m) + 1
    W = int(image.shape[1] * n)
    H = 102
    W = 102
    size = (W, H, 3)
    iJReduce = np.zeros(size, np.float32)

    for i in range(H):
        for j in range(W):
            x1 = int(i / m)
            x2 = int((i + 1) / m)
            y1 = int(j / n)
            y2 = int((j + 1) / n)
            sum = [0, 0, 0]
            num = 0
            for k in range(x1 + 2, x2 - 3):
                for l in range(y1 + 2, y2 - 3):
                    num += 1
                    sum[0] += image[k, l][0]
                    sum[1] += image[k, l][1]
                    sum[2] += image[k, l][2]
            iJReduce[i][j] = [sum[0] / num, sum[1] / num, sum[2] / num]
    return iJReduce


def checkStart(img):
    """
    检查图像以确定解码是否开始。
    """
    global first
    contours, hierachy = getContours(img)
    img = find(img, contours, np.squeeze(hierachy))  # 对空图片进行检测，以确定解码开始
    binstring = ""
    decode(img, binstring)
    return


def decode(image, binstring):
    """
    解码图像中的二维码数据。
    """
    width = x.width - 8
    xxx = np.full((width, width, 3), 0, dtype=np.float32)
    mat = np.full((width, width, 3), 0, dtype=np.float32)
    pwidth = 10
    i = 0

    xxx = JReduce(image, 0.111111111111111, 0.111111111111111)
    tempb, tempg, tempr = cv2.split(xxx)
    tempr = np.array(tempr, dtype="uint8")
    tempg = np.array(tempg, dtype="uint8")
    tempb = np.array(tempb, dtype="uint8")
    ret, tempr = cv2.threshold(tempr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ret, tempg = cv2.threshold(tempg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ret, tempb = cv2.threshold(tempb, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mat = cv2.merge([tempb, tempg, tempr])

    row = 0
    thre = [150, 150, 150]
    col = x.locWidth
    count = 0
    while row < width:
        if row < x.locWidth:
            if (row + col) % 2 == 0:
                demask(mat, row, col, count, thre[count])
            if mat[row][col][count] > thre[count]:
                binstring += "0"
            else:
                binstring += "1"
            count += 1
            if count >= 3:
                count -= 3
                col += 1
                if col > width - x.locWidth - 1:
                    if row != x.locWidth - 1:
                        col = x.locWidth
                    else:
                        col = 0
                    row += 1
        elif row < width - x.locWidth:
            if (row + col) % 2 == 0:
                demask(mat, row, col, count, thre[count])
            if mat[row][col][count] > thre[count]:
                binstring += "0"
            else:
                binstring += "1"
            count += 1
            if count >= 3:
                count -= 3
                col += 1
                if col > width - 1:
                    if row != width - x.locWidth - 1:
                        col = 0
                    else:
                        col = x.locWidth
                    row += 1
        elif row < width - x.sLocWidth:
            if (row + col) % 2 == 0:
                demask(mat, row, col, count, thre[count])
            if mat[row][col][count] > thre[count]:
                binstring += "0"
            else:
                binstring += "1"
            count += 1
            if count >= 3:
                count -= 3
                col += 1
                if col > width - 1:
                    col = x.locWidth
                    row += 1
        else:
            if (row + col) % 2 == 0:
                demask(mat, row, col, count, thre[count])
            if mat[row][col][count] > thre[count]:
                binstring += "0"
            else:
                binstring += "1"
            count += 1
            if count >= 3:
                count -= 3
                col += 1
                if col > width - x.sLocWidth - 1:
                    col = x.locWidth
                    row += 1

    startOrEnd = 170
    startOrEndStr = ""
    for i in range(8):
        startOrEndStr += '{:08b}'.format(startOrEnd)

    global first
    global end

    if first == 0 and binstring[:64] != startOrEndStr:
        return
    elif first == 0 and binstring[:64] == startOrEndStr:
        binstring = binstring[64:]
        first = 1
    elif first == 1 and binstring[:64] == startOrEndStr:
        binstring = binstring[64:]
    elif first == 1 and binstring.find(startOrEndStr) != -1:
        if binstring[:64] != startOrEndStr:
            binstring = binstring[:binstring.find(startOrEndStr)]
            end = 1
    print(binstring)
    return binstring


def decodeFromVideo(filename):
    """
    从视频文件中解码二维码数据。
    """
    global end
    global first

    binstring = ""
    vc = cv2.VideoCapture(filename)
    if vc.isOpened():
        rval = True
    else:
        rval = False
    fps = 3
    k = 1
    count = 1
    record = 0

    while rval:
        rval, frame = vc.read()
        if frame is None:
            break
        if first == 0:
            checkStart(frame)
            if first == 1:
                record = k + 1
        elif (k - record) % fps == 0:
            contours, hierachy = getContours(frame)
            img = find(frame, contours, np.squeeze(hierachy))
            binstring = decode(img, binstring)
            binstring = wirteResult(binstring)
            count += 1
        cv2.imwrite("./output/" + str(k) + ".png", frame)
        k += 1
        if end == 1:
            break

    vc.release()
    return


def wirteResult(binstring):
    """
    将解码后的二进制字符串写入文件。
    """
    global temp
    outputFileName = "./output/output.bin"
    check = "./output/valid.val"

    writer = open(outputFileName, 'ab+')
    writerCheck = open(check, 'ab+')

    while len(binstring) >= 11:
        if CRC_Decoding(binstring[:11], x.key) == True:
            writerCheck.write(struct.pack('B', 255))
        else:
            writerCheck.write(struct.pack('B', 0))

        t = int(binstring[:8], 2)
        res = struct.pack('B', t)
        writer.write(res)
        binstring = binstring[11:]

    writer.close()
    return binstring


if __name__ == '__main__':
    inputFileName = "e1/in.mp4"
    global temp
    temp = input()
    temp = temp.split(" ")
    inputFileName = temp[0]
    decodeFromVideo(inputFileName)
