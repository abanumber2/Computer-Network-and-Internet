import cv2
import numpy as np
import os
import myqrcode
import x
from CRC import CRC_Encoding


def imgToVideo(outputFileName, num):
    fps = 20  # 视频帧数
    size = (x.width * 9, x.width * 9)  # 需要转为视频的图片的尺寸

    # 使用 H264 编解码器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'mp4v' 是常用于 MP4 的编解码器
    video = cv2.VideoWriter(outputFileName, fourcc, fps, size)

    for i in range(num):
        img_path = f"./video/{i}.png"
        if not os.path.exists(img_path):
            print(f"Warning: Image file {img_path} does not exist.")
            continue
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error: Failed to load image {img_path}.")
            continue
        if img.shape[1] != size[0] or img.shape[0] != size[1]:
            print(f"Error: Image {i}.png has incorrect size.")
            continue
        video.write(img)

    video.release()


def main():
    inputFileName = "e2.bin"
    outputFileName = "./e2/in.mp4"  # 输出 MP4 格式的视频
    temp = input()
    temp = temp.split(" ")
    inputFileName = temp[0]
    outputFileName = temp[1]
    ms = temp[2]

    with open(inputFileName, 'rb') as reader:
        data = reader.read()

    binstring = ""
    for ch in data:
        binstring += CRC_Encoding('{:08b}'.format(ch), x.key)
        if len(binstring) > int(int(ms) * 908000 / 1700):
            break

    startOrEnd = 170
    startOrEndStr = ''.join(['{:08b}'.format(startOrEnd)] * 8)
    binstring = startOrEndStr + binstring + startOrEndStr

    myqrcode.genBlankFrame()
    num = 1
    x.datalen = len(binstring)
    print(len(binstring))

    while x.dataindex < len(binstring):
        mat = np.full((x.width, x.width, 3), 255, dtype=np.uint8)
        myqrcode.drawLocPoint(mat)
        binstring = myqrcode.encode(mat, binstring)
        myqrcode.genImage(mat, x.width * 9, f"./video/{num}.png")
        num += 1
        print(x.dataindex)

    imgToVideo(outputFileName, num)


if __name__ == '__main__':
    main()
