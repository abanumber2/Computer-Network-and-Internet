import sys
import x
from CRC import CRC_Encoding
import chardet
"""
处理视频文件的数据，计算传输数据的有效性、误码率和丢失率，并估算有效传输率
"""
if __name__ == '__main__':
    FileName1 = "e4.bin"
    FileName2 = "./output/output.bin"
    FileName3 = "./output/valid.bin"
    videotime = 1700

    temp = input()
    temp = temp.split(" ")
    FileName1 = temp[0]
    FileName2 = temp[1]
    FileName3 = temp[2]
    videotime = temp[3]

    # 读取 FileName1 中的数据并转换为二进制字符串
    with open(FileName1, 'rb') as reader:
        data = reader.read()
    binstring1 = ""
    for ch in data:
        binstring1 += '{:08b}'.format(ch)
        if len(binstring1) > 908000:
            break

    # 读取 FileName2 中的数据并转换为二进制字符串
    with open(FileName2, 'rb') as reader:
        data = reader.read()
    binstring2 = ""
    for ch in data:
        binstring2 += '{:08b}'.format(ch)

    # 读取 FileName3 中的数据并计算丢失率
    with open(FileName3, 'rb') as reader:
        data = reader.read()
    binstring3 = ""
    print("总传输量(b):" + str(len(binstring2)))

    a = 0
    b = 0
    for ch in data:
        if int(ch) == 0:
            a += 1
        else:
            b += 1
    print("丢失率：" + str(a / (a + b) / 8))

    binstring3 = data

    lb2 = len(binstring2)
    b2 = 0
    c2 = 0
    d2 = 0
    for i in range(len(binstring2)):
        if int(binstring3[i // 8]) == 1:
            c2 += 1
            if int(binstring2[i]) == int(binstring1[i]):
                b2 += 1
            else:
                d2 += 1

    print("误码率：" + str(d2 / len(binstring2)))
    print(d2)

    b2 = 0
    c2 = 0
    for i in range(len(binstring2)):
        if int(binstring3[i // 8]) == 1:
            c2 += 1
            if int(binstring2[i]) == int(binstring1[i]):
                b2 += 1
            else:
                break

    print("有效传输量(b):" + str(len(binstring2) - d2))
    tran = int(len(binstring2) - d2) / 8 * 11 / (int(videotime) / 1000) / 1024
    print("有效传输率(kbps)：" + str(tran))
