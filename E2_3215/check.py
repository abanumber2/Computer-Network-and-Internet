import sys
import x
from CRC import CRC_Encoding
import chardet


if __name__ == '__main__':
    FileName1 = "1.bin"
    FileName2 = "./output/output.bin"
    FileName3 = "./output/1.val"
    videotime = 1700
    temp=input()
    temp=temp.split(" ")
    FileName1 = temp[0]
    
    with open(FileName1, 'rb') as reader:
        data = reader.read()
    binstring1 = ""

    for ch in data:

        binstring1 += '{:08b}'.format(ch)
        if (len(binstring1) > 908000):
            break

    with open(FileName2, 'rb') as reader:
        data = reader.read()
    binstring2 = ""

    for ch in data:
        binstring2 += '{:08b}'.format(ch)

    with open(FileName3, 'rb') as reader:
        data = reader.read()
    binstring3 = ""
    print("总传输量(b):" + str(len(binstring2)))
    a = 0
    b = 0

    for ch in data:
        if (int(ch) == 0):
            a = a + 1
        else:
            b = b + 1
    print("丢失率：" + str(a/(a+b)/8))

    binstring3 = data

    lb2 = len(binstring2)
    b2 = 0
    c2 = 0
    d2 = 0
    for i in range(len(binstring2)):
        # print(int(binstring2[i]))
        if (int(binstring3[i // 8]) == 1):
            c2 = c2 + 1
            if (int(binstring2[i]) == int(binstring1[i])):
                b2 = b2 + 1
            else:
                d2 = d2 + 1

    print("误码率：" + str(d2 / len(binstring2)))
    print(d2)
    b2 = 0
    c2 = 0
    for i in range(len(binstring2)):
        # print(int(binstring2[i]))
        if (int(binstring3[i // 8]) == 1):
            c2 = c2 + 1
            if (int(binstring2[i]) == int(binstring1[i])):
                b2 = b2 + 1
            else:
                break

    print("有效传输量(b):" + str(len(binstring2)-d2))
    tran = int(len(binstring2)-d2)/8*11 / (int(videotime) / 1000) / 1024
    print("有效传输率(kbps)：" + str(tran))
