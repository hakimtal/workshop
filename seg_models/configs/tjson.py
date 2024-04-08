

import json
import sys

path = sys.argv[1]
# 读取文件数据

configs = json.load(open(path, "r"))
# 读取每一条json数据
for d in configs:
    print(d, configs[d])


def aaa():
    print("*" * 25)
    print(configs["pretrained_model_path"] % 1)
    print("*" * 25)


if __name__ == "__main__":
    print(eval("True"))
    # print(eval("true"))
    print(eval("False"))
    # print(eval("false"))
    p = configs["pretrained_model_path"] % 255
    print(p)
    print(configs["RESIZE_SIZE"])

    aaa()
