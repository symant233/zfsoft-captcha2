# -*- coding: utf-8 -*-
"""
Created on 2018/5/20

@author: susmote
"""

kv_dict = {}
with open('../right_code.txt') as f:
    for value in f:
        value = value.strip()
        for i in value:
            kv_dict.setdefault(i, 0)
            kv_dict[i] += 1
print(kv_dict)
{'r': 38, 'h': 50, '0': 32, 'j': 42, 'i': 31, 'v': 40, '1': 37, '2': 32, 'p': 38, '6': 31, 'l': 33, 'q': 29, 'a': 34, 'e': 47, 's': 38, '5': 36, 'u': 24,
    'd': 38, 'y': 35, 'c': 40, 'm': 42, 'f': 31, '8': 40, 'b': 35, 't': 47, 'n': 42, 'w': 35, '7': 47, 'g': 28, 'x': 37, '4': 30, '3': 25, 'k': 35, '9': 1}

print(len(kv_dict)) # 34
