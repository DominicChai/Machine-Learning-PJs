#!/usr/bin/python
import re
f1 = open('D:/BI/vivomidcut.txt')
f2 = open('D:/BI/vivomidcut1.txt','r+')
for s in f1.readlines():
    f2.write(s.replace('\'' ,' '))#去除符号’
f1.close()
f2.close()

f2 = open('D:/BI/vivomidcut1.txt')
f3 = open('D:/BI/vivomidcut2.txt','r+')
for s in f2.readlines():
    f3.write(s.replace(']' ,' '))#去除符号]
f2.close()
f3.close()

f3 = open('D:/BI/vivomidcut2.txt')
f4 = open('D:/BI/vivomidcut3.txt','r+')
for s in f3.readlines():
    f4.write(s.replace('[' ,' '))#去除符号[
f3.close()
f4.close()

f4 = open('D:/BI/vivomidcut3.txt')
f5 = open('D:/BI/vivomidcut4.txt','r+')
for s in f4.readlines():
    f5.write(s.replace(',' ,' '))#去除符号,
f4.close()
f5.close()

f5 = open('D:/BI/vivomidcut4.txt')
f6 = open('D:/BI/vivomidcut5.txt','r+')
for s in f5.readlines():
    f6.write(s.replace('"' ,' '))#去除符号”
f5.close()
f6.close()
