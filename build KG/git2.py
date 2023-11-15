# -*- coding: utf-8 -*-
"""
Created on Tue May 21 14:23:04 2019

@author: 慕蒿
@email: muhaocs@yeah.net
"""

#利用git1获得的项目网址批量下载开源代码

import requests
import urllib3
from urllib import request


def download(url, save_path):
    file_name = url.split('/')[-1]

    response = requests.get(url)
    content = response.text
    btn = content.split(' data-open-app="link" ')[1]
    zip_url = 'https://github.com' + btn.split('href="')[1].split('"')[0].strip()

    try:
        request.urlretrieve(zip_url, file_name + '.zip')
    except:
        try:
            with open(save_path + file_name + '.zip', 'wb') as code:
                code.write(requests.get(zip_url).content)
        except:
            try:
                http = urllib3.PoolManager()
                r = http.request('GET', zip_url)
                with open(save_path + file_name + '.zip', 'wb') as code:
                    code.write(r.data)
            except:
                return file_name + ' 下载失败!!!!!!'
    return file_name + ' 下载完成!'


save_path = "E:\Dataset\python\python"
fp = open('topPythonRepos.txt', 'r')
for line in fp.readlines():
    url = line.strip()
    try:
        print(download(url, save_path))
    except:
        print(url, '下载失败!!!!!!')
fp.close()
