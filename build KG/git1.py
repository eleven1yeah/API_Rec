# -*- coding: utf-8 -*-
"""
Created on Tue May 21 14:23:04 2019
@author: 慕蒿
@email: muhaocs@yeah.net
"""
#批量下载GitHub开源代码
#Github-google.txt中是 获取目标对象所有开源项目网址

import requests


def getProject(url):
    response = requests.get(url)
    content = response.text
    '''
    content_ul = content.split('data-hovercard-url')[1]
    content_li = content_ul.split('<ul>')[1].split('</ul>')[0]
    projects = content_li.split('<h3 class="wb-break-all">')[1:]
    '''
    projects=content.split('data-hovercard-url')[1:]
    page_info = []
    for project in projects:
        project_info = []

        project_name = project.split('"')[1].split('/hovercard')[0].strip()
        project_url = 'https://github.com/' + project_name
        project_info.append(project_name)
        project_info.append(project_url)

        try:
            #project_language = project.split('"programmingLanguage">')[1].split('<')[0].strip()
            project_info.append('1')
        except:
            project_language = 'None'
            project_info.append(project_language)

        try:
            #project_star = project.split('stargazers')[1].split('</span>')[1].split('</a>')[0].strip()
            project_info.append('2')
        except:
            project_star = 'None'
            project_info.append(project_star)

        try:
            #project_memb = project.split('members')[1].split('</span>')[1].split('</a>')[0].strip()
            project_info.append('3')
        except:
            project_memb = 'None'
            project_info.append(project_memb)

        page_info.append(project_info)
    return page_info


fp = open('GitHub-google.txt', 'w+')
# 生成一个txt文件，存贮五个变量：项目名，项目网址，项目脚本语言，项目加星情况，项目共享数
fp.write('project_name\tproject_url\tproject_language\tproject_star\tproject_memb\n')
for page in range(41, 51):
    print('1')
    # 此处更改为你目标对象，我先下载了google公司贡献的1774条开源项目
    url = 'https://github.com/google?page=' + str(page)
    page_info = getProject(url)
    print('2')
    for li in page_info:
        fp.write(li[0] + '\t')
        fp.write(li[1] + '\t')
        fp.write(li[2] + '\t')
        fp.write(li[3] + '\t')
        fp.write(li[4] + '\n')
    print(page, 'Done!')
fp.close()
