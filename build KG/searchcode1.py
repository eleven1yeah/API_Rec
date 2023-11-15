# -*- coding: utf-8 -*-
"""
Created on Tue May 21 14:23:04 2019
@author: 慕蒿
@email: muhaocs@yeah.net
"""

import requests


def getProject(url):
    response = requests.get(url)
    content = response.text
    '''
    content_ul = content.split('data-hovercard-url')[1]
    content_li = content_ul.split('<ul>')[1].split('</ul>')[0]
    projects = content_li.split('<h3 class="wb-break-all">')[1:]
    '''
    projectcontnent=content.split('<div style="display:flex;">')[1]
    projectcontnent_c=projectcontnent.split(("width:100%;"))[1]
    projects=projectcontnent_c.split(('</pre>'))[0:-1]
    page_info = []
    for project in projects:
        project_info = []

        project_name = project.split('<a href="')[1].split('"')[0].strip()
        project_url = 'https://searchcode.com' + project_name
        projectraw='https://searchcode.com/codesearch/raw/'+project.split('<a href="/file/')[1].split('/')[0].strip()+'/'


        project_info.append(project_name)
        project_info.append(project_url)

        try:
            #project_language = project.split('"programmingLanguage">')[1].split('<')[0].strip()
            project_info.append(projectraw)
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


fp = open('searchcode.txt', 'w+')
# 生成一个txt文件，存贮五个变量：项目名，项目网址，项目脚本语言，项目加星情况，项目共享数
fp.write('project_name\tproject_url\tproject_language\tproject_star\tproject_memb\n')
for page in range(1, 3):

    # 此处更改为你目标对象，我先下载了google公司贡献的1774条开源项目
    url = 'https://searchcode.com/?q=python&p=' + str(page)+"&lan=19"
    page_info = getProject(url)

    for li in page_info:
        fp.write(li[0] + '\t')
        fp.write(li[1] + '\t')
        fp.write(li[2] + '\t')
        fp.write(li[3] + '\t')
        fp.write(li[4] + '\n')
    print(page, 'Done!')
fp.close()
