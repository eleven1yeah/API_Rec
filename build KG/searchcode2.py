import requests

f= open('searchcode.txt' ,'r')
contents=f.readlines()[1:]
for line in contents:
    linecontent=line.split('	')
    filename=linecontent[0]
    filename_split=filename.split('/')
    filename=""
    for word in filename_split:
        filename+=word+'_'
    url=linecontent[2]
    response = requests.get(url)
    urlcontent = response.text
    fp = open('code/'+filename+'.txt', 'w',encoding='utf-8')
    try:
        fp.write(urlcontent)
    except:
        print(filename+"创建失败")
    fp.close
    print("line end")
f.close()