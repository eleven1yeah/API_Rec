#用python根据关键字爬取Github上包含某关键字的代码链接

import json
from urllib.request import urlopen

import requests
from bs4 import BeautifulSoup
from requests import Request


class github_crawl():

    def __init__(self):
        # 初始化一些必要的参数
        self.login_headers = {
            "Referer": "https://github.com/",
            "Host": "github.com",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/65.0.3325.181 Safari/537.36"
        }

        self.logined_headers = {
            "Referer": "https://github.com/login",
            "Host": "github.com",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/65.0.3325.181 Safari/537.36"
        }

        self.headers = {
            'User-Agent': 'Mozilla/5.0',
            'Authorization': 'token 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',  # 换上自己的token认证
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
        self.login_url = "https://github.com/login"
        self.post_url = "https://github.com/session"
        self.session = requests.Session()

    def parse_loginPage(self):
        # 对登陆页面进行爬取，获取token值
        html = self.session.get(url=self.login_url, headers=self.login_headers, verify=False)
        Soup = BeautifulSoup(html.text, "lxml")
        token = Soup.find("input", attrs={"name": "authenticity_token"}).get("value")

        return token

    # 获得了登陆的一个参数

    def login(self, user_name, password, keyword):
        # 传进必要的参数，然后登陆
        post_data = {
            "commit": "Sign in",
            "utf8": "✓",
            "authenticity_token": self.parse_loginPage(),
            "login": user_name,
            "password": password
        }

        logined_html = self.session.post(url=self.post_url, data=post_data, headers=self.logined_headers, verify=False)
        if logined_html.status_code == 200:
            self.parse_keyword(keyword)  # 获取了页面

    def parse_keyword(self, keyword):
        # 解析登陆以后的页面，筛选出包含这个关键字的python代码
        user_repositorys = set()  # 集合用来存放作者名和库名
        try:

            for i in range(101):
                url = "https://github.com/search?l=Python&p={id}&q={keyword}&type=Code".format(id=i + 1,
                                                                                               keyword=keyword)  # 循环的爬取页面信息
                requests.packages.urllib3.disable_warnings()
                resp = self.session.get(url=url, headers=self.login_headers, verify=False)
                soup = BeautifulSoup(resp.text, "lxml")
                pattern = soup.find_all('a', class_='text-bold')

                for item in pattern:
                    user_repositorys.add(item.string)
            for user_repository in user_repositorys:
                self.get_results(user_repository, keyword)

        except Exception as e:
            print(e)

    def get_results(self, repository, keyword):  # 用Github的官方API爬取数据，解析json
        url = "https://api.github.com/search/code?q={keyword}+in:file+language:python+repo:{w}".format(w=repository,
                                                                                                       keyword=keyword)
        try:
            req = Request(url, headers=self.headers) #需要token认证
            response = urlopen(req).read()
            results = json.loads(response.decode())
            for item in results['items']:
                repo_url = item["repository"]["html_url"]  #得到项目库的链接
                file_path = item['html_url'] #得到代码的链接
                fork = item["repository"]["fork"] #是否引用
                self.loader_csv(repo_url, file_path, fork, keyword)

        except Exception as e:
            print("获取失败")

    def loader_csv(self, repo_url, file_path, keyword): #写入csv文件
        try:
            with open("path", "a") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow([keyword, repo_url, file_path])
            csv_file.close()
        except Exception as e:
            print(e)


if __name__ == "__main__":
    x = github_crawl()
    x.login("user", "password", "keyword")
