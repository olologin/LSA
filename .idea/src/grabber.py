import requests
import string
import os

API_URL = 'http://ru.wikipedia.org/w/api.php'

def removeSections(bad_sections, text):
    '''
    :param bad_sections: list of bad sections of article which we want to delete, for example:
     ['См. также', 'Примечания', 'Ссылки', 'Литература']
    :param text: text with bad content sections
    :return: text without them :D
    '''
    res = text
    for content in bad_sections:
        idx1 = res.find('��2��'+content)
        if(idx1 == -1):
            continue
        idx2 = res.find('��2��', idx1+1)
        if(idx2 == -1):
            res = res[:idx1]
        else:
            res = res[:idx1]+res[idx2:]

    return res


def grab(themes, maximumArticlesPerTheme):
    '''
    :param themes: list of themes
    :param maximumArticlesPerTheme: maximum number of parsed articles per theme
    :return: list of articles which represented by tuples
            (Theme_id, Article_name, id, Text_Of_Article with wikimedia content sections)
    '''
    res = []
    for theme_idx, theme in enumerate(themes):
        search_results = requests.get(API_URL,
                                      params={
                                          "format":"json",
                                          "action":"opensearch",
                                          "search":theme,
                                          "limit":str(maximumArticlesPerTheme)
                                      }
        ).json()
        print("In theme: " + theme)
        for search_result_idx, search_result in enumerate(search_results[1]):
            print("\t getting page: " + search_result)
            response = requests.get(API_URL,
                                    params={
                                        "format":"json",
                                        "action":"query",
                                        "prop":"extracts",
                                        "explaintext":"",
                                        "indexpageids":"",
                                        "titles":search_result,
                                        "exsectionformat":"raw",
                                        "redirects":"true"
                                    }
            ).json()
            pageids = response['query']['pageids']
            if len(pageids) != 1:
                raise Exception("Odd error");
            for pageid in pageids:
                extract = response['query']['pages'][pageid]['extract']
                title = response['query']['pages'][pageid]['title']
                res.append((theme_idx, title, pageid, extract))
    return res

def save_articles(path, articles):
    for themeid, articlename, articleid, text in articles:
        print(path + str(themeid) + '|' + articlename + '|' + articleid + '.txt')
        stream = open(path + str(themeid) + '|' + articlename + '|' + articleid + '.txt', 'w')
        stream.write(text)
        stream.close()

def read_articles(path):
    articles = []
    for fileName in os.listdir(path):
         info = fileName.replace('.txt','').split('|')
         themeid = int(info[0])
         articlename = info[1]
         articleid = info[2]
         stream = open(path + fileName, 'r')
         text = stream.read()
         articles.append((themeid, articlename, articleid, text))
    return articles