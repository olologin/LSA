import stemmer
import grabber
import re
import numpy as np
from scipy import linalg
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy import spatial
import matplotlib.pyplot as plt

# TODO ё и -
# Habr ML
def cosineDistance(u, v):
    return (1+(np.inner(u,v)/np.sqrt(np.inner(u,u)*np.inner(v,v))))*0.5

def stem(text):
    res = ''
    # ё и - неверно обрабатываются регеспом ниже, заменим их вручную
    text = text.replace('ё', 'е')
    text = text.replace('-', ' ')
    text = text.lower()
    for word in re.split('[^а-яА-Я]', text):
        if len(word) <= 1:
            continue
        try:
            res = res + ' ' + stemmer.Porter.stem(word=word)
        except AttributeError as aerr:
            res = res + ' ' + word
            print(word)
    return  res

def tf_idf(articles):
    '''
    :param articles: список статей представленных в виде кортежа
            (Theme_id, Article_name, id, Stemmed_Text_Of_Article)
    :return: кортеж с тремя значениями:
            1. TF-IDF матрица
            2. idx2word список который для любого индекса строки
                позволяет узнать соответствующее слово
            3. idx2ArticleInfo который для любого индекса столбца
                позволяет узнать соответствующую статью
                (Содержит имя статьи, id, id темы)
    '''
    howMuchArticlesWithWord = dict()
    D = len(articles)
    for _, _, _, text in articles:
        wordsUsedInArticle = {word for word in text.split()}
        for word in wordsUsedInArticle:
            if word not in howMuchArticlesWithWord:
                howMuchArticlesWithWord[word] = 1
            else:
                howMuchArticlesWithWord[word] = howMuchArticlesWithWord[word]+1

    word2idx = dict({(key, idx) for idx, key in enumerate(sorted(howMuchArticlesWithWord.keys()))})
    idx2ArticleInfo = [None]*D

    matrix = np.zeros((len(word2idx), D), dtype='d')
    for idx, (themeid, articlename, articleid, text) in enumerate(articles):
        idx2ArticleInfo[idx] = (themeid, articlename, articleid)
        listOfWords = text.split()
        for word in set(listOfWords):
            TF = float(listOfWords.count(word))/len(listOfWords)
            IDF = np.log(float(D)/howMuchArticlesWithWord[word])
            matrix[word2idx[word], idx] = TF*IDF


    idx2word = [None]*len(word2idx)
    for k, v in word2idx.items():
        idx2word[v] = k

    return matrix, idx2word, idx2ArticleInfo

def printMostRelevantWordsToArticles(matrix, idx2word, idx2ArticleInfo):
    for idx, (themeid, articlename, articleid) in enumerate(idx2ArticleInfo):
        print(articlename + '|' + idx2word[matrix[:, idx].argmax()] + '|' + str(matrix[:, idx].max()))

def getReversedIndex(U, Vh):
    res = np.zeros((U.shape[0], Vh.shape[1]), dtype='d')
    cntWordZeros = 0
    zeroWords = set();
    # Цикл по словам
    for rowIdx in range(U.shape[0]):
        for colIdx in range(Vh.shape[1]):
            res[rowIdx, colIdx] = cosineDistance(U[rowIdx,:].flatten(), Vh[:,colIdx].flatten())
    return res

def searchByWords(reversedIndex, idx2ArticleInfo, idx2word, searchString):
    stemmedSearchString = stem(searchString)
    distances = np.zeros(len(idx2ArticleInfo))
    #howMuchWordsInSearchString = np.log(len(stemmedSearchString.split(' ')))
    howMuchWordsInSearchString = len(stemmedSearchString.split(' '))
    for word in stemmedSearchString.split(' '):
        if(word == ''):
            continue
        wordIdx = idx2word.index(word)
        distances = distances + reversedIndex[wordIdx,:]
    distances = distances
    res = [(idx2ArticleInfo[articleIdx], distances[articleIdx]) for articleIdx in reversed(distances.argsort()[-4:])]
    return res

def plotDocumentsAfterSVD(Vh, idx2articleInfo):
    # Циклом по стоблцам третьей матрицы
    for i in range(Vh.shape[1]):
        themeid = idx2ArticleInfo[i][0]
        # Рисуем координаты из первой и второй строки заданного столбца
        plt.plot(Vh[0,i], Vh[1,i], color=COLORS[themeid], marker='o')
    plt.show()

def plotDendrogram(matrix, idx2ArticleInfo):
    '''
    :param matrix: Матрица где каждый столбец соответствует статье
    :param idx2ArticleInfo:
    :return:
    '''
    res = linkage(matrix.T, metric='cosine')
    fig = plt.figure()
    names = [articlename for (themeid, articlename, articleid) in idx2ArticleInfo]
    np.clip(res, 0, res.max(), res)
    dendrogram(res, labels=names, orientation='left', leaf_font_size=5)#,leaf_font_size=15)#, leaf_rotation=45)
    fig.set_size_inches(18.5,37)
    fig.savefig('tree.png',dpi=150, bbox_inches='tight')

THEMES = ['Техника', 'Таврия', 'Биржа', 'Депутат', 'Газ']
COLORS = ['red',     'green',   'blue', 'yellow',  'cyan']
if __name__ == "__main__":
    unfiltered_articles = grabber.grab(THEMES, 100)
    grabber.save_articles("../Texts/", unfiltered_articles)
    unfiltered_articles = grabber.read_articles("../Texts/")
    BAD_SECTIONS = ['См. также', 'Примечания', 'Ссылки', 'Литература']
    filtered_articles = list(map(
        lambda tuple: (tuple[0], tuple[1], tuple[2], stem(grabber.removeSections(BAD_SECTIONS, tuple[3]))),
        unfiltered_articles
    ));
    grabber.save_articles("../TextsStemmed/", filtered_articles)

    stemmed_articles = grabber.read_articles("../TextsStemmed/")
    matrix, idx2word, idx2ArticleInfo = tf_idf(stemmed_articles)
    print("Наиболее релевантные слова к каждой статье после tf-idf:")
    printMostRelevantWordsToArticles(matrix, idx2word, idx2ArticleInfo)
    # Далее - работа самого LSA с последующим построением обратного чемпионского списка

    U,s,Vh = linalg.svd(matrix, full_matrices=False)
    plotDendrogram(Vh[0:9,:], idx2ArticleInfo)
    reversedIndex = getReversedIndex(U[:,0:9], Vh[0:9,:])
    print (searchByWords(reversedIndex, idx2ArticleInfo, idx2word, "география крым"))
    print (searchByWords(reversedIndex, idx2ArticleInfo, idx2word, "автомобиль заз"))
    print (searchByWords(reversedIndex, idx2ArticleInfo, idx2word, "таврия"))
    print (searchByWords(reversedIndex, idx2ArticleInfo, idx2word, "футбол"))
    print (searchByWords(reversedIndex, idx2ArticleInfo, idx2word, "двигатель"))
    print (searchByWords(reversedIndex, idx2ArticleInfo, idx2word, "Депутатский мандат"))
    print (searchByWords(reversedIndex, idx2ArticleInfo, idx2word, "Двигатель внутреннего сгорания"))
    print (searchByWords(reversedIndex, idx2ArticleInfo, idx2word, "крым"))
    print (searchByWords(reversedIndex, idx2ArticleInfo, idx2word, "религии мира"))
    print (searchByWords(reversedIndex, idx2ArticleInfo, idx2word, "древнее искусство"))
    print (searchByWords(reversedIndex, idx2ArticleInfo, idx2word, "древняя медицина"))
    print (searchByWords(reversedIndex, idx2ArticleInfo, idx2word, "страны европы"))
    print (searchByWords(reversedIndex, idx2ArticleInfo, idx2word, "страны америки"))
    print (searchByWords(reversedIndex, idx2ArticleInfo, idx2word, "страны азии"))
    print (searchByWords(reversedIndex, idx2ArticleInfo, idx2word, "20-ый век"))

    pass


