import stemmer
import grabber
import re
import numpy as np
from scipy import linalg
from scipy import spatial
import matplotlib.pyplot as plt

# TODO ё и -
# Habr ML
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
    wordCounterAmongArticle = dict()
    D = len(articles)
    for _, _, _, text in articles:
        wordsUsedInArticle = {word for word in text.split(" ")}
        for word in wordsUsedInArticle:
            if word not in howMuchArticlesWithWord:
                howMuchArticlesWithWord[word] = 1
            else:
                howMuchArticlesWithWord[word] = howMuchArticlesWithWord[word]+1

    word2idx = dict({(key, idx) for idx, key in enumerate(howMuchArticlesWithWord.keys())})
    idx2ArticleInfo = [None]*D

    matrix = np.zeros((len(word2idx), D), dtype=float)
    for idx, (themeid, articlename, articleid, text) in enumerate(articles):
        idx2ArticleInfo[idx] = (themeid, articlename, articleid)
        listOfWords = text.split(" ")
        for word in set(listOfWords):
            TF = listOfWords.count(word)/len(listOfWords)
            IDF = np.log(D/howMuchArticlesWithWord[word])
            matrix[word2idx[word], idx] = TF*IDF

    idx2word = [None]*len(word2idx)
    for k, v in word2idx.items():
        idx2word[v] = k

    return matrix, idx2word, idx2ArticleInfo

def printMostRelevantWordsToArticles(matrix, idx2word, idx2ArticleInfo):
    for idx, (themeid, articlename, articleid) in enumerate(idx2ArticleInfo):
        print(articlename + '|' + idx2word[matrix[idx].argmax()] + '|' + str(matrix[idx].max()))

def getReversedIndex(matrix, idx2word):
    U,s,Vh = linalg.svd(matrix)
    res = dict()
    # TODO
    for rowIdx in range(U.shape[0]):
        # print(idx2word[rowIdx])
        distances = np.zeros(Vh.shape[1])
        wordVec = U[rowIdx,:2]
        for colIdx in range(Vh.shape[1]):
            docVec = Vh[:2,colIdx]
            distances[colIdx] = spatial.distance.cosine(wordVec, docVec)
        res[idx2word[rowIdx]] = distances
    return res

def searchByWords(reversedIndex, idx2ArticleInfo, searchString):
    stemmedSearchString = stem(searchString)
    distances = np.zeros(len(idx2ArticleInfo))
    for word in stemmedSearchString.split(' '):
        distances = distances + reversedIndex[word]
    return idx2ArticleInfo[distances.argmax()]

def plotDocumentsAfterSVD(matrix, idx2articleInfo):
    U,s,Vh = linalg.svd(matrix)
    print(U.shape)
    print(Vh.shape)
    # Циклом по стоблцам третьей матрицы
    for i in range(Vh[0,:].shape[0]):
        themeid = idx2ArticleInfo[i][0]
        # Рисуем координаты из первой и второй строки заданного столбца
        plt.plot(Vh[1,i], Vh[2,i], color=COLORS[themeid], marker='o')
    plt.show()

THEMES = ['Техника', 'Таврия', 'Биржа', 'Депутат', 'Газ']
COLORS = ['red',     'green',   'blue', 'yellow',  'cyan']
if __name__ == "__main__":
    #unfiltered_articles = grabber.grab(THEMES, 100)
    #grabber.save_articles("../Texts/", unfiltered_articles)
    '''unfiltered_articles = grabber.read_articles("../Texts/")
    BAD_SECTIONS = ['См. также', 'Примечания', 'Ссылки', 'Литература']
    filtered_articles = list(map(
        lambda tuple: (tuple[0], tuple[1], tuple[2], stem(grabber.removeSections(BAD_SECTIONS, tuple[3]))),
        unfiltered_articles
    ));
    grabber.save_articles("../TextsStemmed/", filtered_articles)'''
    stemmed_articles = grabber.read_articles("../TextsStemmed/")
    matrix, idx2word, idx2ArticleInfo = tf_idf(stemmed_articles)
    #np.savetxt("/home/user/Projects/R/LSA/matrix.csv", matrix, delimiter=",")
    # printMostRelevantWordsToArticles(matrix, idx2word, idx2ArticleInfo)
    # LSA part


    reversedIndex = getReversedIndex(matrix, idx2word)

    searchResult = searchByWords(searchString, reversedIndex, idx2ArticleInfo)
    searchByWords("футбол", reversedIndex, idx2ArticleInfo)
    print (searchResult)
    pass


