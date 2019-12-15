import logging
import pymysql
import azure.functions as func
import numpy as np
from scipy import spatial
from nltk.stem import PorterStemmer
import re
import json
from nltk.corpus import stopwords
from gensim.test.utils import common_texts, get_tmpfile
import gensim

stopwords = set(stopwords.words('english'))
path = get_tmpfile("word2vec.model")
model = gensim.models.KeyedVectors.load("word2vec.model")


def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')
    dic = {}
    req_body = None
    try:
        req_body = req.get_json()
    except ValueError:
        pass
    db = connSQL()
    doc1 = req_body.get('doc1')
    doc2 = req_body.get('doc2')
    doc1Id = getNewId(db)
    if doc1Id ==-1:
        doc1Id=1
    storeDocument(doc1, doc1Id, db)
    doc2Id = getNewId(db)
    storeDocument(doc2, doc2Id, db)
    similarity = compareTwoDocuments(doc1, doc2)
    doc1InfoList = getDocumentInfo(doc1Id, db)
    doc2InfoList = getDocumentInfo(doc2Id, db)
    dic['similarity'] = similarity
    dic['doc1InfoList'] = json.dumps(doc1InfoList)
    dic['doc2InfoList'] = json.dumps(doc2InfoList)

    if similarity and doc1InfoList and doc2InfoList:
        return func.HttpResponse(json.dumps(dic))
    else:
        # headers = {
        #     "Access-Control-Allow-Origin": "*",
        #     "Access-Control-Allow-Methods": "Get, Post, Options"
        # }
        return func.HttpResponse(
            "Please pass a name on the query string or in the request body",
            status_code=400,
        )


def avg_feature_vector(sentence, model, num_features, index2word_set):
    words = sentence.split()
    feature_vec = np.zeros((num_features,), dtype='float32')
    n_words = 0
    for word in words:
        if word in index2word_set:
            n_words += 1
            feature_vec = np.add(feature_vec, model[word])
    if (n_words > 0):
        feature_vec = np.divide(feature_vec, n_words)
    return feature_vec


def compareTwoDocuments(doc1, doc2):
    model = gensim.models.KeyedVectors.load("word2vec.model")
    index2word_set = set(model.wv.index2word)
    s1 = doc1.replace('\n', '')
    s2 = doc2.replace('\n', '')
    s1_afv = avg_feature_vector(s1, model=model, num_features=300, index2word_set=index2word_set)
    s2_afv = avg_feature_vector(s2, model=model, num_features=300, index2word_set=index2word_set)
    sim = 1 - spatial.distance.cosine(s1_afv, s2_afv)
    return (str(sim))


def connSQL():
    db = pymysql.connect("localhost", "root", "root", "Calculator")
    return db


def isStop(word):
    if word not in stopwords:
        return 0
    else:
        return 1


def stemmWords(word):
    ps = PorterStemmer()
    return ps.stem(word)


def storeDocument(s, docID, db):
    readData(s, docID, db)


def readData(s,docID, db):
    results = dict()

    # remove all blank line under the text
    newLine = re.sub("[\s+\.\!\/_,$%^*(+\"\'\]+|\[+——！，。？、~@#￥%……&*()]+", " ", s)
    newLine = newLine.rstrip().split()
    results = readEachWords(newLine, results)
    insert(results, docID, db)


def readEachWords(newLine, results):
    for eachWord in newLine:
        if eachWord not in results:
            try:
                word2vec = json.dumps(model[eachWord].tolist())
            except KeyError as e:
                print(1)
                word2vec = json.dumps([0.0,0.0,0.0])

            if isStop(eachWord):
                if stemmWords(eachWord) != eachWord:
                    value = [1, "STOP", "STEMMED", word2vec]
                    results[
                        eachWord] = value  # if the word is stopword and stemmed word, dict = {word:[1,"STOP","STEMMED"]}
                else:
                    value = [1, "STOP", "NULL", word2vec]
                    results[eachWord] = value

            else:
                if stemmWords(eachWord) != eachWord:
                    value = [1, "NULL", "STEMMED", word2vec]
                    results[eachWord] = value
                else:
                    value = [1, "NULL", "NULL", word2vec]
                    results[eachWord] = value
        else:
            results[eachWord][0] += 1

    return results


def insert(insertList, docID, db):
    cursor = db.cursor()
    for eachWord in insertList:
        try:
            sql = "INSERT INTO wordsCalculator(document_id,word,stoptype,word2vec,no_of_occurences," \
                  "stemmed_word) VALUES (%s, %s, %s, %s, %s, %s) "
            value = (docID, eachWord, insertList[eachWord][1], insertList[eachWord][3], insertList[eachWord][0],
                     insertList[eachWord][2])
            cursor.execute(sql, value)
            db.commit()
        except:
            db.rollback()


def getNewId(db):
    cursor = db.cursor()
    try:
        sql = "SELECT max(document_id) from wordsCalculator"
        cursor.execute(sql)
        res = cursor.fetchall()

        cnt = res[0][0]
        if not cnt:
            return -1
        db.commit()
    except:
        db.rollback()
        return -1
    return cnt + 1


def getDocumentInfo(docId, db):
    cursor = db.cursor()
    lst = []

    sql = "SELECT * FROM wordsCalculator WHERE document_id=" + str(docId)
    cursor.execute(sql)
    res = cursor.fetchall()

    for row in res:
        document_id = row[0]
        word = row[1]
        stoptype = row[2]
        word2vec = row[3]
        no_of_occurences = row[4]
        stemmed_word = row[5]
        lst.append([document_id, word, stoptype,json.loads(word2vec), no_of_occurences, stemmed_word])

    print(lst)
    return lst
