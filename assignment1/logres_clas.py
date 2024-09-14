from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import argparse

def read_data(path):
    """
    Takes as input a path to a conll-like file, with the label in the first 
    column, and the text in the second. It returns a list of all input texts
    and a separate list with all gold labels.
    """
    txts = []
    golds = []
    for line in open(path):
        tok = line.strip().split('\t')
        txts.append(tok[1])
        golds.append(tok[0])
    return txts, golds


def train(data_path):
    train_txts, train_golds = read_data(data_path)

    word_vectorizer = CountVectorizer(analyzer='word', ngram_range=(1,2))
    train_feats = word_vectorizer.fit_transform(train_txts)

    classifier = LogisticRegression(max_iter=500)
    classifier.fit(train_feats, train_golds)
    return word_vectorizer, classifier

def evaluate(dev_path, vectorizer, classifier):
    dev_txts, dev_golds = read_data(dev_path)
    dev_feats = vectorizer.transform(dev_txts)
    dev_preds = classifier.predict(dev_feats)

    cor = sum([pred==gold for pred, gold in zip(dev_preds, dev_golds)])
    total = len(dev_golds)
    print(dev_path, '{:.2f}'.format(100*cor/total))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--dev_data", type=str, required=True)
    args = parser.parse_args()

    vectorizer, classifier = train(args.train_data)
    evaluate(args.dev_data, vectorizer, classifier)
    
