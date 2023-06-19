'''
This is the function of the URL malware detection of this program with streamlit
'''
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import dill as pickle


class URL_detector:
    '''
    A class to represent an URL detector 
    ...
    Attributes
    ----------
    whitelist: list
        containts the list of URLs that seems to be malicious but actually a benign one

    lgt: LogisticRegression model

    vectorizer: TfidfVectorizer

    '''

    def __init__(self):
        # Using whitelist filter as the model fails in some legit cases
        file = 'Classifier/URL_Detector/pickel_URL_whitelist.pkl'
        with open(file, 'rb') as f:
            whitelist = pickle.load(f)
        f.close()

        # Loading the Linear Regression model
        file1 = "Classifier/URL_Detector/pickel_model.pkl"
        with open(file1, 'rb') as f1:
            lgr = pickle.load(f1)
        f1.close()

        # Loading the vectorizer
        file2 = "Classifier/URL_Detector/pickel_vector.pkl"
        with open(file2, 'rb') as f2:
            vectorizer = pickle.load(f2)
        f2.close()

        self.whitelist: list = whitelist
        self.lgr: LogisticRegression = lgr
        self.vectorizer: TfidfVectorizer = vectorizer

    def scan_url(self, url: str) -> str:
        '''
        Return a result if the give url is malicious or benign

        Parameters
        ----------
        url: string

        Returns
        -------
        'good' or 'bad'
        '''
        s_url = [url if url not in self.whitelist else '']
        # Transform url to Tf-idf-weighted document-term matrix
        x = self.vectorizer.transform(s_url)
        # Predicting url
        y_predict = self.lgr.predict(x)
        # Print the result
        return (y_predict[0])
