import pandas as pd
import numpy as np
import string
from sklearn.model_selection import train_test_split
from dsbox.ml.neural_networks.processing import Text2Sequence
from nltk.stem.snowball import EnglishStemmer
from dsbox.ml.neural_networks.processing.text_classification import Text2Sequence
from dsbox.ml.neural_networks.keras_factory.text_models import CNN_LSTMFactory
from dsbox.ml.neural_networks.processing.workflow import TextNeuralNetPipeline
from dsbox.ml.neural_networks.keras_factory.text_models import LSTMFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier

data = pd.read_csv('data.csv', usecols=['medical_specialty', 'transcription'])
data.loc[data.medical_specialty == ' Cardiovascular / Pulmonary', "medical_specialty"] = 'Cardiovascular'
data.loc[data.medical_specialty == ' Neurosurgery', 'medical_specialty'] = 'Neurology'
data.loc[data.medical_specialty == ' Neurology', 'medical_specialty'] = 'Neurology'
data.loc[data.medical_specialty == ' Urology', 'medical_specialty'] = 'Urology'
data.loc[data.medical_specialty == ' Obstetrics / Gynecology', 'medical_specialty'] = 'Gynecology'
data.loc[data.medical_specialty == ' Gastroenterology', 'medical_specialty'] = 'Gastroenterology'
data.loc[data.medical_specialty == ' Nephrology', 'medical_specialty'] = 'Gastroenterology'
data.loc[data.medical_specialty == ' Orthopedic', 'medical_specialty'] = 'Orthopedic'
data = data[data.medical_specialty.isin(['Cardiovascular', 'Neurology', 'Urology', 'Gynecology', 'Gastroenterology', 'Orthopedic'])]

data = data[['transcription', 'medical_specialty']]

def remove_punct(text):
    for p in string.punctuation:
        text = text.replace(p, ' ')
    text = ' '.join(text.split())
    return text

data.rename(columns = {'transcription':'Report', 'medical_specialty':'speciality'}, inplace = True)
data = data.dropna()
X=data
X['Report'] = X['Report'].map(lambda x: remove_punct(x).lower())

X_train, X_test, y_train, y_test = train_test_split(X['Report'], X['speciality'], test_size=0.2, random_state=42)

y_train = pd.DataFrame(y_train)
y_test = pd.DataFrame(y_test)
y_train = y_train.replace({'Cardiovascular': 0, 'Neurology': 1, 'Urology': 2, 'Gynecology': 3, 'Gastroenterology': 4, 'Orthopedic': 5}).to_numpy()
y_test = y_test.replace({'Cardiovascular': 0, 'Neurology': 1, 'Urology': 2, 'Gynecology': 3, 'Gastroenterology': 4, 'Orthopedic': 5}).to_numpy()

y_train = np.squeeze(y_train)
y_test = np.squeeze(y_test)

# CNNLSTM
t2s=Text2Sequence(stemmer=EnglishStemmer())
model = TextNeuralNetPipeline(text2seq=t2s,
                              factory_class=CNN_LSTMFactory, 
                              num_labels=6)
cnn_lstm = model.fit(X_train, y_train, 
                     epochs=15,
                     batch_size=100, 
                     shuffle=True)

# LSTM
model = TextNeuralNetPipeline(text2seq=t2s,
                              factory_class=LSTMFactory, 
                              num_labels=6)
lstm = model.fit(X_train, y_train, 
                        epochs=15,
                        batch_size=100, 
                        shuffle=True)

vectorizer = TfidfVectorizer(analyzer='word', stop_words='english',ngram_range=(1,3), max_df=0.75,min_df=5, use_idf=True, smooth_idf=True,sublinear_tf=True, max_features=1000)
tfIdfMat  = vectorizer.fit_transform(data['Report'].tolist())
feature_names = sorted(vectorizer.get_feature_names())
del feature_names[0:35]

pca = PCA(n_components=0.45)
tfIdfMat_reduced = pca.fit_transform(tfIdfMat.toarray())

labels = data['speciality'].tolist()

X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(tfIdfMat_reduced, 
                                                                    labels, 
                                                                    stratify=labels,
                                                                    random_state=1)

# log reg
model = LogisticRegression(random_state=0)
log_reg = model.fit(X_train_new, y_train_new)

# svm 
model = svm.SVC()
support_vector_machine = model.fit(X_train_new, y_train_new)

# ran for
model = RandomForestClassifier(n_estimators=100)
ran_for = model.fit(X_train_new, y_train_new)

# ensemble
ensemble_model = VotingClassifier(estimators=[('log_reg', log_reg), ('support_vector_machine', support_vector_machine), ('ran_for', ran_for)],voting='hard')
ensemble = ensemble_model.fit(X_train_new, y_train_new)

class MedicalSpeciality:
    
    def pred_CNNLSTM(self,sample_text):
        value = cnn_lstm.predict([sample_text])[0]
        if value == 0:
            return "Cardiovascular"
        elif value == 1:
            return "Neurology"
        elif value == 2:
            return "Urology"
        elif value == 3:
            return "Gynecology"
        elif value == 4:
            return "Gastroenterology"
        return "Orthopedic"
    
    def pred_LSTM(self,sample_text):
        value = lstm.predict([sample_text])[0]
        if value == 0:
            return "Cardiovascular"
        elif value == 1:
            return "Neurology"
        elif value == 2:
            return "Urology"
        elif value == 3:
            return "Gynecology"
        elif value == 4:
            return "Gastroenterology"
        return "Orthopedic"
    
    def pred_ML_Ensemble(self,sample_text):
        vector = vectorizer.transform([sample_text])
        vector_pca = pca.transform(vector.toarray())
        return ensemble.predict(vector_pca)[0]

    def final_prediction(self,sample_text):
        CNNLSTM = self.pred_CNNLSTM(sample_text)
        LSTM = self.pred_LSTM(sample_text)
        ENSM = self.pred_ML_Ensemble(sample_text)
    
        if CNNLSTM == LSTM and CNNLSTM == ENSM:
            return CNNLSTM
        elif CNNLSTM == LSTM and CNNLSTM != ENSM:
            if CNNLSTM != "Orthopedic" and ENSM == "Orthopedic":
                return ENSM
            elif CNNLSTM != "Neurology" and ENSM == "Neurology":
                return ENSM
            else:
                return CNNLSTM
        return CNNLSTM
    
specialty_detector = MedicalSpeciality()

import pickle

filename = 'model.pkl'
pickle.dump(specialty_detector, open(filename, 'wb'))
