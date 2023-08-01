from datetime import datetime 
from matplotlib import pyplot as plt   
import numpy as np   
import pandas as pd  
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler

class MachineLearning():
    def __init__(self) -> None:
        
        print("Loading Dataset ...")
        self.counter = 0 
        self.flow_dataset = pd.read_csv('FlowStatsfile.csv')
        
        self.flow_dataset.iloc[:, 2] = self.flow_dataset.iloc[:, 2].str.replace('.', '')
        self.flow_dataset.iloc[:, 3] = self.flow_dataset.iloc[:, 3].str.replace('.', '')
        self.flow_dataset.iloc[:, 5] = self.flow_dataset.iloc[:, 5].str.replace('.', '')
        
        self.X_flow = self.flow_dataset.iloc[:, :-1].values
        self.X_flow = self.X_flow.astype('float64')
        
        self.y_flow = self.flow_dataset.iloc[:, -1].values 
        
        self.X_flow_train, self.X_flow_test, self.y_flow_train, self.y_flow_test = train_test_split(self.X_flow, self.y_flow, test_size=0.25, random_state=0)
    
    
    def LR(self):
        print("-"*25)
        print("Logistic Regression ...")
        self.classifier = LogisticRegression(solver='liblinear', random_state=0)
        self.Confusion_matrix()
    
    def KNN(self):
        print("-"*25)
        print("K-nearest neighbors ...")
        self.classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
        self.Confusion_matrix()
        
    
    def SVM(self):
        print("-"*25)
        print("Support Vector Machine ...")
        
        self.classifier = SVC(kernel='rbf', random_state=0)
        self.Confusion_matrix()
    
    def NB(self):
        print("-"*25)
        print("Naive Bayes ...")
        
        self.classifier = GaussianNB()
        self.Confusion_matrix()
    
    def DT(self):
        print("-"*25)
        print("Decision Tree ...")
        self.classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
        self.Confusion_matrix()
    
    
    def RF(self): 
        print("-"*25)
        print("Random Forest ...")
        self.classifier = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
        self.Confusion_matrix()
    
    def GB(self):
        print("-"*25)
        print("Gradient Boosting ...")
        self.classifier = GradientBoostingClassifier(n_estimators=10, criterion='friedman_mse', random_state=0)
        self.Confusion_matrix()
        
    def NN(self):
        print("-"*25)
        print("Neural Network ...")
        self.classifier = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
        self.Confusion_matrix()
        
        
    def Confusion_matrix(self):
        self.counter += 1
        
        if self.classifier == MLPClassifier():
            
            
            scale = StandardScaler()
            self.X_flow_scale = scale.fit_transform(self.X_flow)
            self.y_flow_scale = scale.transform(self.y_flow)
            
            self.flow_model = self.classifier.fit(self.X_flow_scale, self.y_flow_scale)
            self.y_flow_pred = self.flow_model.predict(self.X_flow_test)
            
            print("-"*25)
            print('Confusion Matrix')
            cm = confusion_matrix(self.y_flow_test, self.y_flow_pred)
            print(cm)
            
            acc = accuracy_score(self.y_flow_test, self.y_flow_pred)
            print(f'success accuracy = {acc*100:.2f} %')
            
            fail = 1.0 - acc
            print(f'fail accuracy = {fail*100:.2f} %')
            print('-'*25)
            
            
            x = ['TP', 'FP', 'FN', 'TN']
            x_indexes = np.arange(len(x))
            width = 0.10
            plt.xticks(ticks=x_indexes, labels=x)
            plt.title('Result of Algorithms')
            plt.xlabel('Class predict')
            plt.ylabel('Number of flows')
            plt.tight_layout()
            plt.style.use('seaborn-darkgrid')
            y1 = [cm[0][0], cm[0][1], cm[1][0], cm[1][1]]
            plt.bar(x_indexes-2*width, y1, width=width, color='lightgreen', label='NN')
            plt.legend()
            
        else:
            self.flow_model = self.classifier.fit(self.X_flow, self.y_flow)
            self.y_flow_pred = self.flow_model.predict(self.X_flow_test)
            
            print("-"*25)
            print('Confusion Matrix')
            cm = confusion_matrix(self.y_flow_test, self.y_flow_pred)
            print(cm)
            
            acc = accuracy_score(self.y_flow_test, self.y_flow_pred)
            print(f'success accuracy = {acc*100:.2f} %')
            
            fail = 1.0 - acc
            print(f'fail accuracy = {fail*100:.2f} %')
            print('-'*25)
            
            
            x = ['TP', 'FP', 'FN', 'TN']
            x_indexes = np.arange(len(x))
            width = 0.10
            plt.xticks(ticks=x_indexes, labels=x)
            plt.title('Result of Algorithms')
            plt.xlabel('Class predict')
            plt.ylabel('Number of flows')
            plt.tight_layout()
            plt.style.use('seaborn-darkgrid')
            if self.counter == 1:
                y1 = [cm[0][0], cm[0][1], cm[1][0], cm[1][1]]
                plt.bar(x_indexes-2*width, y1, width=width, color='#1b7021', label='LR')
                plt.legend()
            elif self.counter == 2:
                y2 = [cm[0][0], cm[0][1], cm[1][0], cm[1][1]]
                plt.bar(x_indexes-width, y2, width=width, color='#e46e6e', label='KNN')
                plt.legend()
            elif self.counter == 3:
                y3 = [cm[0][0], cm[0][1], cm[1][0], cm[1][1]]
                plt.bar(x_indexes, y3, width=width, color='#0000ff', label='NB')
                plt.legend()
            elif self.counter == 4:
                y4 = [cm[0][0], cm[0][1], cm[1][0], cm[1][1]]
                plt.bar(x_indexes+width, y4, width=width, color='#e0d692', label='DT')
                plt.legend()
            elif self.counter == 5:
                y5 = [cm[0][0], cm[0][1], cm[1][0], cm[1][1]]
                plt.bar(x_indexes+2*width, y5, width=width, color='#000000', label='RF')
                plt.legend()
                
            elif self.counter == 6:
                y6 = [cm[0][0], cm[0][1], cm[1][0], cm[1][1]]
                plt.bar(x_indexes, y6, width=width, color='skyblue', label='GB')
                plt.legend()
                plt.show()


def main():
    start_script = datetime.now()
    
    ml = MachineLearning()
    
    start = datetime.now()
    ml.LR()
    end = datetime.now()
    print("Learning and Predicting Time: ", (end-start))
    
    start = datetime.now()
    ml.KNN()
    end = datetime.now()
    print("Learning and Predicting Time: ", (end-start))
    
    start = datetime.now()
    ml.NB()
    end = datetime.now()
    print("Learning and Predicting Time: ", (end-start))
    
    start = datetime.now()
    ml.DT()
    end = datetime.now()
    print("Learning and Predicting Time: ", (end-start))
    
    start = datetime.now()
    ml.RF()
    end = datetime.now()
    print("Learning and Predicting Time: ", (end-start))
    
    start = datetime.now()
    ml.GB()
    end = datetime.now()
    print("Learning and Predicting Time: ", (end-start))
    
    
    start = datetime.now()
    ml.NN()
    end = datetime.now()
    print("Learning and Predicting Time: ", (end-start))
    
    
    end_script = datetime.now()
    print("Script Time: ", (end_script-start_script))
    
if __name__ == "__main__":
    main()
# end main
    