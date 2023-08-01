from datetime import datetime  
from matplotlib import pyplot as plt     
import pandas as pd  
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score


class MachineLearning():
    def __init__(self) -> None:
        print("Loading Dataset ...")
        
        self.flow_dataset = pd.read_csv('FlowStatsfile.csv')
        self.flow_dataset.iloc[:, 2] = self.flow_dataset.iloc[:, 2].str.replace('.', '')
        self.flow_dataset.iloc[:, 3] = self.flow_dataset.iloc[:, 3].str.replace('.', '')
        self.flow_dataset.iloc[:, 5] = self.flow_dataset.iloc[:, 5].str.replace('.', '')
        
    def flow_training(self):
        print("Flow Training ...")
        X_flow = self.flow_dataset.iloc[:, :-1].values 
        X_flow = X_flow.astype('float64')
        
        y_flow = self.flow_dataset.iloc[:, -1].values 
        
        X_flow_train, X_flow_test, y_flow_train, y_flow_test = train_test_split(X_flow, y_flow, test_size=0.25, random_state=0)
        
        classifier = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
        flow_model = classifier.fit(X_flow_train, y_flow_train)
        
        y_flow_pred = flow_model.predict(X_flow_test)
        
        print("-"*25)
        print("Confusion Matrix")
        cm = confusion_matrix(y_flow_test, y_flow_pred)
        print(cm)
        
        acc = accuracy_score(y_flow_test, y_flow_pred)
        print(f'Success accuracy = {acc*100:.2f} %')
        fail = 1.0 - acc 
        print(f'fail accuracy = {fail*100:.2f} %')
        print("-"*25)
        
        x = ['TP', 'FP', 'FN', 'TP']
        plt.title('Random Forest')
        plt.xlabel('Class predict')
        plt.ylabel('Number of flow')
        plt.tight_layout()
        plt.style.use('seaborn-darkgrid')
        y = [cm[0][0], cm[0][1], cm[1][0], cm[1][1]]
        plt.bar(x, y, color='#000000', label='RF')
        plt.legend()
        plt.show()
        
    
def main():
    start =datetime.now()
    ml = MachineLearning()
    ml.flow_training()
    
    end = datetime.now()
    print("Training Time: ", (end-start))
    
if __name__ == "__main__":
    main()
# end main