from datetime import datetime  
from matplotlib import pyplot as plt     
import pandas as pd   
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
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
        
        classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
        flow_model = classifier.fit(X_flow_train, y_flow_train) 
        
        y_flow_pred = flow_model.predict(X_flow_test) 
        
        print("-"*25)
        
        print("Confusion Matrix")
        cm = confusion_matrix(y_flow_test, y_flow_pred)
        print(cm)
        
        acc = accuracy_score(y_flow_test, y_flow_pred)
        print(f"Success accuracy = {acc*100:.2f} %") 
        fail = 1.0 - acc 
        print(f"Fail accuracy = {fail*100:.2f} %")
        print("-"*25)
        
        bonin = 0
        ddos = 0 
        for i in y_flow:
            if i == 0:
                bonin += 1
            elif i == 1:
                ddos += 1
        
        print("benin = ", bonin)
        print("ddos = ", ddos)
        print("-"*25)
        
        plt.title("Dataset")
        plt.tight_layout()
        
        explode =  [0, 0.1]
        
        plt.pie([bonin, ddos], labels=['NORMAL', 'DDoS'], wedgeprops={'edgecolor':'black'},
                explode=explode, autopct="%1.2f%%")
        plt.show()
        
        icmp = 0
        tcp = 0 
        udp = 0 
        
        proto = self.flow_dataset.iloc[:, 7].values 
        proto = proto.astype('int')
        for i in proto:
            if i == 6:
                tcp += 1
            elif i == 17:
                udp += 1
            elif i == 1:
                icmp += 1
        
        print("tcp =", tcp)
        print("udp =", udp)
        print("icmp =", icmp)
        
        plt.title("Dataset")
        
        explode = [0, 0.1, 0.1]
        
        plt.pie([icmp, tcp, udp], labels=['ICMP', 'TCP', 'UDP'], wedgeprops={'edgecolor': 'black'},
                explode=explode, autopct="%1.2f%%")
        plt.show()
        
        icmp_normal = 0
        tcp_normal = 0
        udp_normal = 0
        icmp_ddos = 0
        tcp_ddos = 0
        udp_ddos = 0
        
        proto = self.flow_dataset.iloc[:, [7,-1]].values 
        proto = proto.astype('int')
        
        for i in proto:
            if i[0] == 6 and i[1] == 0:
                tcp_normal += 1
            elif i[0] == 6 and i[1] == 1:
                tcp_ddos += 1
            
            if i[0] == 17 and i[1] == 0:
                udp_normal += 1
            elif i[0] == 17 and i[1] == 1:
                udp_ddos += 1
            
            if i[0] == 1 and i[1] == 0:
                icmp_normal += 1
            elif i[0] == 1 and i[1] == 1:
                icmp_ddos += 1
            
        print("tcp_normal = ", tcp_normal)
        print("tcp_ddos = ", tcp_ddos)
        print("udp_normal = ", udp_normal)
        print("udp_ddos = ", udp_ddos)
        print("icmp_normal = ", icmp_normal)
        print("icmp_ddos = ", icmp_ddos)
        
        plt.title("Dataset")
        
        explode = [0, 0.1, 0.1, 0.1, 0.1, 0.1]
        
        plt.pie([icmp_normal, icmp_ddos, tcp_normal, tcp_ddos, udp_normal, udp_ddos],
                labels=['ICMP_Normal', 'ICMP_DDoS', 'TCP_Normal', 'TCP_DDoS', 'UDP_Normal', 'UDP_DDoS'],
                wedgeprops={'edgecolor':'black'}, explode=explode, autopct="%1.2f%%")
        plt.show()
        
        
            

def main():
    start = datetime.now()
    
    ml = MachineLearning()
    ml.flow_training()
    
    end = datetime.now() 
    print("Training Time: ", (end-start)) 
    
    
if __name__ == "__main__":
    main()
# end main