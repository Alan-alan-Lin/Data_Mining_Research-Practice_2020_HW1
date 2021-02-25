# -Data_Mining_Research-Practice_2020_HW1
HW1 - Decision Tree  
資料集 : Game of Thrones(Explore deaths and battles from this fantasy world)  
目標 : 利用第二份資料character-deaths.csv，將其中三個欄位Death Year, Book of Death, Death Chapter取其中一個欄位當做預測目標  
流程 :   
+1.利用pandas套件將資料讀取進來  
+2.把空值以0替代  
+3.Death Year, Book of Death, Death Chapter三者取一個，將有數值的轉成1  
+4.將Allegiances轉成dummy特徵(底下有幾種分類就會變成幾個特徵，值是0或1，本來的資料集就會再增加約20種特徵)  
+5.亂數拆成訓練集(75%)與測試集(25%)   
+6.使用scikit-learn的DecisionTreeClassifier進行預測  
+7.印出Confusion Matrix，並計算Precision, Recall, Accuracy   
+8.產出決策樹的圖(產生depth=3和depth=4的圖)  
