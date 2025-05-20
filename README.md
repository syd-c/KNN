# Accelerated Discovery of Multifunctional K0.5Na0.5NbO3-Based Ceramics via Integrated High-Throughput Computation and Machine Learning
Potassium sodium niobate (KNN)-based ceramics have emerged as a cutting-edge intelligent material due to their remarkable multifunctionality, exhibiting both exceptional piezoelectric properties for medical transducers and precision sensors, along with unique photochromic-photoluminescent capabilities for optical windows and anti-counterfeiting technologies. Recent breakthroughs in artificial intelligence have facilitated machine learning applications in composition optimization, property prediction, and microstructure design of KNN ceramics, significantly accelerating material development and enabling their deployment in emerging fields like flexible electronics and energy harvesting. In this work, a comprehensive KNN database containing 300 data points is first constructed using high-throughput density function theory calculations, and the reliability of the database is verified through experimental methods. In addition, a critical appraisal is performed for employing different machine learning models to predict the structural stability and electronic structure of KNN-based ceramics based on material composition and crystal structure. The results show that the models based on the random forest algorithm can accurately predict the structural stability and electronic structure of KNN-based ceramics, exhibiting strong generalization ability. Furthermore, the physically meaningful descriptors are extracted to guide the design of KNN-based ceramics with high performance by using feature importance analysis of RF models. This work has accelerated the development of KNN-based ceramics with electro-optical multifunctional coupling

![image](https://github.com/user-attachments/assets/ce2bd95b-77e0-40e8-af6d-c070852177e8)
#### 1. Screening the models
```bash
python Screening_model.py
```
#### 2. Training
```bash
models.py
```
#### 3. Screening key features
```bash
python RFE.py
```
## Requirements
```bash
pip install pandas
pip install matplotlib
pip install numpy
pip install scikit-learn
pip install pyecharts
pip install xgboost
