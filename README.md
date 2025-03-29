# PLMs-ATG：Identification of Autophagy Protein by Integrating Protein Language Model Embeddings and PSSM-based Features
## 1.Abstract
Autophagy critically regulates cellular development while maintaining pathophysio-logical homeostasis. Since the autophagic process is tightly regulated by the coordina-tion of autophagy-related proteins (ATGs), precise identification of these proteins is essential. Although current computational approaches have addressed experimental recognition's costly and time-consuming challenges, they still have room for im-provement since handcrafted features inadequately capture the intricate patterns and relationships hidden in sequences. In this study, we proposed PLM-ATG, a novel computational model that integrates support vector machine with the fusion of protein language model (PLM) embeddings and position-specific scoring matrix (PSSM)-based features for the ATGs identification. First, we extracted sequence-based features and PSSM-based features as the inputs of six classifiers to establish baseline models. Among these, the combination of the SVM classifier and the AADP-PSSM feature set achieved the best prediction accuracy. Second, two popular PLM embeddings, i.e., ESM-2 and ProtT5, were fused with the AADP-PSSM features to further improve the prediction of ATGs. Third, we selected the optimal feature subset from the combination of the ESM-2 embeddings and AADP-PSSM features to train the final SVM model. The proposed PLM-ATG achieved an accuracy of 99.5% and an MCC of 0.990, which are nearly 5% and 0.1 higher than those of the state-of-the-art model EnsembleDL-ATG, respectively.
## 2.Requirements
Before running, please make sure the following packages are installed in Python environment:
```
numpy==1.24.3
pandas==2.2.2
scikit_learn==1.5.1
torch==2.0.0
biopython==1.84
seaborn==0.13.2
```
For convenience, we strongly recommended users to install the Anaconda Python 3.9.19 (or above) in your local computer.
In additon, to facilitate environment configuration, we offer requirements.txt. So, you can choose to directly run the following command to quickly install the dependencies.
```
pip install requirements.txt
```
## 3.Running
Changing working dir to PLM-ATG, and then running the following command:
```
python predict.py test.fasta
```
test.fasta: the path of input file in fasta format
You will see some information output by the prediction process and the final prediction results in the terminal.
## 4.Other Notes
### About PLM embeddings extraction
We have provided PLM embeddings for the protein sequences in this dataset in the /data/pretrain/. However, if you want to try it yourself, or extract PLM embeddings for other protein sequence, you can find the corresponding scripts in the /feature_extraction/.
- get_ProtBert_feature.py: extract ProtBERT embedding.
- get_ProtT5_feature.py: extract ProtT5 embedding.
- get_esm2_feature.py: extract ESM-2 embedding.
---
! You need note:
- You must change the path of input and output files to your actual path.
- You must pay attention to the directory structure of the input and output files.
- It is best to use GPU acceleration. Of course, you can also use only CPU, but you will have to wait for a while.
### About models
You can find our trained baseline model and final model in the /models folder. 
If you want to train yourself or explore our work in depth, you can find all the model training source code in the /training folder. 
In addition, parameters of our model are provided in the /parameters folder.
## 5.Web Server
Webserver is available at: https://www.cciwyy.top
