# PLMs-ATG：Identification of Autophagy Protein by Integrating Protein Language Model Embeddings and PSSM-based Features
## 1.Abstract
Autophagy critically regulates cellular development while maintaining pathophysio-logical homeostasis. Since the autophagic process is tightly regulated by the coordina-tion of autophagy-related proteins (ATGs), precise identification of these proteins is essential. Although current computational approaches have addressed experimental recognition's costly and time-consuming challenges, they remain constrained by handcrafted features that inadequately extract crucial features related to ATGs. In this study, we proposed PLM-ATG, a novel computational model that integrates support vector machine with the fusion of protein language model (PLM) embeddings and position-specific scoring matrix (PSSM)-based features for the ATGs identification. First, we extracted sequence-based features and PSSM-based features as the inputs of six classifiers to establish baseline models. Of these models, the combination of the SVM classifier and the AADP-PSSM feature set achieved the best prediction accuracy. Sec-ond, two popular PLM embeddings, i.e., ESM-2 and ProtT5, were fused with the AADP-PSSM features to further improve the prediction of ATGs. Third, we selected the optimal feature subset from the combination of the ESM-2 embeddings and AADP-PSSM features to train the final SVM model. The proposed PLM-ATG achieved an accuracy of 99.5% and an MCC of 0.990, which are nearly 5% and 0.1 higher than those of the state-of-the-art model EnsembleDL-ATG, respectively. 
## 2.Requirements
```
numpy==1.24.3
pandas==2.2.2
scikit_learn==1.5.1
torch==2.0.0
torchvision==0.15.0
biopython==1.84
seaborn==0.13.2
```
