# PLMs-ATG：Identification of autophagy protein by integrating protein language model embeddings and PSSM-based features using machine learing model
## 1.Abstract
Autophagy plays a crucial role in cell development and differentiation, as well as being associated with various human diseases and pathophysiological conditions. Since the autophagic process is tightly regulated by the synergistic actions of autophagy-related proteins (ATGs), rapid and precise identification of these proteins is essential. Recent years have witnessed the emergence of several promising computational approaches to address experimental recognition's costly and time-consuming challenges. However, they primarily rely on handcrafted features. In this study, we propose PLMs-ATG, a novel computational model that integrates the support vector machine (SVM) with a composition of protein language model (PLM) embeddings and position-specific scoring matrix (PSSM)-based features for ATGs identification. Firstly, we extracted sequence-based and PSSM-based features as inputs and selected six classifiers to establish 36 baseline models. Through this analysis, we identified the SVM classifier and the AADP-PSSM feature set. Next, we evaluated the performance of three PLM embeddings, i.e., ProtT5, ESM-2, and ProtBERT in the ATGs identification task. After finding the best feature combinations ESM-2+AADP-PSSM, we employed Shapley Additive Explanations (SHAP) to quantify the contribution of each feature for feature selection. Finally, by applying t-SNE to visualize the representations learned by the SVM classifier in ATGs detection, PLMs-ATG showed strong interpretability. On the same independent test set, PLMs-ATG achieves an accuracy of 99.50% and an MCC of 0.9900, which are nearly 5% and 0.1 higher than the state-of-the-art model EnsembleDL-ATG, respectively. 
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
