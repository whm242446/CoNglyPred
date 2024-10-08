# CoNglyPred
## Accurate prediction of N-glycosylation sites using ESM-2 and structural features with graph network and co-attention
Motivated by the progress of protein pre-trained language models (pLMs) and the breakthrough in protein structure prediction, we introduced a high-accuracy model called CoNglyPred. Having compared various pLMs, we opt for the large-scale pLM ESM-2 to extract sequence embeddings, thus mitigating certain limitations associated with manual feature extraction. Meanwhile, our approach employs a graph transformer network to process the 3D protein structures predicted by AlphaFold2. The final graph output and ESM-2 embedding are intricately integrated through a co-attention mechanism. Among a series of comprehensive experiments on the independent test dataset, CoNglyPred outperforms state-of-the-art models and demonstrates exceptional performance in case study. In addition, we are the first to report the uncertainty of N-linked glycosylation predictors using Expected Calibration Error (ECE) and Expected Uncertainty Calibration Error (UCE). 
![Graphical abstract](https://github.com/whm242446/CoNglyPred/assets/105725880/26ca05e2-6a03-4b78-bf2e-4d7cd48a3568)

## Model framework
![Figure 1](https://github.com/whm242446/CoNglyPred/assets/105725880/19563308-dc3b-4c01-9435-a8539cb203b9)

## System requirement
CoNglyPred is developed under Linux/Windows environment with:  
•	python == 3.8   
•	torch == 2.1.2  
•	pytorch-lightning == 2.1.3  
•	torch_geometric == 2.4.0  

You need to pay attention to the packages installed during the environment configuration:
```
pip install fair-esm
```
# Usage
## Data Preprocessing
First, we need to extract the features of proteins.  
1)plms features:
```
# input protein fasta file
run /data_preprocessing/ESM_representations.py
```
2)one-hot features;  
3)pssm features:
```
# using psi-blast tool; input protein fasta file
os.system('psiblast -query ".../A001.fasta" -db .../BLAST/db/swissprot -evalue 0.001 -num_iterations 3 -out_ascii_pssm .../A001.pssm')
```
4)physicochemical property features;
```
# AAindex; For example(FAUJ880113)
from aaindex import aaindex1
full_record = aaindex1['FAUJ880113']
```
5)dssp features.
```
# input protein PDB file
# using mkdssp version(4.4.0) https://github.com/PDB-REDO/dssp/releases/tag/v4.4.0
run /data_preprocessing/dssp.py
```
## Train&Test  
1)Feature-based construction of protein graph datasets using ProteinGraphDataset;  
2)Train CoNglyPred model:
```
run .../train.py
```
3)Test CoNglyPred model using the best.pth
```
run .../test.py
```
