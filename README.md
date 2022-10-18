# SubGTR
This is the code for the manuscript "Temporal Knowledge Graph Question Answering via Subgraph Reasoning" (KBS).
Paper: https://www.sciencedirect.com/science/article/pii/S0950705122005603


![Architecture of SubGTR](https://s1.ax1x.com/2022/10/18/xrUN34.png)

## Installation

Clone and create a conda environment
``` 
git clone https://github.com/czy1999/SubGTR.git
```

The implementation is based on TempoQR in [TempoQR: Temporal Question Reasoning over Knowledge Graphs](https://arxiv.org/abs/2112.05785) and their code from https://github.com/cmavro/TempoQR. You can find more installation details there.
We use TComplEx KG Embeddings as implemented in https://github.com/facebookresearch/tkbc.


## Dataset and pretrained models download

Complex-CronQuestions can be found in ./ComplexCronQuestions folder.

For CronQueestions:
Download and unzip ``data.zip`` and ``models.zip`` in the root directory.

Drive: https://drive.google.com/drive/folders/1aS2s5sZ0qlDpGZ9rdR7HcHym23N3pUea?usp=sharing.

## Running the code


SubGTR on CronQuestions:
```
python ./train_qa_model.py --model subgtr --subgraph_reasoning --time_sensitivity --aware_module
 ```
SubGTR on Complex-CronQuestions (create the wikidata_big_complex folder in advance ):
```
python ./train_qa_model.py --model subgtr --dataset_name wikidata_big_complex --subgraph_reasoning --time_sensitivity --aware_module
 ```

Please explore more argument options in train_qa_model.py.

Note：Score Fusion module will be released soon. 

## Cite

If you find our method, code, or experimental setups useful, please cite our paper:
```
@article{DBLP:journals/kbs/ChenZLLK22,
  author    = {Ziyang Chen and
               Xiang Zhao and
               Jinzhi Liao and
               Xinyi Li and
               Evangelos Kanoulas},
  title     = {Temporal knowledge graph question answering via subgraph reasoning},
  journal   = {Knowl. Based Syst.},
  volume    = {251},
  pages     = {109134},
  year      = {2022},
}
```

