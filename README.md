# Vietnamese Punctuation Prediction Using Deep Neural Networks

Our paper can be found at https://link.springer.com/chapter/10.1007/978-3-030-38919-2_32.

In this paper, we have studied the punctuation prediction problem for the Vietnamese language. We collect two large-scale datasets and conduct extensive experiments with both traditional method (using CRF models) and a deep learning approach.

# Requirements
This code has been tested with 

python 3.6.8

tensorflow 1.13.1

# Dataset

In this work, we collect over 40,000 articles from the Vietnamese Newspapers and 86 Novels to build two datasets with a total of over 900000 sentences.

### Data Preprocessing
We label each word by its immediately following punctuation, where label O denotes a space. Example:
```
Biển tạo ra 1/2 lượng oxy con người hít thở, giúp lưu chuyển nhiệt quanh Trái Đất và hấp thụ một lượng lớn CO2.
(The ocean produces a half of the amount of oxygen that humans can breathe, and help to circulate heat around the Earth and absorb large amounts of CO2.)
```
This paragraph can be labeled as follows:
```
biển tạo ra 1/2 lượng oxy con người  hít  thở   giúp lưu chuyển nhiệt quanh trái đất và hấp thụ một lượng lớn co2
 O    O  O   O    O    O   O    O    O   Comma    O    O     O     O     O     O   O   O  O   O   O    O    O  Period 
```
# Resources
Vietnamese Newspapers (https://baomoi.com/)

Vietnamese Novels (https://gacsach.com/tac-gia/nguyen-nhat-anh.html)

Embeddings: Word2vec vectors for vietnamese (fasttext) ( You can download cc.vi.300.vec at https://fasttext.cc/docs/en/crawl-vectors.html)

All requests related to the dataset can sent to the corresponding author via email at ngtbinh@hcmus.edu.vn.

# Usage
All the configurations are put in the train_[model's name]_model.py with
 - train_BiLSTM_Attention_focal_loss_model.py: BiLSTM + Attention model with focal loss
 - train_BiLSTM_CRF_focal_loss_model.py: BiLSTM + CRF model with focal loss
 - train_BiLSTM_focal_loss_model.py: BiLSTM model with focal loss
 - train_BiLSTM_Attention_model.py: BiLSTM + Attention model without focal loss
 - train_BiLSTM_CRF_model.py: BiLSTM + CRF model without focal loss
 - train_BiLSTM_model.py: BiLSTM model without focal loss
 - model.py: model definition with focal loss
 - model_without_focal_loss.py: model definition without focal loss
 
 # References
 ```
@InProceedings{10.1007/978-3-030-38919-2_32,
author="Pham, Thuy
and Nguyen, Nhu
and Pham, Quang
and Cao, Han
and Nguyen, Binh",
editor="Chatzigeorgiou, Alexander
and Dondi, Riccardo
and Herodotou, Herodotos
and Kapoutsis, Christos
and Manolopoulos, Yannis
and Papadopoulos, George A.
and Sikora, Florian",
title="Vietnamese Punctuation Prediction Using Deep Neural Networks",
booktitle="SOFSEM 2020: Theory and Practice of Computer Science",
year="2020",
publisher="Springer International Publishing",
address="Cham",
pages="388--400",
abstract="Adding appropriate punctuation marks into text is an essential step in speech-to-text where such information is usually not available. While this has been extensively studied for English, there is no large-scale dataset and comprehensive study in the punctuation prediction problem for the Vietnamese language. In this paper, we collect two massive datasets and conduct a benchmark with both traditional methods and deep neural networks. We aim to publish both our data and all implementation codes to facilitate further research, not only in Vietnamese punctuation prediction but also in other related fields. Our project, including datasets and implementation details, is publicly available at https://github.com/BinhMisfit/vietnamese-punctuation-prediction.",
isbn="978-3-030-38919-2"
}
```
