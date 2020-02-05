# Vietnamese Punctuation Prediction Using Deep Neural Networks
(link paper)

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
