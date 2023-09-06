# EmoSense
EmoSense: Advanced Emotion Recognition in Conversations

## Emotion Recognition Task Using the DailyDialog Public Dataset

### Explanation of the functions of the programs in the folder:

acc_graph1.py: Visualization function for accuracy.

balance.py: Program for data balancing.

full_version.py: Final version of the model.

loss_graph.py: Visualization function for loss.

main.py: Basic version of the model.

without_conver.py: Model that only models the speaker.

without_person.py: Model that only models the context.

Specific details of the content will be provided later.

# Dataset
The dataset used in this experiment is DailyDialogue, an English public dataset. The data in this dataset consists of English text in the form of dialogues. Sentences within the dialogues are separated by "eou" and each sentence is annotated with an emotion label. The emotion labels are separated by spaces and include seven categories: Neutral, Happiness/Joy, Surprise, Sadness, Anger, Disgust, and Fear. 

The statistical information for the various emotions in the dataset is as follows:

Emotion Categories |  Emotion Quantities
--- | --- 
Neutral | 85572
Happiness/Joy |	12885
Surprise | 1823
Sadness | 1150
Anger |1022
Disgust	| 353
Fear | 174

# Data Processing

## Data Cleaning

First, the data needs to be cleaned and classified into three categories: positive, neutral, and negative emotions. Since the DailyDialogue dataset contains multiple emotions, the emotions need to be merged. The processing approach is as follows:
I will merge Sadness, Anger, Disgust, and Fear into the negative emotion category. Happiness/Joy will be considered as the positive emotion category, while Neutral will remain unchanged.
As for the Surprise emotion, since it can be both positive and negative (such as surprise or fright), making it difficult to determine its polarity, it will be removed from the dataset for ease of processing. The final distribution of the three emotions is shown in the table below.

Emotion Categories |  Emotion Quantities
--- | --- 
Positive | 11207
Neutral	| 75065
Negative | 2189

## Data Balancing

As can be seen from the table above, there is a clear imbalance in the data that needs to be addressed.
However, we need to consider the characteristics of dialogue texts themselves. Unlike evaluative texts such as movie reviews, dialogues often contain many factual statements and many sentences with less obvious emotions. This results in a significantly higher proportion of neutral emotions compared to non-neutral emotions in the majority of dialogues.

This imbalance is caused by the characteristics of dialogue texts rather than sampling, and forcibly eliminating this imbalance is difficult and inappropriate. When augmenting data for dialogue texts, it needs to be done on a dialogue level rather than a sentence level. This inevitably leads to an increase in the number of neutral emotions while increasing non-neutral emotions. However, since non-neutral emotions are generally less prevalent in dialogues, data augmentation would result in a few dialogues being duplicated multiple times, increasing the risk of overfitting.

Therefore, I believe that we can only control the imbalance between non-neutral and neutral emotions. For example, we can remove dialogues consisting entirely of neutral emotions from the dataset. Additionally, within the non-neutral emotions, it is relatively easier to balance positive and negative emotions, and we can focus on achieving balance in this aspect.

Therefore, I have removed dialogues that are entirely composed of neutral emotions from the dataset and reduced the number of positive emotions within the non-neutral emotions. After balancing, the final distribution of the data is as follows:

Emotion Categories |  Emotion Quantities
--- | --- 
Positive | 2040
Neutral	| 11228
Negative | 2036
