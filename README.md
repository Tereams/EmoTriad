# EmoSense
EmoSense: Advanced Emotion Recognition in Conversations

## Emotion Recognition Task Using the DailyDialog Public Dataset

### Explanation of the functions of the programs in the folder:

acc_graph1.py: Visualization function for accuracy.<br>
balance.py: Program for data balancing.<br>
full_version.py: Final version of the model.<br>
loss_graph.py: Visualization function for loss.<br>
main.py: Basic version of the model.<br>
without_conver.py: Model that only models the speaker.<br>
without_person.py: Model that only models the context.<br>
Specific details of the content will be provided later.<br>

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

# Model
This section introduces the models used in this experiment. The experiment is based on the BERT model, and two improvement approaches are proposed on top of it. The final model is a combination of these two improvement approaches.

## Baseline Model
The experiment utilizes BERT as the baseline model for the task of dialogue emotion classification.<br>
The approach of the baseline model is to split the dialogue into individual sentences and directly classify each sentence as follows:<br>
**a.** Firstly, each sentence is embedded using the embedding layer provided by BERT.<br>
**b.** Next, the embedded sentences are passed through the BERT model to extract information.<br>
**c.** The output of the CLS (classification) token from BERT is extracted as the comprehensive semantic representation of the entire sentence.<br>
**d.** The CLS representation is then fed into a fully connected layer, and the output of the fully connected layer is used for classification.<br>
The workflow diagram of the entire process is shown below:

![image](https://github.com/Tereams/EmoTriad/assets/106360504/35ca2bf3-7f31-483c-8920-eeb4f6f20322)

## Improvement Approach 1
In the baseline approach, it can be observed that the lack of consideration for contextual information within the dialogue, as each sentence is processed individually, leads to a simplistic approach that disrupts the inherent semantic connections among the sentences. Therefore, an improvement is needed to address this issue.<br>
In the first improvement approach, a BiLSTM (Bidirectional Long Short-Term Memory) is introduced to model the context. The specific steps are as follows:<br>
**a.** The first three steps of this approach are the same as the baseline approach.<br>
**b.** After obtaining the CLS outputs of each sentence in a dialogue, the CLS outputs of all the sentences in the dialogue are fed into a BiLSTM layer. This results in an equal number of outputs as the inputs.<br>
**c.** The outputs of the BiLSTM layer are then passed through a fully connected layer, and the output of the fully connected layer is used for classification.<br>
The workflow diagram of the entire process is as follows:

![image](https://github.com/Tereams/EmoTriad/assets/106360504/1c86533e-0372-448d-8117-13a8bfec8976)

## Improved Approach 2
In the foundational approach, there is another aspect that has been overlooked, which is the modeling of the speaker. The speaker's own emotions also have a certain coherence. For example, if a person is in a happy mood while saying a sentence, it is likely that their next sentence will also have a positive emotion. Therefore, it is necessary to model the speaker. The second improved approach mainly addresses this issue:<br>
The specific approach is as follows:<br>
a. The first three steps of this approach are the same as the foundational approach.<br>
b. After obtaining the output of the CLS label for each sentence in a dialogue, separate the sentences in the dialogue into two groups based on the two speakers. Then, feed the content spoken by each speaker into two separate BiLSTM models to obtain their respective outputs.<br>
c. Reassemble the outputs of the two BiLSTM layers in the original order. Feed the recombined result into a fully connected layer, and classify the output of the fully connected layer.<br>
The flowchart of the entire process is as follows:

![image](https://github.com/Tereams/EmoTriad/assets/106360504/09a154ef-31ef-4f68-929e-5c7e0e66d86e)

## Final Model

The final model I adopted is a combination of Improved Approach 1 and Improved Approach 2, which involves both contextual modeling and speaker modeling. The specific approach is as follows:<br>
**a.** The first three steps of this approach are the same as the foundational approach.<br>
**b.** The second part of this approach is the same as step b in Improved Approach 1.<br>
**c.** Take the output of step a and process it using the methods described in steps b and c of Improved Approach 2.<br>
**d.** Concatenate the results from steps b and c, and then feed them into a fully connected layer. The output of the fully connected layer is used for classification.<br>
The flowchart of the entire process is as follows:

![image](https://github.com/Tereams/EmoTriad/assets/106360504/18b4c41c-bfb5-4eee-a118-732b71d16132)

# Experiment

## Experimental Procedure
The experimental procedure for this study is as follows:<br>
**a.** Divide the processed dataset into training, validation, and test sets. The ratio of division is 6:2:2.<br>
**b.** Set the model parameters, including learning rate, batch size, and number of epochs.<br>
**c.** Select a model and train it using the training set.<br>
**d.** Fine-tune the hyperparameters using the validation set. <br>
**e.** Test the trained model on the test set and compare the results.<br>
