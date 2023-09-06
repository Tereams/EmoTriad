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

