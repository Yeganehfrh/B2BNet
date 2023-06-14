# B2BNet
This project presents a deep neural network aimed at quantifying brain-to-brain coupling during the process of hypnosis induction. Our approach uses a multi-output sequence-to-sequence deep neural network applied to raw EEG data recorded from 51 participants using 59 electrodes. At its core, the model employs a one-dimensional convolutional neural network (CNN) and a long short-term memory (LSTM) encoder to embed the spatial-temporal dynamics inherent in the raw EEG signal into a lower-dimensional space.
This embedded representation is then utilized for two downstream heads: one head to predict the hypnotist's brain activity, and the other head to classify the level of hypnotic depth. A detailed schematic of the model architecture can be found in the subsequent figure.


![DL_architecture](https://github.com/Yeganehfrh/B2BNet/assets/36996819/5349f910-5a66-4d3f-b2ac-7668a56f68fa)
