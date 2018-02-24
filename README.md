# augment_input_method
This folder open source the work I have done for the CS 512 (Data Mining Principles) final project at the University of Illinois at Urbana-Champaign in 2017 Spring. 
The project focus on improving input methods' ability to predict the users' next word by leveraging Geo-temporal information.
The statistic model used for word prediction is a hybrid Deep Neural Network (DNN) with projection layers (fully connected layers) and bi-directional LSTM layers. The model also leverages word embedding. Using embedding results from GloVe (https://nlp.stanford.edu/projects/glove/).
The model is evaluated on Twitter data collected in 2 weeks from the Twitter API leveraging Tweepy (http://www.tweepy.org/). The geographic information of these tweets has been collected using Google Place API (https://developers.google.com/places/).
More details on the project can be found in the augmenting-input-method.pdf included with the repository.
