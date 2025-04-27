# Team 6 content-moderation

[March 11 Progress Presentation Slides](March%2011%20Progress%20Pres.pdf)

## Run the Model

Run the   `n_use_model.ipynb`   notebook. You can modify the last cell with your test prompt.

## Dataset Info

The notebook expects the input dataset in the `/data` folder with the name `labeled_data.csv`

https://github.com/t-davidson/hate-speech-and-offensive-language/blob/master/data/labeled_data.csv - [License](LICENSE.txt)

The data is stored as a CSV. Each data file contains these columns:

* count = number of CrowdFlower users who coded each tweet (min is 3, sometimes more users coded a tweet when judgments were determined to be unreliable by CF).
* hate_speech = number of CF users who judged the tweet to be hate speech.
* offensive_language = number of CF users who judged the tweet to be offensive.
* neither = number of CF users who judged the tweet to be neither offensive nor non-offensive.
* class = class label for majority of CF users. 0 - hate speech 1 - offensive language 2 - neither
* tweet = the text of the tweet