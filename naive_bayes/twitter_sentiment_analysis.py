import csv
import nltk
import random
import time

POSITIVE_KEYWORDS = ["good", "great", "awesome", "rock", "best", "love", "happy"]
NEGATIVE_KEYWORDS = [
    "bad", "hate", "suck", "fuck", "damn", "hell"
]
NEGATIVE_EMOJIS = [":(", ":'(", ":/", "=(", "='("]
POSITIVE_EMOJIS = [":)", "=)", ":-D", "<3", "&lt;3"]
LIMIT = 100000
counter = []


def get_training_and_validation_sets(rows):
    """
    Randomly shuffles the rows to ensure we're
    taking an unbiased sample, and then splits into
    a training set and a validation set of rows.
    """
    # randomly shuffle the rows
    random.shuffle(rows)

    # get the number of data points that we have
    count = len(rows)
    # 20% of the set, also called "corpus", should be training, as a rule of thumb, but not gospel.

    # we'll slice this list 20% the way through
    slicing_point = int(.20 * count)

    # the training set will be the first segment
    training_set = rows[:slicing_point]

    # the validation set will be the second segment
    validation_set = rows[slicing_point:]
    return training_set, validation_set


def get_most_common_words(rows):
    all_words = []
    for (tweet, sentiment) in rows:
        # Get all words longer than 3 letters from each individual tweet
        words = [i.lower() for i in tweet.split() if len(i) > 3]
        all_words.extend(words)

    # return the words that appear more than 100 times in the set of tweets
    return [
        word
        for word, count in nltk.FreqDist(all_words).most_common()
        if count > 100
    ]

def get_feature_function(word_list):
    """
    Given a list of words, returns a function
    that will output a feature dictionary
    that checks for certain features like tweet length
    as well as checking if the tweet contains the words
    in the word_list
    """
    def twitter_features(tweet):
        """
        Returns a dictionary of the features of the tweet we want our model
        to be based on, e.g. tweet_length.
        word_list is a list of all the words in all the tweets
        """
        features = {
            # The following features up to 'contains_many_caps' gets me to 60% accuracy
            "is_short": len(tweet) < 10,
            "contains_positive_words": any([keyword in tweet.lower() for keyword in POSITIVE_KEYWORDS]),
            "contains_negatve_words": any([keyword in tweet.lower() for keyword in NEGATIVE_KEYWORDS]),
            "contains_positive_emojis": any([emoji in tweet.lower() for emoji in POSITIVE_EMOJIS]),
            "contains_negative_emojis": any([emoji in tweet.lower() for emoji in NEGATIVE_EMOJIS]),
            "contains_question": "?" in tweet,
            "starts_with_I": tweet and tweet[0].upper() == "I",
            "contains_lol": "lol" in tweet.lower(),
            "contains_many_exclamations": "!!!" in tweet,
            "contains_ellipses": ".." in tweet,
            "contains_many_caps": len([word for word in tweet.split() if word == word.upper()]) > 3,
        }
        for word in word_list:
            features["contains_{}".format(word)] = word in tweet.lower()
        print len(counter)
        counter.append(1)
        return features
    return twitter_features

def get_feature_sets():
    rows = []
    with open('/home/vagrant/repos/datasets/clean_twitter_data.csv', 'rb') as f:
        for row in csv.reader(f):
            # The convention in nltk is to have the label be the second element of the tuple
            # So we're just switching them so we have (0, "I am sad") as the format
            rows.append((row[1], row[0]))

    # the 0th row is the header
    rows = rows[1:]
    word_list = get_most_common_words(rows[:LIMIT])
    training_set, validation_set = get_training_and_validation_sets(rows[:LIMIT])
    feature_function = get_feature_function(word_list)
    training_feature_set = nltk.classify.apply_features(feature_function, training_set)
    validation_feature_set = nltk.classify.apply_features(feature_function, validation_set)
    return training_feature_set, validation_feature_set

def run_classification(training_set, validation_set):
    print "training the training set"
    classifier = nltk.NaiveBayesClassifier.train(training_set)
    print "checking accuracy"
    accuracy = nltk.classify.accuracy(classifier, validation_set)
    print "The accuracy was.... {}".format(accuracy)
    return classifier

start_time = time.time()
our_training_set, our_validation_set = get_feature_sets()
classifier = run_classification(our_training_set, our_validation_set)
end_time = time.time()
completion_time = end_time - start_time
print "It took {} seconds to run the algorithm".format(completion_time)
classifier.show_most_informative_features(50)
