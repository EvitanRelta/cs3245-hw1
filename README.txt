This is the README file for A0235143N's submission
Email: e0727143@u.nus.edu

== Python Version ==

I'm using Python Version <3.10.12 or replace version number> for
this assignment.

== General Notes about this assignment ==

Give an overview of your program, describe the important algorithms/steps 
in your program, and discuss your experiments in general.  A few paragraphs 
are usually sufficient.


# Classes/Functions

1. `NGramLM` class is a custom class implementing a general N-gram model, where in
   this case n=4 (ie. 4-gram model). It contains the code for training the model
   on texts, computing the (log) probabilities of grams, add-one smoothing, and
   breaking text into grams.

2. `build_LM` function trains 3x 4-gram models on the training data, one for
   each label (ie. malaysian, indonesian and tamil), and then does add-one
   smoothing to each model based on all the grams seem by all models.

3. `test_LM` function tests the 3 trained 4-gram models against unlabelled text,
   writing the predicted labels into a file. If the percentage of unseen grams is
   `>= OTHER_PERCENT_UNSEEN_THRESHOLD` (defined in the function), that string is
   considered an alien language thus labelled "other".


# Design decisions

1. String-grams:
   Each 4-grams are represented as 4-character strings instead of 4-element
   tuples of characters, to avoid tuple-tuple equality comparisons which is kinda
   wack.

2. Log-probabilities:
   The multiplied probabilities of grams often gets so small that it gets rounded
   to 0. Thus, the log of those probabilities are used instead.

3. Dealing with "other" language label:
   Upon meeting a gram not seen during training, it's ignored (by having a
   log-probability of 0 in `NGramLM._get_gram_log_probability`).

   I assumed that if a lot of the grams in a text is unseen, it's likely an
   unseen language. Thus, if the percentage of unseen grams is
   `>= OTHER_PERCENT_UNSEEN_THRESHOLD` (defined in `test_LM`), that text is
   labelled "other".

   A threshold of 0.6 percent unseen grams is chosen as, in the test data, the
   lowest percent for an "other" text is 0.8 and highest percent for a non-other
   text is ~0.41, and 0.6 is in the middle of those 2.


== Files included with this submission ==

List the files in your submission here and provide a short 1 line
description of each file.  Make sure your submission's files are named
and formatted correctly.

- build_test_LM.py : my code.
- README.txt : overview of my code.

== Statement of individual work ==

Please put a "x" (without the double quotes) into the bracket of the appropriate statement.

[x] I, A0235143N, certify that I have followed the CS 3245 Information
Retrieval class guidelines for homework assignments.  In particular, I
expressly vow that I have followed the Facebook rule in discussing
with others in doing the assignment and did not take notes (digital or
printed) from the discussions.  

[ ] I, A0235143N, did not follow the class rules regarding homework
assignment, because of the following reason:

NIL

I suggest that I should be graded as follows:

NIL

== References ==

1. https://tedboy.github.io/nlps/generated/generated/nltk.FreqDist.html
   (for NLTK documentation)
