# CP423 Assignment 1

## Preprocessing Text

### Methodology

To accomplish the task of text preprocessing a lot of the code was insipired from the previous assignment so most of it was copy and paste.  

Getting and storing the positions of the terms was a brute force method aimed to try and not think to hard about it

### Steps Taken

- Step 1: turn all the terms to lowercase using the .casefold() function on the text
  - We use .casefold() over .lower() function as some terms are not being lowered cases
  - [Refer to this document to see an example](https://dylancastillo.co/nlp-snippets-clean-and-tokenize-text-with-python/)
- Step 2: Tokenize the terms using NLTK
- Step 3: In a dictionary save the positions of the terms in the documents
- Step 4: remove stop words from the tokenized text using stopwords from nltk
- Step 5: remove punctuations using sub and escape from re library using punctuations from the strings library
- Step 6: remove special characters using regex
- Step 7: remove empty spaces
- Step 8: remove single occuring characters
- Step 9: create a new dictionary only containing the newly tokenized and only valid tokens
- Step 10: return results of the text preprocessing

### TF-IDF Matrix Methodology
For the TF-IDF matrix, we simply applied the formulas in the assignment instructions to our TF-IDF matrix.
There are 5 different weighting schemes which were used to construct 5 different versions of the TF-IDF matrix.
A matrix of vocabulary_size * total_number_of_documents was initialized, each entry being initialized to 0.
For the binary scheme, the only possible values are 0 and 1. For this we simply looped through the docID postings list of each
term and set the entry to 1 for every docID in the posting list, since this implied the appearance of that term in that docID.
For raw count, we utilized the lengths of each docID positional index list and summed them together to get the raw count of each term,
updating the matrix accordingly.
Term frequency was similar to raw count, except we needed to store all the document lengths in a separate list so we could
compute the raw count / document length for each term and its corresponding docID.
Log normalization was similar to raw count, except a logarithmic function is applied to the raw count to lessen the impact of
common words in longer documents.
Doubble normalization was to provide the most optimal weighting to each term in the TF-IDF matrix. We computed the most frequently
occurring term in each document and used it in the double normalization formula provided in the assignment instructions.

### Results
We used to the query 'high school' to retrieve the top 5 rankings with each TF-IDF weighting scheme below:

Enter a phrase query: high school
['high', 'school']
Matched: {248, 191, 68, 37, 198, 39, 108, 79, 50, 52, 152, 59, 63, 31}
The phrase: "high school" appears in the following documents:
Number of documents retrieved: 14
zombies.txt
radar_ra.txt
cooldark.txt
bgcspoof.txt
running.txt
blabnove.txt
girlclub.txt
dskool.txt
bulironb.txt
bulmrx.txt
long1-3.txt
bulzork1.txt
cardcnt.txt
bagelman.txt

Number of documents retrieved: 5
([14, 29, 31, 37, 39], 5)
The top 5 relevant docs to 'high school' using binary scheme are:
abyss.txt
arctic.txt
bagelman.txt
bgcspoof.txt
blabnove.txt
Number of documents retrieved: 5

The top 5 relevant docs to 'high school' using raw count scheme are:
gulliver.txt
vgilante.txt
dskool.txt
darkness.txt
hound-b.txt
([117, 239, 79, 72, 136], 5)
Number of documents retrieved: 5

The top 5 relevant docs to 'high school' using term frequency scheme are:
bagelman.txt
bullove.txt
bulzork1.txt
dskool.txt
tree.txt
([31, 51, 59, 79, 232], 5)
Number of documents retrieved: 5

The top 5 relevant docs to 'high school' using log normalization scheme are:
gulliver.txt
dskool.txt
hound-b.txt
radar_ra.txt
arctic.txt
([117, 79, 136, 191, 29], 5)
most frequent term is 6117
Number of documents retrieved: 5

The top 5 relevant docs to 'high school' using double normalization scheme are:
bagelman.txt
bullove.txt
bulzork1.txt
dskool.txt
arctic.txt
([31, 51, 59, 79, 29], 5)

## Cosine Similarity

### Methodology: Consine Similarity

The methodology behind Cosine Similarity was that it just did an implementation on what the assignment prompt said to do which was use the equation: `A dot B / (norm(A) * norm(B))`

One problem that occured was due to the `tf_idf_matrix`. It had columns and rows in the opposite position as I desired so I had to use the `zip()` function to solve it. For context see below:

- The columns and rows were in the opposite position as I desired (rows were of size 249 (total document count in positional index) and columns were of size 37819 (total term count) and I needed vector with the size 37819 to work with the query vector which was of size 37819)
- So I used the `zip()` function to get the columns of the matrix to be able to calculate the cosine similarity
- There are obviously performance implications of the choosen method to solve the problem but due to the time available we weren't able to choose the desired solution which is refactoring the code

### Results

In this situation and testing we used the query "high school"

#### Binary

```text
Number of documents retrieved: 5
bullove.txt
bagelman.txt
tinsoldr.txt
abyss.txt
hareleph.txt
```

#### Raw count

```text
Number of documents retrieved: 5
bagelman.txt
bullove.txt
dskool.txt
arctic.txt
buldream.txt
```

#### Term Frequency

```text
Number of documents retrieved: 5
bagelman.txt
bullove.txt
dskool.txt
arctic.txt
buldream.txt
```

#### Log Normalization

```text
Number of documents retrieved: 5
bagelman.txt
bullove.txt
bulzork1.txt
arctic.txt
buldream.txt
```

#### Double Normalization

```text
Number of documents retrieved: 5
bullove.txt
bagelman.txt
tinsoldr.txt
abyss.txt
hareleph.txt
```
