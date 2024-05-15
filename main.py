from string import punctuation
from re import sub, escape
from os import listdir
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from numpy import dot, argsort, zeros
from numpy.linalg import norm
import math

# Global variables and constants
PREPROCESSED_DATA_FOLDER_PATH = "preprocessed_data"
SUPPORTED_OPERATORS = ("OR", "AND", "AND NOT", "OR NOT")


# a list of all files (with file extension) in the preprocessed_data folder
preprocessed_data = listdir(PREPROCESSED_DATA_FOLDER_PATH)

# this code requires punkt and stopwords in nltk library 
# nltk.download('punkt')
# nltk.download('stopwords')

class PositionalIndex:
    """
        Variables:
            index (dictionary): The Positional Index
                in the form:
                {
                    term: string
                    [
                        document count: int,
                        [
                            hashmap: 
                            {
                                document id: int 
                                    [
                                        term count: int 
                                            [
                                                term position in document: int
                                            ]
                                    ]
                            }
                        ]
                    ]
                }

                ex. 
                {"hello" : 
                    [5, 
                        [
                            {3 : [3, [120, 125, 278]]}, 
                            {5 : [1, [28]]}, 
                            {10: [2, [132, 182]]}, 
                            {23: [3, [0, 12, 28]]}, 
                            {27: [1, [2]]}
                    ]
                }
    """
    def __init__(self):
        self.index = {}
        self.total_term_count = 0
        self.total_doc_count = 0
            
    def update_index(self, doc_id, term, position):
        """
        Update the index with a term its position pointing to a document id 

        Args:
            doc_id (str): Unique identifier for the document.
            term (str): A term to be updated.
            position (int): the position of where the term exists in the document.
        """
        # check if term exists in index
        if term in self.index:
            # increment total term frequency by 1
            self.index[term][0] += 1
            # check if document id already exists in the index
            if doc_id in self.index[term][1]:
                # Check if position already exists in the positions list
                if position not in self.index[term][1][doc_id]:
                    # Add position to positions list
                    self.index[term][1][doc_id].append(position)
            else:
                # Initialize positions list for the document
                self.index[term][1][doc_id] = position
        # if term doesn't exist in the positional index
        else:
            # initialize the list
            self.index[term] = []
            # initial total term frequency is set to 1
            self.index[term].append(1)
            # initialize the postings list to be initially empty
            self.index[term].append({})
            # add document id to the postings list 
            self.index[term][1][doc_id] = position
            # increment total term count
            self.total_term_count += 1
    
    def get_postings(self, term):
        """
        Get the postings (document IDs and positions) for a given term.

        Args:
            term (str): The term to look up.

        Returns:
            list: List of (doc_id, position) tuples.
        """
        return self.index[term]
    
    def process_query(self, query):
        """
        Parameter: query - a string of words to be queried (phrase query)
        Steps:
        1. Tokenize the query
        2. Retrieve the positional index lists of all the tokenized terms
        3. Find the term with the smallest posting list
        4. Take the first term and its first positional index (position)
        5. Check if position + 1 appears in the next term's positional index list. If it does, go to the next term and repeat.
        6. If the positional indices appear in a consecutive order for all terms, then add that docID to the list of matched docIDs
        7. If not, then go to the next position in of the first term and repeat steps 5-6
        8. Repeat the above steps until all positions of the first term have been examined
        """
        matched_docs = set() # The set to be returned (i.e. set of docIDs that contain the phrase query)

        # Step 1: Tokenize query
        tokenized_query = tokenize_phrase(query)
        print(tokenized_query)

        # Step 2: Retrieve positional index lists
        positional_lists = {} # Format: positional_lists[i][0] = doc frequency | positional_lists[i][1] = {docID: [positions]}
        for term in tokenized_query:
            positional_lists[term] = self.get_postings(term)

        # Step 3: Find the term with the smallest posting list. This is for efficiency purposes, since it can narrow down
        # our search size substantially
            
        smallest_list_term = tokenized_query[0]
        # First, find the smallest posting list. For this purpose, we can examine the document frequencies
        for i in range(1, len(tokenized_query)):
            term = tokenized_query[i]
            # if the doc freq of the current doc is less than the smallest_list doc freq, then update it
            if positional_lists[term][0] < positional_lists[smallest_list_term][0]:
                smallest_list_term = term
            
        # print("Term with the smallest list is {} with doc freq {}".format(smallest_list_term, positional_lists[smallest_list_term][0]))
        # Add the docIDs from this smallest posting list to a set
        doc_ids = list(positional_lists[smallest_list_term][1].keys())
        # print(doc_ids)
        matched_docs = set()
        
        # Step 4 and 5: Take first position of the first term, and check if position + 1 exists in the next term's posting list
        first_term = tokenized_query[0]
        for id in doc_ids: # for each docID in the smallest posting list
            # print("Current id:",id)
            # check if the docID is in the first posting list
            if id in positional_lists[first_term][1]:
                # for each position index in the posting list of docID of the first term...
                for position in positional_lists[first_term][1][id]:
                    phrase_exists = True
                    # print("position:",position)
                    for i in range(1, len(tokenized_query)): # check to see if position + 1 exists in the next query term
                        # print("token:",tokenized_query[i])
                        if id in positional_lists[tokenized_query[i]][1]:
                            if position + 1 in positional_lists[tokenized_query[i]][1][id]:
                                position += 1 # update position to see if the next query term appears adjacent to the current position
                            else:
                                phrase_exists = False
                                break
                        else:
                            phrase_exists = False
                            break
                    if phrase_exists == True:
                        matched_docs.add(id)
                        # print("updated matched docs:",matched_docs)
        
        print("Matched:", matched_docs)

        return matched_docs
    
    def get_terms_count(self):
        """
        Returns the term frequency for each word in the positional index vocabulary.
        It returns the raw count of the term in each docID in its posting list, as well as a simplified data structure
        of (term: total_raw_count) pairs
        Given the positional_index data structure, calculates the raw count of each term (term frequency)
        To do this, we follow these steps:
        1. For each term in the positional index, get the length of each doc ID's positional index list. The sum of these lengths is the term_frequency
        2. Store the raw count of each docID in a nested dictionary as a key-value pair {term: {docID: raw_count}} 
        """
        term_frequencies = {}
        term_raw_counts = {} # simplified version of term_frequencies, where term_frequencies stores the raw count for each docID
        # term_raw_counts is simply a dictionary that stores each term: totalFrequency
        for term in self.index:
            total_raw_count, docID_tc = self.get_term_count(term)
            term_frequencies[term] = docID_tc
            term_raw_counts[term] = total_raw_count

        return term_frequencies, term_raw_counts

    def get_term_count(self, term):
        postings = self.get_postings(term)
        docID_tc = {} # dictionary of key-value pairs of the docID: raw_count 
        term_frequency = 0
        for doc_id in postings[1]:
            raw_count = len(postings[1][doc_id])
            docID_tc[doc_id] = raw_count
            term_frequency += raw_count
        return term_frequency, docID_tc

    def get_idf(self, term):
        # Assignment instructions specifies the following formula: IDF(word) = log(total number of documents / (document frequency(word) + 1))
        total_docs = self.total_term_count

        # Next, get the document frequency of the term passed to this function
        doc_freq = self.get_postings(term)[0] # returns the doc frequency 

        # Then calculate the IDF for that term
        idf = round(math.log(total_docs/ (doc_freq + 1)), 2) # Get IDF and round to two decimal places
        return idf

    def tf_idf_matrix(self, scheme, doc_lengths):
        """
        Generates the TF-IDF matrix for the desired scheme.
        The scheme codes are as follows:
        scheme = 0 --> Binary
        scheme = 1 --> Raw count (doesn't account for length of a document)
        scheme = 2 --> Term frequency (divides raw count by length of each document)
        scheme = 3 --> Log Normalization: log(1 + term_frequency)
        scheme = 4 --> Double Normalization: 0.5+0.5*(f(t,d)/ max(f(t`,d))
        """
        if scheme == 0:
            matrix = self.tf_idf_binary()
        elif scheme == 1:
            matrix = self.tf_idf_raw_count()
        elif scheme == 2:
            matrix = self.tf_idf_term_freq(doc_lengths)
        elif scheme == 3:
            matrix = self.tf_idf_log_norm()
        elif scheme == 4:
            matrix = self.tf_idf_double_norm(doc_lengths)

        return matrix

    def tf_idf_binary(self): # binary scheme for generating the TF-IDF matrix
        # Only accounts for if a term appears in a document or not. Ignores frequency
        # Construct the matrix with dimensions (number of documents) x (vocabulary size).
        vocab_size = self.total_term_count # rows
        num_of_docs = self.total_doc_count # columns
        # print("Matrix: {} x {}".format(vocab_size, num_of_docs))
        matrix = [[0 for _ in range(num_of_docs)] for _ in range(vocab_size)] # Initialize matrix with dimensions: num_of_docs x vocab_size

        # Iterate through the vocabulary
        # For each docID in each term's posting list, set that entry equal to 1 in the matrix (since if the docID exists in the list,
        # then the term must appear in that doc
        terms_list = list(self.index.keys()) # We cannot index the term based on index position to faciliate building the matrix, so this line enables
        # indexing based on numerical values rather than the term itself

        for i in range(vocab_size):
            term = terms_list[i]
            for doc_id in self.index[term][1]:
                matrix[i][doc_id] = 1
                matrix[i][doc_id] *= self.get_idf(term) # comment out this line if you just want the term frequency matrix (no IDF weighting added)
        # print_matrix(matrix)
        return matrix
    
    def tf_idf_raw_count(self): # raw count scheme for generating the TF-IDF matrix
        # Only accounts for the total number of times a term appears in each document
        # Construct the matrix with dimensions (number of documents) x (vocabulary size).
        vocab_size = self.total_term_count # rows
        num_of_docs = self.total_doc_count # columns
        # print("Matrix: {} x {}".format(vocab_size, num_of_docs))
        matrix = [[0 for _ in range(num_of_docs)] for _ in range(vocab_size)] # Initialize matrix with dimensions: num_of_docs x vocab_size

        # Iterate through the vocabulary and utilize the get_terms_count() function to obtain the raw count of the term in each doc

        terms_list = list(self.index.keys()) # We cannot index the term based on index position to faciliate building the matrix, so this line enables
        # indexing based on numerical values rather than the term itself

        term_counts, _ = self.get_terms_count()

        for i in range(vocab_size):
            term = terms_list[i]
            for doc_id, count in term_counts[term].items():
                matrix[i][doc_id] = count
                matrix[i][doc_id] *= self.get_idf(term) # comment out this line if you just want the term frequency matrix (no IDF weighting added)

        return matrix
    
    def tf_idf_term_freq(self, doc_lengths): # term frequency scheme for generating the TF-IDF matrix
        # Takes the raw count of a term in each document and divides it by the document length
        # Construct the matrix with dimensions (number of documents) x (vocabulary size).
        vocab_size = self.total_term_count # rows
        num_of_docs = self.total_doc_count # columns
        matrix = [[0 for _ in range(num_of_docs)] for _ in range(vocab_size)] # Initialize matrix with dimensions: num_of_docs x vocab_size

        # Iterate through the vocabulary and utilize the get_terms_count() function to obtain the raw count of the term in each doc

        terms_list = list(self.index.keys()) # We cannot index the term based on index position to faciliate building the matrix, so this line enables
        # indexing based on numerical values rather than the term itself

        term_counts, _ = self.get_terms_count()

        for i in range(vocab_size):
            term = terms_list[i]
            for doc_id, count in term_counts[term].items():
                doc_length = doc_lengths[doc_id]
                # print("doc length:", doc_length)
                # print(round(count/doc_length, 1))
                matrix[i][doc_id] = round(count/doc_length, 8) # compute term frequency for that term, rounded to 8 decimal points (since the frequencies
                # for each term are typically extremely small)
                matrix[i][doc_id] *= self.get_idf(term) # comment out this line if you just want the term frequency matrix (no IDF weighting added)

        return matrix
    
    def tf_idf_log_norm(self): # log normalization scheme for generating the TF-IDF matrix
        # Takes the logarithm of 1 + the raw count of each term
        vocab_size = self.total_term_count # rows
        num_of_docs = self.total_doc_count # columns
        matrix = [[0 for _ in range(num_of_docs)] for _ in range(vocab_size)] # Initialize matrix with dimensions: num_of_docs x vocab_size

        # Iterate through the vocabulary and utilize the get_terms_count() function to obtain the raw count of the term in each doc

        terms_list = list(self.index.keys()) # We cannot index the term based on index position to faciliate building the matrix, so this line enables
        # indexing based on numerical values rather than the term itself

        term_counts, _ = self.get_terms_count()

        for i in range(vocab_size):
            term = terms_list[i]
            for doc_id, count in term_counts[term].items():
                matrix[i][doc_id] = round(math.log(1 + count), 4) # compute log normalization and round to 4 decimal points
                matrix[i][doc_id] *= self.get_idf(term) # comment out this line if you just want the term frequency matrix (no IDF weighting added)

        return matrix

    def tf_idf_double_norm(self, doc_lengths): # term frequency scheme for generating the TF-IDF matrix
        # Takes the raw count of a term in each document and divides it by the document length
        # Construct the matrix with dimensions (number of documents) x (vocabulary size).
        vocab_size = self.total_term_count # rows
        num_of_docs = self.total_doc_count # columns
        matrix = [[0 for _ in range(num_of_docs)] for _ in range(vocab_size)] # Initialize matrix with dimensions: num_of_docs x vocab_size

        # Iterate through the vocabulary and utilize the get_terms_count() function to obtain the raw count of the term in each doc

        terms_list = list(self.index.keys()) # We cannot index the term based on index position to faciliate building the matrix, so this line enables
        # indexing based on numerical values rather than the term itself

        term_counts, term_total_counts = self.get_terms_count()
        # For this normalization scheme, we need to find the most frequently occurring term (max(f(t`, d))
        # For this purpose we can make use of term_total_counts, which is a dictionary of term: totalCount key-value pairs
        most_freq_term = max(term_total_counts, key=term_total_counts.get) # returns the term (key) with the highest frequency (value)
        # print("most frequent term is", term_total_counts[most_freq_term])

        for i in range(vocab_size):
            term = terms_list[i]
            for doc_id, count in term_counts[term].items():
                doc_length = doc_lengths[doc_id]
                # print("doc length:", doc_length)
                # print(round(count/doc_length, 1))
                term_freq = round(count/doc_length, 8) # compute term frequency for that term, rounded to 8 decimal points (since the frequencies
                # for each term are typically extremely small)
                double_normalization = 0.5 + 0.5*(term_freq/term_total_counts[most_freq_term])
                matrix[i][doc_id] = double_normalization
                matrix[i][doc_id] *= self.get_idf(term) # comment out this line if you just want the term frequency matrix (no IDF weighting added)

        return matrix

    def score_query(self, scheme, doc_lengths, query, matrix):
        # Create the query vector with size = vocabulary size
        tokenized_query = tokenize_phrase(query)
        vocab_size = self.total_term_count
        matrix = self.tf_idf_matrix(scheme, doc_lengths)
        query_vec = [0 for _ in range(vocab_size)] # initialize the query vector

        terms_list = list(self.index.keys()) # We cannot index the term based on index position to faciliate building the matrix, so this line enables
        # indexing based on numerical values rather than the term itself
        
        # For each word in the tokenized query...
        # 1. check to see if it exists. If it doesn't exist, proceed to the next term
        # 2. If it exists, determine its TF-IDF score using the TF-IDF matrix
        # 3. Find the five highest scores which were computed in step 5 and their corresponding docs in the TF-IDF matrix. 
        # 4. Return the docs with the top 5 highest scores as the rank

        for term in tokenized_query:
            if self.index.get(term) is not None: # if the current query word exists in the positional index vocabulary
                term_index = terms_list.index(term)
                for doc_id in range(len(matrix[term_index])):
                    query_vec[doc_id] += matrix[term_index][doc_id] # Add TF-IDF score of the term to the corresponding document's score in the query vector
    
        # Rank all documents and sort in decreasing order (since higher scores are more relevant)
        top_docs = sorted(range(len(query_vec)), key=lambda i: query_vec[i], reverse=True)[:5]

        retrieved_docs_num = len(top_docs)
        print("Number of documents retrieved:", retrieved_docs_num)

        return top_docs, retrieved_docs_num

    def generate_query_vector(self, query):
        """
        generate a query vector 

        Args:
            query (string): a string that a user want to query for

        Returns:
            np.array: query vector
        """
        query_vector = zeros(self.total_term_count)
        tokenized_query = tokenize_phrase(query)
        terms_list = list(self.index.keys())

        for term in tokenized_query:
            if term in self.index:
                term_index = terms_list.index(term)
                query_vector[term_index] = 1

        return query_vector

    def calculate_cosine_similarity(self, query_vector, tf_idf_matrix):
        """
        Calculates cosine similarity scores between the query vector and a list of document vectors
        
        Args:
            tf_idf_matrix:List of vectors representing documents
        
        Returns:
            list of float: Top 5 documents using cosine similarity scores
        """
        similarity_score = []
        cols = list(zip(*tf_idf_matrix))
        
        # calculate the score and store in array
        for doc_index in range(self.total_doc_count):
            doc_vector = cols[doc_index]
            dot_product = dot(query_vector, doc_vector)
            norm_query  = norm(query_vector)
            norm_doc = norm(doc_vector)
            similarity = dot_product / (norm_query * norm_doc)
            similarity_score.append(similarity)

        # Sort and get top 5 relevant documents
        top_5_docs = argsort(similarity_score)[::-1][:5]

        return top_5_docs

def get_doc_length(doc_id, file_map):
    file_name = file_map[doc_id]
    with open(f"{PREPROCESSED_DATA_FOLDER_PATH}/{file_name}", "r", encoding="utf-8", errors='ignore') as file:
        text = file.read()
        tokenized_text = tokenize_phrase(text)
        # print(tokenized_text)
        return len(tokenized_text)

def get_doc_length_list(file_map):
    # Given the file map (list of doc IDs: file names), computes the doc length for each document and returns it in a matrix
    doc_lengths = [] # list is in the same order as file_map. So doc_lengths[0] is the length of docID 0.
    for id in file_map:
        doc_lengths.append(get_doc_length(id, file_map))
    return doc_lengths

def print_matrix(matrix): # helper function for printing 2D list
    for row in matrix:
        print(row)
    return

def preprocess_text(text):
    """
        Process the text to be lowered, tokenized and removed stopwords, punctuations, special characters, empty spaces and single occuring characters

        Args:
            query (string): a string that a user want to query for

        Returns:
            result (dictionary): a dictionary that contains the tokenized terms with its positions
    """
    text = text.casefold()
    # tokenize text
    tokenizer = TweetTokenizer()
    tokenized_text = tokenizer.tokenize(text)

    # save positions of every token in a dictionary
    token_positions = {}
    token_position = 1
    for token in tokenized_text:
        # If token not in token_positions, initialize list with current position
        if token not in token_positions:
            token_positions[token] = [token_position]
        else:
            # If token already in token_positions, append current position to its list
            token_positions[token].append(token_position)
        token_position += 1

    # stop_words is a set of stop words in the English language
    stop_words = set(stopwords.words("english"))
    
    # removed stop words from tokenized text
    removed_stop_words = [word for word in tokenized_text if word not in stop_words]

    # remove punctuations
    removed_punctuations = [sub(f"[{escape(punctuation)}]", "", token) for token in removed_stop_words]

    # removed special characters from tokenized text with no stop words and punctuation
    removed_special_characters = [sub('[^a-zA-Z]', ' ', token) for token in removed_punctuations]

    # removed empty spaces from tokenized text with no stop words and punctuation and special characters
    removed_empty_space = [word.strip() for word in removed_special_characters]

    # eliminating single character words
    removed_single_characters = [token for token in removed_empty_space if len(token) > 1]

    result = {}

    for token, positions in token_positions.items():
        if token in removed_single_characters:
            result[token] = positions

    return result

def tokenize_phrase(phrase):
    # Basically preprocesses the data the same way as preprocess_text(), but includes stop words
    tokenizer = TweetTokenizer()
    token_list = tokenizer.tokenize(phrase)

    # Remove special characters and convert to lowercase
    processed_tokens = [sub('[^a-zA-Z]', '', token).lower() for token in token_list]

    # Eliminate empty tokens if there is any
    processed_tokens = [token for token in processed_tokens if token]
    return processed_tokens

def print_doc_names(doc_ids, file_map):
    """
    Given an array of doc IDs and the file_map, prints the corresponding file name for each of the documents (helper function)
    file_map is a list of key-value pairs where each key is the doc Id and its value is the file name
    """
    print(f"Number of documents retrieved: {len(doc_ids)}")
    for id in doc_ids:
        print(file_map[id])

    return

def input_query():
    user_query = input("Enter a phrase query: ")
    return user_query


#region Script Entry Point

# initialize doc_id for the document, a dictionary to map document id to a filename
doc_id = 0
file_map = {}
# initialize the positional index
positional_index = PositionalIndex()

# loop through the list of files 
for file_name in preprocessed_data:
    # open file 
    with open(f"{PREPROCESSED_DATA_FOLDER_PATH}/{file_name}", "r", encoding="utf-8", errors='ignore') as file:
        text = file.read()
        terms = preprocess_text(text)
        
        # update the positional index with document id, terms and postions
        for term, position in terms.items():
            positional_index.update_index(doc_id, term, position)

        # map the document id to the file name
        file_map[doc_id] = file_name
        doc_id += 1
        # increment total document size in the positional index
        positional_index.total_doc_count += 1


# Output starts here
query = input_query()
docs_length = get_doc_length_list(file_map)
terms_list = list(positional_index.index.keys())

# query = "high school"
matched_docs = positional_index.process_query(query)
print(f"The phrase: \"{query}\" appears in the following documents:")
print_doc_names(matched_docs, file_map)

# Generate TF-IDF matrices for all schemes
matrix_binary = positional_index.tf_idf_matrix(0, docs_length) # Pros: Easy and simple to generate, less computation heavy. Cons: Less accurate results
matrix_rc = positional_index.tf_idf_matrix(1, docs_length) # Pros: Easy to generate, more accurate than binary. Cons: Doesn't account for doc length
matrix_tf = positional_index.tf_idf_matrix(2, docs_length) # Pros: More accurate in determining a word's significance to the entire doc
# Cons: Slightly more computation heavy, need to store the lengths of all the documents beforehand which requires more space

matrix_lognorm = positional_index.tf_idf_matrix(3, docs_length) # Pros: Generally accurate and doesn't need to store all document lengths
# Cons: More computation heavy than raw count scheme and doesn't account as much for doc length since it doesn't store doc lengths

matrix_doublenorm = positional_index.tf_idf_matrix(4, docs_length) # Pros: Ensures less relevant words don't over-dominate more relevant ones
# if they are frequent, accounts for doc length while still maintaining the integrity of relevant term significance
# Cons: Computation heavy, and more steps are involved. Requires computation of the most frequently occurring term each the document

# Test all five schemes for TF-IDF weighting

relevant_docs = positional_index.score_query(0, docs_length, query, matrix_binary)
print(relevant_docs)

print("The top 5 relevant docs to '{}' using {} scheme are:".format(query, "binary"))
for doc_id in relevant_docs[0]:
    print(file_map[doc_id])
# print(relevant_docs)

relevant_docs = positional_index.score_query(1, docs_length, query, matrix_rc)

print("\nThe top 5 relevant docs to '{}' using {} scheme are:".format(query, "raw count"))
for doc_id in relevant_docs[0]:
    print(file_map[doc_id])
print(relevant_docs)

relevant_docs = positional_index.score_query(2, docs_length, query, matrix_tf)

print("\nThe top 5 relevant docs to '{}' using {} scheme are:".format(query, "term frequency"))
for doc_id in relevant_docs[0]:
    print(file_map[doc_id])
print(relevant_docs)

relevant_docs = positional_index.score_query(3, docs_length, query, matrix_lognorm)

print("\nThe top 5 relevant docs to '{}' using {} scheme are:".format(query, "log normalization"))
for doc_id in relevant_docs[0]:
    print(file_map[doc_id])
print(relevant_docs)

relevant_docs = positional_index.score_query(4, docs_length, query, matrix_doublenorm)

print("\nThe top 5 relevant docs to '{}' using {} scheme are:".format(query, "double normalization"))
for doc_id in relevant_docs[0]:
    print(file_map[doc_id])
print(relevant_docs)

#endregion

#region Testing Cosine Similarity

query_vector = positional_index.generate_query_vector(query)
result = positional_index.calculate_cosine_similarity(query_vector, matrix_doublenorm)
print(f"resulting cosine similarity calulation: ")
print_doc_names(result, file_map)








#endregion

#endregion Script Exit Point   


#region Testing Misc Functions

# print("high:",positional_index.get_postings("high"))
# print("school:",positional_index.get_postings("school"))

# query = "high school"
# print(file_map)
# print("high:",positional_index.get_postings("high"))
# print("school:",positional_index.get_postings("school"))

# docs_list = list(file_map.values())
# doc_id = docs_list.index("poem-1.txt")
# doc_len = get_doc_length(doc_id, file_map)
# print("Length of poem-1:", doc_len)

# doc_id = docs_list.index("oxfrog.txt")
# doc_len = get_doc_length(doc_id, file_map)
# print("Length of oxfrog:", doc_len)

# print(file_map)

# print(docs_length)

# print(positional_index.get_postings("fuck"))
# term_count, docID_tc = positional_index.get_term_count("fuck")
# print("term count:",term_count)
# print(docID_tc)

# idf = positional_index.get_idf("fuck")
# print("IDF({}): {}".format("fuck",idf))

#endregion

#region Testing binary scheme

# matrix = positional_index.tf_idf_matrix(0, docs_length)

# print(terms_list)
# i = terms_list.index('fuck')
# print("index:", i)
# print(matrix[i])
# print(matrix[i][31])
# print(matrix[i][68])
# print(matrix[i][69])
# print(matrix[i][77])

#endregion

#region Testing raw count scheme

# matrix = positional_index.tf_idf_matrix(1, docs_length)

# # print(terms_list)
# i = terms_list.index('fuck')
# print("index:",i)
# print(matrix[i])
# print(matrix[i][31])
# print(matrix[i][68])
# print(matrix[i][69])
# print(matrix[i][77])

#endregion

#region Testing term frequency scheme

# matrix = positional_index.tf_idf_matrix(2, docs_length)

# # print(terms_list)
# i = terms_list.index('fuck')
# print("index:",i)
# print(matrix[i])
# print(matrix[i][31])
# print(matrix[i][68])
# print(matrix[i][69])
# print(matrix[i][77])

#endregion

#region Testing log normalization scheme



# # print(terms_list)
# i = terms_list.index('fuck')
# print("index:", i)
# print(matrix[i])
# print(matrix[i][31])
# print(matrix[i][68])
# print(matrix[i][69])
# print(matrix[i][77])

#endregion

#region Testing double normalization scheme

# matrix = positional_index.tf_idf_matrix(4, docs_length)

# print(terms_list)
# i = terms_list.index('fuck')
# print("index:", i)
# print(matrix[i])
# print(matrix[i][31])
# print(matrix[i][68])
# print(matrix[i][69])
# print(matrix[i][77])

#endregion

#region Testing get all the terms count

# term_frequencies, term_raw_counts = positional_index.get_terms_count()
# print(f"term frequency raw: {len(term_frequencies)}")

#endregion

#region Testing get the term count of the word "fuck"

# term_frequency, term_count_raw = positional_index.get_term_count("fuck")
# print(f"term term_frequency: {term_frequency}")
# print(f"term frequency raw: {term_count_raw}")

#endregion

#region Testing process query

# query = "high school"
# matched_docs = positional_index.process_query(query)
# print(f"The phrase: \"{query}\" appears in the following documents:")
# print_doc_names(matched_docs, file_map)

#endregion

#region Test score query

#you can choose any weighting scheme and the results will be slightly different. Refer to the tf_idf_matrix() function for the weighting scheme codes.
# rank documents based on weighting scheme 4 (double normalization)
# query = "high school"
# relevant_docs = positional_index.score_query(4, docs_length, query) 
# print("The top 5 relevant docs to '{query}' are:")
# for doc_id in relevant_docs:
#     print(file_map[doc_id])
# print(relevant_docs)



