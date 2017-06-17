import csv
import time
import sys
import math
import tmdbsimple as tmdb
import numpy as np
import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL']='2' # Look into building tensorflow from source to improve speed
tmdb.API_KEY = 'fbcbee8de7fe5d215b5f7a16969027d3'
attributes = {}

def read_data():
    movie_ids = []
    with open('data/links.csv') as infile:
        reader = csv.reader(infile)
        next(reader) # skip headers ['movieId', 'imdbId', 'tmdbId']
        for row in reader:
            if row[2] != "" and row[2] != None:
                movie_ids.append(row[2])
    np.random.shuffle(movie_ids)
    return movie_ids

def create_data_set(movie_ids, batch_size):
    now = time.time()
    labels = np.zeros((batch_size, 11))
    for x in range(batch_size):
        try:
            movie = tmdb.Movies(movie_ids[x])
            response = movie.info(append_to_response='credits')
            create_movie_vector(movie, x)
            rating = int(round(movie.vote_average))
            labels[x, rating] = 1

            time.sleep(0.25) # Rate Limit for Tmdb is 40 requests every 10 seconds
        except Exception as e:
            print(e)

    data = np.zeros((batch_size, len(attributes)))
    attr_index = 0
    for key, value in iter(attributes.items()):
        for id in value:
            data[id][attr_index] = 1


    print("Time elapsed: " + str(round(time.time() - now, 2)) + " secs")
    return data, labels

def create_movie_vector(movie, index):
    print(str(index) + ": " + movie.title + " - " + str(movie.id))
    add_production_companies(movie.production_companies, index)
    add_cast_and_crew(movie.credits, index)
    add_genres(movie.genres, index)

def add_genres(genres, index):
    for genre in genres:
        genre_key = genre['name'] + str(genre['id'])
        add_attribute(genre_key, index)

def add_cast_and_crew(credits, index):
    for cast in credits['cast']:
        person_id = cast['id']
        add_attribute(person_id, index)
    for crew in credits['crew']:
        person_id = crew['id']
        add_attribute(person_id, index)

def add_production_companies(companies, index):
    for company in companies:
        company_name = company['name']
        add_attribute(company_name, index)

def add_attribute(key, index):
    if key not in attributes:
        attributes[key] = [index]
    else:
        attributes[key].append(index)

# Read data and create train and test data sets
movies = read_data()
batch_size = int(sys.argv[1])
set_cutoff = int(batch_size * .75)
movie_data, movie_labels = create_data_set(movies, batch_size)
num_of_attrs = len(movie_data[0])
print("-------Model Constructed-------")
print("Number of movies: " + str(len(movie_data)))
print("Number of attributes: " + str(num_of_attrs))

movie_train_data = movie_data[:set_cutoff]
movie_train_labels = movie_labels[:set_cutoff]

movie_test_data = movie_data[set_cutoff:]
movie_test_labels = movie_labels[set_cutoff:]

print("------Train Model--------")
x = tf.placeholder(tf.float32, [None, num_of_attrs])
W = tf.Variable(tf.zeros([num_of_attrs, 11]))
b = tf.Variable(tf.zeros([11]))
y = tf.nn.softmax(tf.matmul(x, W) + b)

y_ = tf.placeholder(tf.float32, [None, 11])
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Crazy dynamic determination of the number of training steps involved
train_number = int(math.ceil(batch_size / 100.0))
start = 0
for i in range(train_number):
    end = i * 100
    train_batch = movie_train_data[start:end]
    label_batch = movie_train_labels[start:end]
    if i%100 == 0 and i > 0:
        train_accuracy = sess.run(accuracy, feed_dict={ x: train_batch, y_: label_batch})
        print("step %d, training accuracy %g"%(i, train_accuracy))
    sess.run(train_step, feed_dict={x: train_batch, y_: label_batch})
    start = end


print("-------Prediction Accuracy--------")
print(sess.run(accuracy, feed_dict={x: movie_test_data, y_: movie_test_labels}))
