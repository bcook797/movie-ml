import csv
import time
import tmdbsimple as tmdb
import numpy as np
import tensorflow as tf

tmdb.API_KEY = 'fbcbee8de7fe5d215b5f7a16969027d3'
attributes = {}

def read_data():
    movie_ids = []
    with open('data/links.csv') as infile:
        reader = csv.reader(infile)
        reader.next() # skip headers ['movieId', 'imdbId', 'tmdbId']
        for row in reader:
            if row[2] != "" and row[2] != None:
                movie_ids.append(row[2])
    return movie_ids

def create_data_set(movie_ids, batch_size):
    now = time.time()
    labels = np.zeros((batch_size, 11))
    for x in range(batch_size):
        movie = tmdb.Movies(movie_ids[x])
        try:
            response = movie.info(append_to_response='credits')
        except:
            print sys.exc_info()[0]
        create_movie_vector(movie, x)
        rating = int(round(movie.vote_average))
        labels[x, rating] = 1

        time.sleep(0.25) # Rate Limit for Tmdb is 40 requests every 10 seconds

    data = np.zeros((batch_size, len(attributes)))
    attr_index = 0
    for key, value in attributes.iteritems():
        for id in value:
            data[id][attr_index] = 1


    print "Time elapsed: " + str(round(time.time() - now, 2)) + " secs"
    return data, labels

def create_movie_vector(movie, index):
    print str(index) + ": " + movie.title + " - " + str(movie.id)
    add_production_companies(movie.production_companies, index)
    add_cast_and_crew(movie.credits, index)

def add_cast_and_crew(credits, index):
    for cast in credits['cast']:
        person_id = cast['id']
        if person_id not in attributes:
            attributes[person_id] = [index]
        else:
            attributes[person_id].append(index)
    for crew in credits['crew']:
        person_id = crew['id']
        if person_id not in attributes:
            attributes[person_id] = [index]
        else:
            attributes[person_id].append(index)

def add_production_companies(companies, index):
    for company in companies:
        company_name = company['name']
        if company_name not in attributes:
            attributes[company_name] = [index]
        else:
            attributes[company_name].append(index)

# Read data and create train and test data sets
movies = read_data()
movie_data, movie_labels = create_data_set(movies, 30000)
num_of_attrs = len(movie_data[0])
print "Number of movies: " + str(len(movie_data))
print "Number of attributes: " + str(num_of_attrs)

movie_train_data = movie_data[:25000]
movie_train_labels = movie_labels[:25000]

movie_test_data = movie_data[25000:]
movie_test_labels = movie_labels[25000:]

# Start Tensorflow Training
x = tf.placeholder(tf.float32, [None, num_of_attrs])
W = tf.Variable(tf.zeros([num_of_attrs, 11]))
b = tf.Variable(tf.zeros([11]))
y = tf.nn.softmax(tf.matmul(x, W) + b)

y_ = tf.placeholder(tf.float32, [None, 11])
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

sess.run(train_step, feed_dict={x: movie_train_data, y_: movie_train_labels})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(sess.run(accuracy, feed_dict={x: movie_test_data, y_: movie_test_labels}))
