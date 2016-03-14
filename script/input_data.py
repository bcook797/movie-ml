import csv
import time
import tmdbsimple as tmdb
import numpy as np

tmdb.API_KEY = 'fbcbee8de7fe5d215b5f7a16969027d3'
attributes = {}

def read_data():
    movie_ids = []
    with open('data/links.csv') as infile:
        reader = csv.reader(infile)
        reader.next() # skip headers ['movieId', 'imdbId', 'tmdbId']
        for row in reader:
            movie_ids.append(row[2])
    return movie_ids

def create_data_set(movie_ids, batch_size):
    now = time.time()
    labels = np.zeros((batch_size, 11))
    for x in range(batch_size):
        movie = tmdb.Movies(movie_ids[x])
        response = movie.info(append_to_response='credits')
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
    print movie.title
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

movies = read_data()
train_movies, train_labels = create_data_set(movies, 100)
print "Number of movies: " + str(len(train_movies))
print "Number of attributes: " + str(len(train_movies[0]))
# test_set = create_train(movies, 500)
# validation_set = create_train(movies, 500)


