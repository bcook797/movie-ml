import csv
import time
import tmdbsimple as tmdb

tmdb.API_KEY = 'fbcbee8de7fe5d215b5f7a16969027d3'
attributes = { 'budget' : 0 }

def read_data():
    movie_ids = []
    with open('data/links.csv') as infile:
        reader = csv.reader(infile)
        reader.next() # skip headers ['movieId', 'imdbId', 'tmdbId']
        for row in reader:
            movie_ids.append(row[2])
    return movie_ids

def create_train(movie_ids, batch_size):
    now = time.time()
    for x in range(batch_size):
        movie = tmdb.Movies(movie_ids[x])
        response = movie.info(append_to_response='credits')
        create_movie_vector(movie)

        time.sleep(0.25) # Rate Limit for Tmdb is 40 requests every 10 seconds

    print "Time elapsed: " + str(round(time.time() - now, 2)) + " secs"

def create_movie_vector(movie):
    print movie.title
    add_production_companies(movie.production_companies)

def add_production_companies(companies):
    for company in companies:
        company_id = company['id']
        if company_id not in attributes:
            attributes[company_id] = len(attributes)


movies = read_data()
train_set = create_train(movies, 50)
print len(attributes)
# test_set = create_train(movies, 500)
# validation_set = create_train(movies, 500)


