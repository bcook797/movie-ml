import csv
import time
import tmdbsimple as tmdb

tmdb.API_KEY = 'fbcbee8de7fe5d215b5f7a16969027d3'
movie_ids = []

with open('data/links.csv') as infile:
    reader = csv.reader(infile)
    reader.next() # skip headers ['movieId', 'imdbId', 'tmdbId']
    for row in reader:
        movie_ids.append(row[2])

for x in range(100):
    movie = tmdb.Movies(movie_ids[x])
    response = movie.info()
    print movie.title
    time.sleep(0.25) # Rate Limit for Tmdb is 40 requests every 10 seconds
