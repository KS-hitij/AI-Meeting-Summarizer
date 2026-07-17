import redis
import os



REDIS_HOST = os.getenv('REDIS_HOST')
REDIS_PASSWORD = os.getenv('REDIS_PASSWORD')
REDIS_USERNAME= os.getenv('REDIS_USERNAME')
REDIS_PORT= os.getenv('REDIS_PORT')

r = redis.Redis(
    host=REDIS_HOST, 
    port=REDIS_PORT,
    decode_responses=True,
    username=REDIS_USERNAME,
    password=REDIS_PASSWORD
)
