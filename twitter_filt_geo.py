#!/usr/bin/python
import sys
import tweepy

consumer_key="gdLsE3urZ6HqjE2RjiwaFBwag"
consumer_secret="rubc1WvJoYOnsBoUMXXV660MUOhw685uTFjqnYmrRWdqoq6Y48"
access_key="846813944186650627-CpBzbP1i8ag3pREHkd6YcGQeMHbaLOx"
access_secret="rX2ikxxtI4zplBp9kbEEpibWVKyCwKJmvuxTiKxFgeJKA"

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_key, access_secret)
api = tweepy.API(auth)


class CustomStreamListener(tweepy.StreamListener):
    def on_status(self, status):
        if 'manchester united' in status.text.lower():
            print status.text

    def on_error(self, status_code):
        print >> sys.stderr, 'Encountered error with status code:', status_code
        return True # Don't kill the stream

    def on_timeout(self):
        print >> sys.stderr, 'Timeout...'
        return True # Don't kill the stream

sapi = tweepy.streaming.Stream(auth, CustomStreamListener())    
sapi.filter(locations=[-122.75,36.8,-121.75,37.8,-74,40,-73,41])
