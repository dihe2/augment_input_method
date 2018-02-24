#!/usr/bin/python
#source goo_token.sh before running this file

import os
import json
import goo_setup

def get_google_key():
    """Setup Twitter authentication.
    Return: tweepy.OAuthHandler object
    """
    try:
        consumer_key = os.environ['GOOGLE_API_KEY']
    except KeyError:
        sys.stderr.write("TWITTER_* environment variables not set\n")
        sys.exit(1)    
    return consumer_key


