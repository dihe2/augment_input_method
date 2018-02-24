#!/usr/bin/python

import urllib, json

#Grabbing and parsing the JSON data
def GoogPlac(lat,lng,radius,types,key):
  #making the url
  AUTH_KEY = key
  LOCATION = str(lat) + "," + str(lng)
  RADIUS = radius
  TYPES = types
  if TYPES == None:
  	MyUrl = ('https://maps.googleapis.com/maps/api/place/nearbysearch/json'
           '?location=%s'
           '&radius=%s'
           '&sensor=false&key=%s') % (LOCATION, RADIUS, AUTH_KEY)
  else:
  	MyUrl = ('https://maps.googleapis.com/maps/api/place/nearbysearch/json'
           '?location=%s'
           '&radius=%s'
           '&types=%s'
           '&sensor=false&key=%s') % (LOCATION, RADIUS, TYPES, AUTH_KEY)
  #grabbing the JSON result
  response = urllib.urlopen(MyUrl)
  jsonRaw = response.read()
  jsonData = json.loads(jsonRaw)
  return jsonData

def GoogPlac_name(lat, lng, name, radius, key):
    #making the url
    AUTH_KEY = key
    LOCATION = str(lat) + "," + str(lng)
    RADIUS = radius
    NAME = name
    MyUrl = ('https://maps.googleapis.com/maps/api/place/nearbysearch/json'
           '?location=%s'
           '&radius=%s'
           '&keyword=%s'
           '&key=%s') % (LOCATION, RADIUS, NAME, AUTH_KEY)

    #grabbing the JSON result
    response = urllib.urlopen(MyUrl)
    jsonRaw = response.read()
    jsonData = json.loads(jsonRaw)

    return jsonData


