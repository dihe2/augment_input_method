#!/usr/bin/python

STAR_IND = 0

PLACE_MAX_QUERY_COUNT = 10
PLACE_MIN_QUERY_COUNT = 1
PLACE_RAD = 50

COOR_MAX_QUERY_COUNT = 50
COOR_MIN_QUERY_COUNT = 10
COOR_RAD = 150

twe_file = '/home/hd89cgm/cs512/fp/data/filtered_geo_ext_full.pickle'
twe_goo_place_file = '/home/hd89cgm/cs512/fp/data/geo_place_full.pickle'
twe_goo_coor_file = '/home/hd89cgm/cs512/fp/data/geo_coor_2.pickle'


import pickle
import json
import os
import sys
sys.path.insert(1, '/home/hd89cgm/cs512/fp/')
import goo_setup

goo_key = os.environ['GOOGLE_API_KEY']
'''
########################################
with open(twe_goo_place_file, 'r') as fid:
    twi_with_place, err_ind = pickle.load(fid)

STAR_IND = err_ind[-1]
del err_ind[-1]
########################################
'''

plk_file = open(twe_file,'r')
twi_data_place, twi_data_coor = pickle.load(plk_file)
plk_file.close()


twi_with_place = []
err_ind = []
skip_ind = False
trig_err = False


#place_counter = 0
for elem_ind in range(STAR_IND, len(twi_data_place)):
    elem = twi_data_place[elem_ind]
    goo_result = goo_setup.GoogPlac_name(elem['lat'],elem['lon'],elem['place_name'], PLACE_RAD, goo_key)
    #quer_elem = {'index':place_counter, 'place_name': elem['place_name'],
    #             'lat':elem['lat'], 'lon':elem['lon'],
    #             'goo_result':goo_result['results'][:min(MAX_QUERY_COUNT,len(goo_result['results']))]}
    if goo_result.has_key('error_message'):
        if str(goo_result['error_message']).__eq__('You have exceeded your daily request quota for this API.'):
            err_ind.append(elem_ind)
            print 'QUERY LIMIT at {}'.format(elem_ind)
            break
        else:
            #err_ind.append(elem_ind)
            trig_err = True

    if len(goo_result['results']) < PLACE_MIN_QUERY_COUNT or trig_err:
        goo_result = goo_setup.GoogPlac_name(elem['lat'], elem['lon'], elem['place_name'], PLACE_RAD + 100, goo_key)
        if goo_result.has_key('error_message'):
            if str(goo_result['error_message']).__eq__('You have exceeded your daily request quota for this API.'):
                err_ind.append(elem_ind)
                print 'QUERY LIMIT at {}'.format(elem_ind)
                break
            elif trig_err:
                err_ind.append(elem_ind)
                skip_ind = True

    if skip_ind:
        elem['query_succ'] = False
        elem['goo_result'] = goo_result['error_message']
        print goo_result['error_message']
    else:
        elem['query_succ'] = True
        elem['goo_result'] = goo_result['results'][:min(PLACE_MAX_QUERY_COUNT, len(goo_result['results']))]

    #print(json.dumps(quer_elem))
    twi_with_place.append(elem)
    #place_counter += 1
    trig_err = False
    skip_ind = False

if elem_ind >= len(twi_data_place) - 1:
    err_ind.append(elem_ind)
    print 'Complete'
else:
    print 'Incomplete, curr ind: {}'.format(elem_ind)

with open(twe_goo_place_file, 'w') as out_file:
    pickle.dump([twi_with_place, err_ind], out_file)

'''
STAR_IND = 0
twi_with_coor = []
err_ind = []
skip_ind = False
trig_err = False


#place_counter = 0
for elem_ind in range(STAR_IND, len(twi_data_coor)):
    elem = twi_data_coor[elem_ind]
    goo_result = goo_setup.GoogPlac(elem['lat'],elem['lon'], COOR_RAD, None, goo_key)

    if goo_result.has_key('error_message'):
        if str(goo_result['error_message']).__eq__('You have exceeded your daily request quota for this API.'):
            err_ind.append(elem_ind)
            print 'QUERY LIMIT at {}'.format(elem_ind)
            break
        else:
            #err_ind.append(elem_ind)
            trig_err = True

    if len(goo_result['results']) < COOR_MIN_QUERY_COUNT or trig_err:
        goo_result = goo_setup.GoogPlac(elem['lat'], elem['lon'], COOR_RAD + 100, None, goo_key)
        if goo_result.has_key('error_message'):
            if str(goo_result['error_message']).__eq__('You have exceeded your daily request quota for this API.'):
                err_ind.append(elem_ind)
                print 'QUERY LIMIT at {}'.format(elem_ind)
                break
            elif trig_err:
                err_ind.append(elem_ind)
                skip_ind = True

    if skip_ind:
        elem['query_succ'] = False
        elem['goo_result'] = goo_result['error_message']
        print goo_result['error_message']
    else:
        elem['query_succ'] = True
        elem['goo_result'] = goo_result['results'][:min(COOR_MAX_QUERY_COUNT, len(goo_result['results']))]

    #print(json.dumps(quer_elem))
    twi_with_coor.append(elem)
    #place_counter += 1
    trig_err = False
    skip_ind = False

if elem_ind >= len(twi_data_coor) - 1:
    err_ind.append(elem_ind)
    print 'Complete'
else:
    print 'Incomplete, curr ind: {}'.format(elem_ind)

with open(twe_goo_coor_file, 'w') as out_file:
    pickle.dump([twi_with_coor, err_ind], out_file)
'''



