import sqlite3
from time import *
from lmtanalysis.Chronometer import Chronometer
from lmtanalysis.Animal import *
from lmtanalysis.Detection import *
from lmtanalysis.Measure import *
import numpy as np
from lmtanalysis.Event import *
from lmtanalysis.Measure import *
#from affine import Affine
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from lmtanalysis.EventTimeLineCache import EventTimeLineCached

def flush( connection ):
    ''' flush event in database '''
    deleteEventTimeLineInBase(connection, "Food Zone" )
    deleteEventTimeLineInBase(connection, "Food Stop" )


def reBuildEvent( connection, file, tmin=None, tmax=None, pool = None ):

    ''' use the pool provided or create it'''
    if ( pool == None ):
        pool = AnimalPool( )
        pool.loadAnimals( connection )
        pool.loadDetection( start = tmin, end = tmax, lightLoad=True )
    '''
    Event Food Zone
    - the animal is in the zone around the Food source
    - the animal is stopped in this zone for
    '''

    for animal in pool.animalDictionnary.keys():
        print(pool.animalDictionnary[animal])

        eventName1 = "Food Zone"
        eventName2 = "Food Stop"
        print ( "A is in the zone around the Food source")
        print ( eventName1 )

        FoodZoneTimeLine = EventTimeLine( None, eventName1 , animal , None , None , None , loadEvent=False )
        FoodStopTimeLine = EventTimeLine( None, eventName2 , animal , None , None , None , loadEvent=False )


        stopTimeLine = EventTimeLineCached( connection, file, "Stop", animal, minFrame=None, maxFrame=None )
        stopTimeLineDictionary = stopTimeLine.getDictionary()

        resultFoodZone={}
        resultFoodStop={}

        animalA = pool.animalDictionnary[animal]
        #print ( animalA )
        dicA = animalA.detectionDictionnary

        for t in dicA.keys():
            if (dicA[t].getDistanceToPoint(xPoint = 114, yPoint = 63) == None):
                continue

            #Check if the animal is entering the zone around the Food point:
            if (dicA[t].getDistanceToPoint(xPoint = 114, yPoint = 63) <= MAX_DISTANCE_TO_POINT*2):
                resultFoodZone[t] = True

            #Check if the animal is drinking (the animal should be in a tight zone around the Food point and be stopped):
            if (dicA[t].getDistanceToPoint(xPoint = 114, yPoint = 63) <= MAX_DISTANCE_TO_POINT):
                if t in stopTimeLineDictionary.keys():
                    resultFoodStop[t] = True


        FoodZoneTimeLine.reBuildWithDictionnary( resultFoodZone )
        FoodZoneTimeLine.endRebuildEventTimeLine(connection)

        FoodStopTimeLine.reBuildWithDictionnary( resultFoodStop )
        FoodStopTimeLine.removeEventsBelowLength( maxLen = MIN_Food_STOP_DURATION )
        FoodStopTimeLine.endRebuildEventTimeLine(connection)


    # log process
    from lmtanalysis.TaskLogger import TaskLogger
    t = TaskLogger( connection )
    t.addLog( "Build Event Food Point" , tmin=tmin, tmax=tmax )

    print( "Rebuild event finished." )







