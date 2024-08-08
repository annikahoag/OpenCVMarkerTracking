#------------------------------------------------------------------------Imports to be used for various functionalities--------------------------------- 
import cv2 as cv # Importing OpenCV library
import os # For operating system dependent functionalities 
import subprocess # Used to run command line command to get video creation time

# To be used to write information to a CSV file
import csv
from csv import writer 

# To ignore an error when indexing into list of corners
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# For figuring out when to start using Claudi's GNSS data
import time
import calendar

# For plotting data
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker 
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter)
import numpy  as np
import math

# Use when running on remote machine
import matplotlib
matplotlib.use('tkagg')



#--------------------------------------------------------------------------------------------Constants--------------------------------------------------
# Fixed geographical postions of the base markers, formatted as [M1[lat, long], M2[lat, long]]
BASE_MARKER_POS = [[50.92566303, 13.33171600], [50.92570884, 13.33163245]]

# Actual geographical coordinates of the markers (not including base markers and Claudi)
REAL_MARKER_POS =[[50.92573663, 13.33171395], [50.92569358, 13.33170130], [50.92570700, 13.33176451], [50.92569378, 13.33164172], [50.92565252, 13.33170003]]

# Stores our predefined dictionary to be used for marker tracking, 4x4 bits and can contain 50 markers
DICT = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)

# Video we're taking frames from, corresponding CSV file for GNSS data, and index of last row in that file
VID = "DJI_0596.MP4"
CSV_FILE = "converted_bag4.csv"
NUM_ROWS = 19

# Calculated frames per second of our 3 videos
FPS = 29.8

# Number of markers we have including base, robot, and all others
NUM_MARKERS = 8

# Colors to be used for each marker when we graph, index 0 used for marker 0, 1 for marker 1 and so on
COLORS = ["gray", "black", "pink", "red", "orange", "green", "blue", "purple"]



#---------------------------------------------------------------------------------------------Functions-----------------------------------------------
"""Function to generate our markers (only need to do once for printing the markers)
    :param dict: Dictionary object we created previously"""
def createMarkers(dict):
    markers = [None]*NUM_MARKERS

    for i in range(len(markers)):
        markers[i] = cv.aruco.generateImageMarker(dict, i, 100)
        cv.imwrite(("marker" + str(i) + ".png"), markers[i])



"""Run command line process to find the creation time of video
    :return [year, month, day, hour, min, sec]: List representing this date and time""" 
def calcStartTime():

    # Command which contains the video creation time 
    result = subprocess.run(["ffprobe", VID], capture_output=True, text=True)

    # Write the result of the subprocess to a text file as a string
    with open("mediaInfo.txt", "w") as f:
        f.write(str(result))
      
    # Read in result of subprocess from text file, split it into individual phrases by spaces
    with open("mediaInfo.txt", "r") as file:
        result = file.read()
        result = result.split(" ")

        # Find phrase "creation_time"
        for i in range(len(result)):
            # The time we want comes after the phrase "creation_time"
            if result[i].startswith("creation_time"):
                
                # Loop until we're past the spaces and : (the default formatting each time we run the command line process above)
                j=i+1
                while(result[j]=="" or result[j]==":"):
                    j+=1

                # Only return the creation date and time string located at result[j]
                # Format will always be YYYY-MM-DDTHH:MM:SS.SSSSSSZ\n i.e. 2024-05-29T11:39:41.000000Z\n when we run the above command
                date = result[j][0:10]
                time = result[j][11:26]
                
                break

        # Split the date and time into individual integer/float units, always will be formatted like this: 2024-05-29 and 11:39:41.000000Z\n
        year = int(date[0:4])
        month = int(date[5:7])
        day = int(date[8:10])
        hour = int(time[0:2]) + 2 # Linux has the time zone as UTC -> convert to CEST by adding 2
        min = int(time[3:5])
        sec = float(time[6:15])

        # Return units as a list
        return [year, month, day, hour, min, sec]



"""Function to figure out what time the current frame was taken at, assumes 24 hr time
    :param startTime: The time the video was created at, will be formatted as [year, month, day, hour, minute, second]
    :param timeAdd: Represents the number of seconds we should add to the starting time
    :return [yearFinal, monthFinal, dayFinal, hrFinal, minFinal, secFinal]: List representing the time the current frame was taken at""" 
def calcTimeStamp(startTime, timeAdd):

    # Indexing startTime list to see what each unit was at the beginning of the video 
    # Only need these variables separate from the final values for year, month, and second
    yearStart = startTime[0]
    monthStart = startTime[1]
    secStart = startTime[5]


    # Calculating what the final times of each unit would be if we just simply added timeAdd
    yearFinal = yearStart
    monthFinal = monthStart
    dayFinal =  startTime[2]
    hrFinal = startTime[3]
    minFinal = startTime[4]
    secFinal = timeAdd + secStart


    # Check to see if adding timeAdd would bring us to a new minute
    while secFinal >= 60:
        # Update how much more time needs to be added after we've reached 60 seconds
        timeAdd = timeAdd - (60 - secStart)
        minFinal += 1

        # Check to see if increasing our minute would bring us to a new hour
        # minFinal should never be >60 since we're only ever adding 1 to a number that is at most 59
        if minFinal == 60:
            minFinal = 0
            hrFinal +=1

            # Check if increasing our hour would bring us to a new day
            if hrFinal == 24:
                hrFinal = 0
                dayFinal += 1

                # Check if increasing our day would bring us to a new month
                if ((dayFinal>30 and (monthStart==9 or monthStart==4 or monthStart==6 or monthStart==11)) or
                    (dayFinal>31 and (monthStart==1 or monthStart==3 or monthStart==5 or monthStart==7 or monthStart==8 or 
                        monthStart==10 or monthStart==12)) or
                    (dayFinal>29 and monthStart==2 and yearStart%4==0) or 
                    (dayFinal>28 and monthStart==2 and yearStart%4!=0)):
                    monthFinal += 1
                    dayFinal = 1

                    # Check if increasing our month would bring us to a new year 
                    if monthFinal > 12:
                        monthFinal = 1
                        yearFinal += 1
        
        secFinal = timeAdd
        secStart = 0
    
    secFinal = secStart + timeAdd

    return [yearFinal, monthFinal, dayFinal, hrFinal, minFinal, secFinal]



"""Find center of the marker whose corner coordinates are given in cornerCoords
    :param cornerCoords: 2D list containing the 4 corner coordinates of a marker, each index is a point formatted as [x,y] in units of pixels
    :return [float(centerX), float(centerY)]: List representing the coordinates of the center of the marker, in units of pixels"""
def calcCenter(cornerCoords):

    # Coordinates of 3/4 of corners (don't need all 4)
    topLeft = cornerCoords[0]
    topRight = cornerCoords[1]
    bottomLeft = cornerCoords[3]

    # Find center on x and y axis
    centerX = (topLeft[0] + topRight[0]) / 2
    centerY = (topLeft[1] + bottomLeft[1]) / 2

    # Return center of the entire marker
    return [float(centerX), float(centerY)]



"""Martin Plank's function to calculate the haversine distance between two points
    @arg lat1: Latitude of Point1.
    @arg lon1: Longitude of Point1.
    @arg lat2: Latitude of Point2.
    @arg lon2: Longitude of Point2.
    @return Haversine-distance in meter"""
def get_distance_haversine(lat1, lon1, lat2, lon2):

    R = 6371000

    dx = math.radians(lat2 - lat1)
    dy = math.radians(lon2 - lon1)

    a = (math.sin(dx/2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        *math.sin(dy/2) **2)

    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

    d = R * c

    return d



"""Calculate scaling factor for pixels to meters
    :param pt1: Pixel coordinates of base marker 0
    :param pt2: Pixel coordinates of base marker 1
    :param lat1: Latitude of base marker 0
    :param lat2: Latitude of base marker 1
    :param long1: Longitude of base marker 0
    :param long2: Longitude of base marker 1
    :return scale: The calculated scaling factor"""
def calcScale(pt1, pt2, lat1, lat2, long1, long2):

    # Calculate the distance between the 2 markers in meters
    meters = get_distance_haversine(lat1, long1, lat2, long2)

    # Calculate the distance between the 2 markers in pixels
    pixels = math.dist(pt1, pt2)

    # Calculate the scaling factor
    scale = meters / pixels

    return scale



"""Function to write the given data to one of our CSV files
    :param data: List of strings, each item is a column
    :param fileName: The CSV file to write data to"""
def writeToCSV(data, fileName):

    # Use writer from CSV to append information to CSV file
    with open(fileName, "a") as f3:
        writerObject = writer(f3)
        writerObject.writerow(data)



"""Find geographical location of robot and other markers using fixed position of base markers on each frame, write that to a CSV file as well as a time stamp and the frame number that we're on
Also keeps track of how many markers we've skipped overall
    :param name: The string that we should name the image we're examining
    :param frame: The image itself
    :param frameNum: The integer representing which frame we're on, used to calculate time stamp and be written to CSV
    :param skipCount: The number of markers we've skipped so far
    :return skipCount: Updated number of markers we've skipped so far
    :return positions: List of each detected marker's position in meters, if marker wasn't detected that index is set to [None, None]
    :return timeStamp: Date/time that the current frame was taken at, formatted as [YYYY, MM, DD, HH, MM, SS.SSSSSS]"""
def findLoc(name, frame, frameNum, skipCount):

    # Write and then read in image that we got from the video
    cv.imwrite(name, frame)
    frame = cv.imread(name)

    
    # Create a CSV file to write location of robot and markers, frame number, and timestamps, and write headers if file doesn't already exist
    if not os.path.isfile("locationTracked.csv"):
        with open("locationTracked.csv", "w") as csvF:
            pass

        # Write headers 
        headers = ["Marker", "Location (meters)", "Frame", "Date", "Time"]
        writeToCSV(headers, "locationTracked.csv")


    # Nested list to contain the pixel location of all markers, each index has list formatted as [x coordinate, y coordinate], index # corresponds to the marker at those pixel coordinates
    ptsPix = [[None]*2]*NUM_MARKERS 


    # Nested list to contain the location in meters of all markers, each index has list formatted as [x coordinate, y coordinate], index # corresponds to the marker at those meter coordinates
    positions =[[None]*2]*NUM_MARKERS


    # Creating detection parameters and then editing the threshold values
    arucoParams = cv.aruco.DetectorParameters()
    arucoParams.adaptiveThreshWinSizeMin =10
    arucoParams.adaptiveThreshWinSizeMax=40
    arucoParams.adaptiveThreshConstant=15


    # Detect our markers
        # corners = a list with the pixel x-y coordinates of the corners of the markers we detected
        # ids = ids of the markers we detected
        # rejectedImgPoints = a list of possible markers that were found but were later rejected, used for testing/debugging only 
    detector = cv.aruco.ArucoDetector(DICT, arucoParams)
    corners, ids, rejectedImgPoints = detector.detectMarkers(frame)


    # Calculate time stamp that this frame was taken at 
    # Format: [YYYY, MM, DD, HH, MM, SS.SSSSSS]
    timeStamp = calcTimeStamp(calcStartTime(), (frameNum/FPS))
    
    # Creating a readable string of the timeStamp to write to our CSV file later
    dateStr = str(timeStamp[2]) + "/" + str(timeStamp[1]) + "/" + str(timeStamp[0])
    timeStr = str(f"{int(timeStamp[3]):02d}") + ":" + str(f"{int(timeStamp[4]):02d}") + ":" + str("{{:.{}f}}".format(6).format(timeStamp[5]))


    # Try-except block for if ids=None i.e. no markers were detected and we can't index into ids 
    # Put pixel coordinates of the center of all markers into ptsPix or [None, None] if marker was not detected
    try:

        # Make sure that the 2 base markers and Claudi have been detected before attempting to track any of the markers
        if ([0] in ids) and ([1] in ids) and ([2] in ids):

            # Putting center coordinates of the detected markers into our list of pixel coordinates
            # We can skip any of these markers if they weren't detected since we have the 3 necessary ones
            for i in range(len(ids)):

                # Second part of conditional is for if we mistakingly don't reject something that looked like a marker and end up w/ too many markers
                if ids[i] != None and ids[i]<NUM_MARKERS:
                    
                    # Markers are detected & put into ids in a random order, but we want the Marker ID to correspond to the index in ptsPix
                    ptsPix[int(ids[i])] = calcCenter(corners[i][0])


            # Setting up variables for conversions
            x1 = ptsPix[0][0] # x pixel of base marker 0
            y1 = ptsPix[0][1] # y pixel of base marker 0
            x2 = ptsPix[1][0] # x pixel of base marker 1
            y2 = ptsPix[1][1] # y pixel of base marker 1
            lat1 = BASE_MARKER_POS[0][0] # latitude of base marker 0
            long1 = BASE_MARKER_POS[0][1] # longitude of base marker 1
            lat2 = BASE_MARKER_POS[1][0] # latitude of base marker 0
            long2 = BASE_MARKER_POS[1][1] # longitude of base marker 1

            # Calculate the scaling factor for pixels to meters
            scale = calcScale([x1, y1], [x2, y2], lat1, lat2, long1, long2)

            # Convert x/y pixel coordinates to meters 
            for i in range(NUM_MARKERS):
                if ptsPix[i] != [None, None]:
                    positions[i] = [ptsPix[i][0]*scale, ptsPix[i][1]*scale]
                else:
                    positions[i] = [None, None]


    # If error comes up when none of the markers were detected we can just continue to writing "skipped" in CSV file
    except TypeError:
        pass


    # Write location, frame number, and time stamps (or "skipped") of all markers to CSV file 
    for i in range(NUM_MARKERS):

        # If marker was detected, write location to CSV file
        if ptsPix[i] != [None, None]:

            # List of information to be written to the CSV file -> marker number, location, frame number, date, and time 
            data = [str(i), str(str("{{:.{}f}}".format(8).format(positions[i][0])) + ", " + str("{{:.{}f}}".format(8).format(positions[i][1])) ), str(frameNum), dateStr, timeStr]
            
        else:
            # List of information to be written to the CSV file -> marker number, skipped note, frame number, date, and time
            data = [str(i), "skipped", str(frameNum), dateStr, timeStr]
            skipCount+=1

        # Write data to CSV file for the tracked locations
        writeToCSV(data, "locationTracked.csv")


    # Return number of times we've skipped up to this frame, meters locations, and the date/time of this frame
    return skipCount, positions, timeStamp



"""Formula for converting latitude into meters, source: https://stackoverflow.com/questions/639695/how-to-convert-latitude-or-longitude-to-meters/39540339#39540339 (2nd answer) 
    :param lat: Latitude of the point we're converting
    :return: Result of formula"""
def get_m_per_deg_lat(lat):
    return 111132.954 - 559.822 * math.cos(2 * math.radians(lat)) + 1.175 * math.cos(4 * math.radians(lat))



"""Formula for converting longitude into meters, source: https://stackoverflow.com/questions/639695/how-to-convert-latitude-or-longitude-to-meters/39540339#39540339 (2nd answer)
    :param lat: Latitude of the point we're converting
    :return: Result of formula"""
def get_m_per_deg_lon(lat):
    return 111132.954 * math.cos(math.radians(lat))



"""Function to convert a point from latitude/longitude coordinates into meters, source: https://stackoverflow.com/questions/639695/how-to-convert-latitude-or-longitude-to-meters/39540339#39540339 (2nd answer)
    :param lat: Latitude of the point to be converted
    :param lon: Longitude of the point to be converted
    :return [meterX, metersY]: Inputted point's latitude and longitude converted into an x and y coordinate in meters, respectively
    :return [None, None]: Returned if either or both of lat and lon equal None b/c that means it's impossible to convert"""
def convertLatLonToMeters(lat, lon):

    if lat!=None and lon!=None:
        metersX = lat * get_m_per_deg_lat(BASE_MARKER_POS[0][0])
        metersY = lon * get_m_per_deg_lon(BASE_MARKER_POS[0][0])
        return [metersX, metersY]
    else:
        return [None, None]



"""Read in the seconds since epoch time of Claudi from our file with GNSS data 
    :param index: Row number to read the time from
    :return int(secSinceEpoch): The seconds since epoch time (as an integer) in the file corresponding to the row index  saying how many times we've already read in a location"""
def getClaudiSecSinceEpoch(index):

    # Read in CSV file
    with open(CSV_FILE, "r") as f:

        # Making 2D list containing all the elements of our file
        data = []
        reader = csv.reader(f)
        for line in reader:
            data.append(line)

        # Seconds since epoch is always in column 8
        secSinceEpoch = data[index][8]
    
    return int(secSinceEpoch)



"""Read in Claudi's new latitude and longitude from CSV file to be set to Claud's correct current position
    :param newIndex: The row of the CSV file that contains the new location to be read in
    :return [float(newLat), float(newLon)]: The new latitude and longitude of Claudi (as floats)"""
def changeRobotLat(newIndex):
    
    # Read in CSV file
    with open(CSV_FILE, "r") as f:

        # Making 2D list containing all the elements of our file
        data = []
        reader = csv.reader(f)
        for line in reader:
            data.append(line)

    # We always read from the 9th and 10th column by the way the file is formatted
    newLat = data[newIndex][9]
    newLon = data[newIndex][10]

    return [float(newLat), float(newLon)]



"""Function to rotate a point after we've done the offset
    :param originX: x coordinate of the point we're rotating around
    :param originY: y coordinate of the point we're rotating around
    :param ptX: x coordinate of the point to be rotated
    :param ptY: y coordinate of the point to be roatated
    :param measBaseMarker0: Measured location (in meters) of base marker 0
    :param measBaseMarker1: Measured location (in meters) of base marker 1
    :param realBaseMarker0: Hard-coded correct location (in latitude and longitude) of base marker 0
    :param realBaseMarker1: Hard-coded correct location (in latitude and longitude) of base marker 1
    :return [ptX, ptY]: Rotated point"""
def rotate(originX, originY, ptX, ptY, measBaseMarker0, measBaseMarker1, realBaseMarker0, realBaseMarker1):

    # Calculate angle between measured base markers
    yDist1 = measBaseMarker1[1] - measBaseMarker0[1]
    xDist1 = measBaseMarker1[0] - measBaseMarker0[0]
    angle1 = math.atan((yDist1/xDist1))

    # Calculate angle between real base markers
    yDist2 = realBaseMarker1[1] - realBaseMarker0[1]
    xDist2 = realBaseMarker1[0] - realBaseMarker0[0]
    angle2 = math.atan((yDist2/xDist2))

    # Final angle to be used for rotation
    angle = math.radians(180) + angle1 - angle2

    # Translate point back to the origin
    ptX = ptX - originX
    ptY = ptY - originY

    # Rotate the point
    newX = (ptX*(math.cos(angle))) - (ptY*(math.sin(angle)))
    newY = (ptX*(math.sin(angle))) + (ptY*(math.cos(angle)))

    # Translate point back away from the origin
    ptX = newX + originX
    ptY = newY + originY

    return [ptX, ptY]



"""Function to offset the meters that were converted from latitude/longitude to match with meters that were converted from pixels
    :param meters: List of every correct marker location in meters
    :param measuredBase0: Pixel coordinates of base marker 0 we tracked in findLoc
    :param realBase0: Correct latitude/longitude (converted into meters) location of base marker 0
    :param marker0: Most recent tracked position of base marker 0
    :param marker1: Most recent tracked position of base marker 1
    :return metersOffset: meters list with offset done on each location"""
def offset(meters, measuredBase0, realBase0, marker0, marker1):
    
    # List to contain all the locations in meters wiht the offset done 
    metersOffset = meters.copy()
    
    # If base marker 0 wasn't able to be tracked or base marker 1 wasn't able to be tracked, we can't do offset
    if measuredBase0!=[None, None] and realBase0!=[None, None] and marker1!=[None, None]:

        # Calculate the delta in the x and y direction needed to correctly offset each point
        xDist = realBase0[0] - measuredBase0[0]
        yDist = realBase0[1] - measuredBase0[1]
        
        # Correct locations of base markers 0 and 1 respectively
        realMarker0 = meters[0]
        realMarker1 = meters[1]

        # Loop through real locations of markers to offset and rotate them 
        for i in range(0, NUM_MARKERS):

            if i==2:
                # When i=2 that means we have to offset each meters location of Claudi
                for j in range(len(meters[i])):
                    if meters[i][j] != [None, None]:
                        metersOffset[i][j] = rotate(realBase0[0]-xDist, realBase0[1]-yDist, meters[i][j][0]-xDist, meters[i][j][1]-yDist, marker0, marker1, realMarker0, realMarker1)
                    else:
                        metersOffset[i][j] = [None, None]
            else:
                metersOffset[i] = rotate(realBase0[0]-xDist, realBase0[1]-yDist, meters[i][0]-xDist, meters[i][1]-yDist, marker0, marker1, realMarker0, realMarker1)
    

    # Means we couldn't track base marker 0 so we return a list of nones since we can't do the offset     
    else:
        for i in range(0, NUM_MARKERS):
            if i==2:
                for j in range(len(meters[i])):
                    metersOffset[i][j] = [None, None]
            else:
                metersOffset[i] = [None, None]
        

    return metersOffset



"""Function to print out the most recent tracked position of each marker
    :param pos: Nested list containing the newest coordinate locations of the markers"""
def printRecentMarkers(pos):
    for i in range(len(pos)):
        printStr = "Marker " + str(i) + ": " + str(pos[i]) + "\n"
        print(printStr)
      


"""Function to account for any movement the camera may have had
    :param origFrame: List of coordinates that are the locations of each marker when they were detected for the first time
    :param currentFrame: List of coordinates that are the most recent tracked location of each marker
    :return currentFrame: The most recent tracked locations of each marker after adjusting coordinates for any camera shifting"""
def shiftForCamera(origFrame, currentFrame):

    # List to contain differences  between the last tracked location and the first tracked location of each marker in the x and y directions respectively
    xDifferences = []
    yDifferences = []
        

    # Calculate differences between first and most recent frames, except for Claudi's (i=2) since this marker should be moving and will skew the other differences
    for i in range(len(origFrame)):
        if currentFrame[i]!=[None, None] and origFrame[i]!=[None, None] and i!=2:
            xDifferences.append(currentFrame[i][0] - origFrame[i][0])
            yDifferences.append(currentFrame[i][1] - origFrame[i][1])


    # Calculate the median of the distances
    medianX=0
    medianY=0
    if len(xDifferences)>0:
        xDifferences.sort()
        medianX = xDifferences[int(len(xDifferences)/2)]
    if len(yDifferences)>0:
        yDifferences.sort()
        medianY = yDifferences[int(len(yDifferences)/2)]


    # Subract that median from every marker tracked in this frame
    for i in range(len(currentFrame)):
        if currentFrame[i] != [None, None]:
            currentFrame[i][0] = currentFrame[i][0] - medianX 
            currentFrame[i][1] = currentFrame[i][1] - medianY 


    return currentFrame



"""Calculate the distance between each measured marker and the correct location of that marker
    :param allMeters: List of all newly detected markers' positions in meters
    :param realMarkerPosMeters: List of all correct marker location in meters
    :return distances: List of the calculated distances, each index corresponds to the marker number with that distance between measured and correct locations"""
def makeDistList(allMeters, realMarkerPosMeters):
    
    # List to store the distances between measured and correct markers
    distances = []


    # Use Python's distance function to calculate the distances and append to our list
    for i in range(len(allMeters)):

        # When we reach Claudi (i=2) we have to make sure we calculate the distance relative to Claudi's most recent correct location
        if i==2:
            if allMeters[i] != [None, None] and realMarkerPosMeters[i][len(realMarkerPosMeters[i])-1] != [None, None]:
                dist = abs(math.dist(allMeters[i], realMarkerPosMeters[i][len(realMarkerPosMeters[i])-1]))
                distances.append(dist)
            else:
                distances.append(None)

        else:
            if allMeters[i] != [None, None]:
                dist = abs(math.dist(allMeters[i], realMarkerPosMeters[i]))
                distances.append(dist)
            else:
                distances.append(None)


    return distances 



"""Function to draw our graph in units of meters 
    :param metersPoints: Coordinates of every point (tracked and correct) converted to meters
    :param frameNum: Frame we're currently on, used to make graph title"""
def drawMetersGraph(metersPoints, frameNum):
    
    # Create subplot
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Setting parameters for x axis 
    plt.xlim(0, 20)
    start, end = ax.get_xlim()
    ax.xaxis.set_ticks(np.arange(start, end, 10))
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_minor_locator(MultipleLocator(1))

    # Setting parameters for y axis
    plt.ylim(0, 20)
    start, end = ax.get_ylim()
    ax.yaxis.set_ticks(np.arange(start, end, 10))
    ax.yaxis.set_major_locator(MultipleLocator(5))
    ax.yaxis.set_minor_locator(MultipleLocator(1))


    # Loop through our 6 colors representing each of the markers 
    for i in range(NUM_MARKERS):

        # Loop by incrementing by NUM_MARKERS so we draw each occurence of a marker with their respective color
        # Triangles for the correct meters, circles for the measured ones
        for j in range(i, len(metersPoints), NUM_MARKERS):
            if j!=2:
                if (j>=0 and j<NUM_MARKERS) and metersPoints[j] != [None, None]:
                    ax.plot(metersPoints[j][0], metersPoints[j][1], marker="^", markersize=5, markeredgecolor=COLORS[i], markerfacecolor=COLORS[i])
                elif metersPoints[j] != [None, None]:
                    ax.plot(metersPoints[j][0], metersPoints[j][1], marker="o", markersize=5, markeredgecolor=COLORS[i], markerfacecolor=COLORS[i])
            
            else:
                for k in range(len(metersPoints[j])):
                    if metersPoints[j][k] != [None, None]:
                        ax.plot(metersPoints[j][k][0], metersPoints[j][k][1], marker="^", markersize=5, markeredgecolor=COLORS[i], markerfacecolor=COLORS[i])


    # Labeling axises and graph title
    ax.set_xlabel('Meters (Latitude)', fontsize=12)
    ax.set_ylabel('Meters (Longitude)', fontsize=12)
    plt.title("Meters", loc="center")
    name = str(frameNum) + "meters.png"

    # Save plot in directory
    plt.savefig(name, dpi='figure', format=None)

    # Display graph
    plt.grid()
    plt.show()



"""Function to draw boxplot representing distances between real and measured location of each marker
    :param distances: Distances between each measured marker and its corresponding correct location
    :param frameNum: Frame we're currently on, used to make graph title"""
def drawBoxplot(distances, frameNum):
    
    # Create subplot
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Create boxplot instance
    ax.boxplot(distances)

    # Setting up tick labels to go with our marker nunbers since ours start at 0 (default is starting at 1)
    origTicks = [] 
    newLabels = []
    for i in range(1, NUM_MARKERS+1):
        origTicks.append(i)
        newLabels.append(i-1)
    plt.xticks(origTicks, newLabels)

    # Label axises and graph title
    ax.set_xlabel('Marker Number')
    ax.set_ylabel('Distance Btwn Measured & Correct (m)', fontsize=12)
    name = str(frameNum) + "boxplot.png"

    # Save plot in directory
    plt.savefig(name, dpi='figure', format=None)

    # Display boxplots
    plt.show()



"""Our main function
For each frame of our video set up parameters to write an image in findLoc and draw graphs of results, also write the number of skips to a text file after the last frame
    :param vid: The video we're examining"""
def getImage(vid):

    #----------------------------------------------Variables------------------------------------------
    # To keep track of which frame we're on
    frameNum = 1

    # List to store every tracked and correct location in meters, allows us to graph locations over time
    allMeters = []

    # Lists to store Claudi's GNSS points (lat/lon and meters respectively)
    realClaudi = []
    realClaudiMeters = []

    # Lists to store parameters for calling makeDist
    realMarkerPosMeters = []

    # Used to store the distance btwn each marker's measured location and the actual location
    # Formatted as nested lists, where each inner list is the distances for the marker corresponding to that index
    allDistances =[[] for i in range(NUM_MARKERS)]   

    # Index representing which row in the CSV file we're at
    robotIndex = 0

    # Flag representing if we can start comparing Claudi with GNSS data (times don't necessarily line up in the beginning)
    startGNSS = False

    # List to contain the position of each marker at the frame for which the marker was successfully tracked for the 1st time
    firstTracked = [[None, None] for i in range(NUM_MARKERS)]

    # Number of indices in firstTracked that are not equal to [None, None]
    numTracked  = 0

    # Running count of the number of skips we have
    skipCount = 0


    # Loop until we have read every frame from the video
    while(True):

        # OpenCV function to read the video and take out a frame
        # ret tells us whether or not the reading was successful, unsuccessful means we've read every frame
        ret, frame = vid.read()


        # If there is still more of the video left to read
        if ret:

            # Print frame number every iteration 
            print("\n\nFrame: ", frameNum, "\n")

            # Name our image based on the frame we're at 
            imgName = str(frameNum) + ".jpg"


            # Find the location of the robot in this frame, keep count of how many times we've skipped a marker, and get the date/time of the frame
            skipCount, positions, timeStamp = findLoc(imgName, frame, frameNum, skipCount)


            # Update firstTracked if a marker has now been tracked for the firts time
            if numTracked < NUM_MARKERS:
                for i in range(len(firstTracked)):
                    if firstTracked[i] == [None, None]:
                        if positions[i] != [None, None]:
                            firstTracked[i] = positions[i].copy()
                            numTracked+=1


            # Adjust all new tracked locations if there was any movement with the camera 
            positions = shiftForCamera(firstTracked, positions)


            # Store the tracked location of base marker 0 (meters) and the unconverted correct location of base marker 0
            metersBase0 = positions[0]
            latLonBase0 = [BASE_MARKER_POS[0][0], BASE_MARKER_POS[0][1]]


            # Create a list to store the correct location of each marker in meters, we empty each time to be able to show progression over time of Claufdi
            realMarkerPosMeters = []

            # Convert correct locations of both base markers 0 and 1
            realMarkerPosMeters.append(convertLatLonToMeters(BASE_MARKER_POS[0][0], BASE_MARKER_POS[0][1]))
            realMarkerPosMeters.append(convertLatLonToMeters(BASE_MARKER_POS[1][0], BASE_MARKER_POS[1][1]))


            # Convert timestamp to seconds since epoch to decide if we should start looking at Claudi's GNSS yet
            timeStamp[3]-=2 #Account for different timezones
            secSinceEpoch = int(calendar.timegm(timeStamp))
            secSinceEpochClaudi = getClaudiSecSinceEpoch(robotIndex+1)
        
            # Check to see if we've reached the next row (next time) in the GNSS CSV file
            if secSinceEpoch==secSinceEpochClaudi and robotIndex<=NUM_ROWS:

                # Append Claudi's new location
                realClaudi.append(changeRobotLat(robotIndex+1))
                
                # Convert new location to meters
                realClaudiMeters.append(convertLatLonToMeters(realClaudi[robotIndex+1][0], realClaudi[robotIndex+1][1]))

                # Increment which row we're at 
                robotIndex+=1

            else:
                
                # Convert all previous Claudi locations to meters if not at first frame
                if frameNum>1:
                    realClaudiMeters=[]
                    for i in range(len(realClaudi)-1):
                        realClaudiMeters.append(convertLatLonToMeters(realClaudi[i][0], realClaudi[i][1])) 
                    # Convert most recent Claudi location to meters
                    realClaudiMeters.append(convertLatLonToMeters(realClaudi[robotIndex][0], realClaudi[robotIndex][1]))
                else:
                    realClaudi.append([None, None])
                    realClaudiMeters.append([None, None])

            # Add Claudi's correct location to the list of the real marker positions 
            realMarkerPosMeters.append(realClaudiMeters)


            # Convert the rest of the markers to meters
            for i in range(len(REAL_MARKER_POS)):
                if (REAL_MARKER_POS[i] != [None, None]):
                    meters = convertLatLonToMeters(REAL_MARKER_POS[i][0], REAL_MARKER_POS[i][1])
                    realMarkerPosMeters.append(meters)
             

            # Offset all of the correct locations we've converted to meters 
            realMarkerPosMeters = offset(realMarkerPosMeters, metersBase0, convertLatLonToMeters(latLonBase0[0], latLonBase0[1]), positions[0], positions[1])
           

            # Create a CSV file to write correct location of robot and markers, frame number, and timestamps, and write headers if file doesn't already exist
            if not os.path.isfile("locationCorrect.csv"):
                with open("locationCorrect.csv", "w") as csvF:
                    pass

                # Write headers 
                headers = ["Marker", "Location (meters)", "Frame", "Date", "Time"]
                writeToCSV(headers, "locationCorrect.csv")
            
            # Creating a readable string of the timeStamp to write to our CSV file
            dateStr = str(timeStamp[2]) + "/" + str(timeStamp[1]) + "/" + str(timeStamp[0])
            timeStr = str(f"{int(timeStamp[3])+2:02d}") + ":" + str(f"{int(timeStamp[4]):02d}") + ":" + str("{{:.{}f}}".format(6).format(timeStamp[5]))
            
            # Set up data to be written to CSV files
            for i in range(NUM_MARKERS):
    
                # List of information to be written to the CSV file -> location, frame number, date, and time 
                if i!=2:
                    if realMarkerPosMeters[i]!=[None, None]:
                        data = [str(i), str(str("{{:.{}f}}".format(8).format(realMarkerPosMeters[i][0])) + ", " + str("{{:.{}f}}".format(8).format(realMarkerPosMeters[i][1])) ), 
                        str(frameNum), dateStr, timeStr]    
                    else:
                        data = [str(i), "skipped", str(frameNum), dateStr, timeStr]    
                else:
                    lastClaudiIndex = len(realMarkerPosMeters[i])-1
                    if realMarkerPosMeters[i][lastClaudiIndex]!=[None, None]:
                        data = [str(i), str(str("{{:.{}f}}".format(8).format(realMarkerPosMeters[i][lastClaudiIndex][0])) + ", " + str("{{:.{}f}}".format(8).format(realMarkerPosMeters[i][lastClaudiIndex][1])) ), 
                        str(frameNum), dateStr, timeStr]    
                    else:
                        data = [str(i), "skipped", str(frameNum), dateStr, timeStr]                        
               
                # Write data to CSV file
                writeToCSV(data, "locationCorrect.csv")


            # Append offsetted correct marker locations to our list of all positions in meters
            if frameNum == 1:
                for marker in realMarkerPosMeters:
                    allMeters.append(marker)
            else:
                for i in range(len(realMarkerPosMeters)):
                    allMeters[i] = realMarkerPosMeters[i]
          
            
            # Append newly detected locations to our list of all positions in meters
            for i in positions:
                if (i != [None, None]):
                    allMeters.append(i)
                else:
                    allMeters.append([None, None])


            # Store a list of the distances between each measured marker and the coresponding correct marker's locations
            dist = makeDistList(allMeters[(len(allMeters)-(NUM_MARKERS)):len(allMeters)], realMarkerPosMeters)


            # Add the distances btwn real and measured markers to allDistances (so we have the correct format for drawing box plots)
            for i in range(len(allDistances)):
                if dist[i] != None:
                    allDistances[i].append(dist[i])


            # Create CSV file for distances if it doesn't already exist and write headers 
            if not os.path.isfile("distances.csv"):
            
                with open("distances.csv", "w") as csvF:
                    pass

                headers = []
                for i in range(NUM_MARKERS):
                    headers.append(str("Marker" + str(i)))
                writeToCSV(headers, "distances.csv")
            
            # Write newly calculated distances to CSV file
            writeToCSV(dist, "distances.csv")


            # Every 50 frames draw a graph with all meters locations and a box plot for each frame
            # If you want to display more or less times just change the number next to % accordingly 
            if frameNum%50 == 0:
                drawMetersGraph(allMeters, frameNum) 
                drawBoxplot(allDistances, frameNum)


            # Increment which frame we're on
            frameNum += 1


            # Remove the image from directory once we've finihsed our tracking so the folder doesn't get overcrowded
            os.remove(imgName)
        

        # If ret is False then we've finished reading all of the images of the frame so we leave our loop
        else:
            break


    # Create file to write the number of skips to if it doesn't already exist   
    if not os.path.isfile("skip.txt"):
        with open("skip.txt", "w") as f:
             pass
        
    # String containing number of skips to be written to skip.txt    
    writeSkipped = str(VID) + " skipped " + str(skipCount) + " times out of " + str(frameNum-1) + " frames (" + str(frameNum*6) + " opportunities to skip)." + "\n"       
    
    # Write number of skips to our text file now that we've completely gone thorugh the video
    with open ("skip.txt", 'a') as f:
        f.write(writeSkipped)


    # Display final graphs
    print("Final Graphs")
    drawMetersGraph(allMeters, frameNum)
    drawBoxplot(allDistances, frameNum)



# Called to make the markers (only has to be done once after you've chosen your number of markers)
#createMarkers(DICT)


# Call the main function using our video as a parameter
getImage(cv.VideoCapture(VID))