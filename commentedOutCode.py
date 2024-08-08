# Place to store code I commented out, but am worried I may need again 
#-----------------------------------------------------------------------------------------------------------------------------------------



# For resizing an image
# import imutils

# Astropy imports 
# from astropy.io import fits
# from astropy import wcs

# Was to be used when converting between pixels and latitude/longitude 
# import numpy as np



#------------------------------Old Conversion Process--------------------------------------------
# Use the constant geographical location in MARK_POS to find geographical location of robot and each nonstationary marker 
# for i in range(2, NUM_MARKERS):
#     if ptsPix[i] != [None, None]:
#         positions[i-2] = [calcLat(ptsPix[0][1], ptsPix[1][1], MARKER_POS[0][0], MARKER_POS[1][0], ptsPix[i][1]), 
#                         calcLong(ptsPix[0][0], ptsPix[1][0], MARKER_POS[0][1], MARKER_POS[1][1], ptsPix[i][0])]


#--------------------------Attempt At Converting w/ Astropy--------------------------------------
# # Create wcs object
# wcsObj = createWCS(ptsPix[0], [(MARKER_POS[1][0]-MARKER_POS[0][0]), (MARKER_POS[1][1]-MARKER_POS[0][1])], MARKER_POS[0])

# # Use astropy library to find geographical location of robot and each nonstationary marker 
# pixsToConvert=[[None]*2]*(NUM_MARKERS-2)
# for i in range(2, NUM_MARKERS):
#     if ptsPix[i] != [None, None]:
#         pixsToConvert[i-2] = ptsPix[i]
# # positions = convertCoordinates(np.array(pixsToConvert, np.float64))
# positions = convertCoordinates(wcsObj, pixsToConvert)

# print("positions: ", positions)

#-------------------------Attempt At Converting w/ Martin's Code----------------------------------
# Use constant geographical location in MARK_POS to find geographical location of robot and each nonstationary marker 

# Calculate the degree per pixel (conversion factor) using pixel coordinates of Markers 1 and 2
# latDegPerPixel, lonDegPerPixel = calcDegPerPixel(ptsPix[0], ptsPix[1])
# # print("latDegPerPixel: ", latDegPerPixel)
# # print("lonDegPerPixel: ", lonDegPerPixel)
# for i in range(2, NUM_MARKERS):
#     #print("in 1st for loop iteration ", (i-2))
#     if ptsPix[i] != [None, None]:
#         #print("positions: ", positions)
#         positions[i-2] = convertCoordinates(latDegPerPixel, lonDegPerPixel, ptsPix[i], ptsPix[0])
#         #print("positions[", i, "-2]:", positions[i-2])
# #print("out of for loop")


# Takes frame from camera displays it with markers highlighted every 50 run (for testing)
# if frameNum%50==0:
#     frameFinal = frame.copy()
#     # hsv = cv.cvtColor(frameFinal, cv.COLOR_BGR2HSV)
#     # saturation = 50
#     # hsv[:,2,:] = cv.subtract(hsv[:,2,:], saturation)
#     # frameFinal = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

#     cv.aruco.drawDetectedMarkers(frameFinal, corners, ids)
#     cv.aruco.drawDetectedMarkers(frameFinal, rejectedImgPoints)
#     cv.imshow("final frame", frameFinal)
#     cv.waitKey(0)  # Wait for any key press to close the window

#     # Close all windows
#     cv.destroyAllWindows()

#     frameFinal = imutils.resize(frameFinal, height=1500)
#     cv.aruco.drawDetectedMarkers(frameFinal, corners, ids)
#     cv.aruco.drawDetectedMarkers(frameFinal, rejectedImgPoints)
#     cv.imshow("final frame", frameFinal)
#     cv.waitKey(0)  # Wait for any key press to close the window

#     # Close all windows
#     cv.destroyAllWindows()



# Check distance between written points and correct points
# def checkErorr(positions):
#     actual = [[50.92561868, 13.33166132], [50.92561868, 13.3317382], [50.92568494, 13.33170740],
#                 [50.92568242, 13.33166373], [50.92572391, 13.33168045]]

#     R = 6371

#     # Convert latitudes and longitudes to Cartesian coordinates so we can find Euclidean distance
#     for i in range(0, len(actual)):

#         # Actual
#         xA = R * (math.cos(actual[i][0])) * (math.cos(actual[i][1]))
#         yA = R * (math.cos(actual[i][0])) * (math.sin(actual[i][1]))

#         # Written
#         xW = R * (math.cos(positions[i+1][0])) * (math.cos(positions[i+1][1]))
#         yW = R * (math.cos(positions[i+1][0])) * (math.sin(positions[i+1][1]))

#         print("Distance Marker ", (i+3), ": ", math.dist([xA, yA], [xW, yW]))



#-------------------Original Conversion Process-------------------------------
# # Calculate latitude of robot and/or stationary markers 
# # y1=y value of pixel coordinates of Marker1
# # y2=y value of pixel coordinates of Marker2
# # lat1=latitude of Marker1
# # lat2=latitude of Marker2
# # yT=y value of pixel coordinates of the target object 
# def calcLat(y1, y2, lat1, lat2, yT):

#     scaleVert = (lat2 - lat1) / ( (y2 - y1))

#     # Delta between y values of the pixel coordinates between Marker1 and Robot
#     deltaY = y1 - yT

#     # What the delta btwn the latitude of Marker1 and the latitude of Robot will be
#     deltaLat = deltaY * scaleVert

#     # Using deltaLat and latitide of Marker1 we calculate the Robot's latitude
#     lat = lat1 + deltaLat

#     return lat


# # Calculate longitude of robot and/or stationary markers 
# # x1=x value of pixel coordinates of Marker1
# # x2=x value of pixel coordinates of Marker2
# # long1=longitude of Marker1
# # long2=longitude of Marker2
# # xT= x value of pixel coordinates of the target object
# def calcLong(x1, x2, long1, long2, xT):

#     scaleHoriz = (long2 - long1) / ( (x2 - x1))

#     # Delta between x values of the pixel coordinates between Marker1 and Robot
#     deltaX = x1 - xT

#     # What the delta btwn the longitude of Marker1 and the longitude of Robot will be
#     deltaLong = deltaX * scaleHoriz

#     # Using deltaLong and longitude of Marker1 we calculate the Robot's longitude
#     long = long1 + deltaLong

#     return long




#----------------------Attempt At Converting Using Astropy-----------------------
# # Function to create a wcs object
# # Note: I have no idea if I'm doing this right, especially parameters with *
# # Parameters: refPixel=pixel location of stationary marker, cdelt=difference between latitude and longitude of stationary markers*, 
# #   coordVal=latitude and longitude of the stationary marker whose pixel location is refPixel 
# def createWCS(refPixel, incrementAxis, coordVal):
#     w = wcs.WCS(naxis=2)
#     #w.wcs.naxis = 2
#     w.wcs.crpix = refPixel
#     w.wcs.cdelt = np.array(incrementAxis)
#     w.wcs.crval = coordVal
#     w.wcs.ctype = ["RA--AIR", "DEC--AIR"]

#     return w

# def convertCoordinates(wcsObj, pixCoord):
#     print("pixCoord: ", pixCoord)
#     print("wcsObj: ", wcsObj)
#     # # Example file containing cylindrical equal area projection
#     # filename = "1904-66_CEA.fits"

#     # # Load FITS hdulist
#     # hdulist = fits.open(filename)

#     # # Parse the WCS keywords in the primary HDU
#     # w = wcs.WCS(hdulist[0].header)

#     # w = wcs.WCS(naxis=2)

#     # w.wcs.crpix = [-234.75, 8.3393]
#     # w.wcs.cdelt = np.array([-0.066667, 0.066667])
#     # w.wcs.crval = [0, -90]
#     # w.wcs.ctype = ["RA--AIR", "DEC--AIR"]
#     # w.wcs.set_pv([(2, 1, 45.0)])

#     # Convert pixel coordinates to wordl coordinates
#     world = wcsObj.wcs_pix2world(pixCoord, 0)

#     print("World: ", world)

#     return world



#----------------------------------------Attempt At Converting w/ Martin's Code----------------------------------------------------
# Convert pixel locations to latitude/longitude
# # Parameters: degPerPixel=conversion factor at this frame for lat/lon, nsMarker=pixel coordinates of non-stationary marker, sMarker=pixel coordinates
# #   of stationary marker
# def convertCoordinates(latDegPerPixel, lonDegPerPixel, nsMarker, sMarker):
#     #print("in convertCoordinates")

#     # Get pixel distance between non-stationary marker and stationary marker 
#     pixelDistance = math.dist(nsMarker, sMarker)

#     # Convert this distance into degree latitude and degree longitude
#     lat = pixelDistance*latDegPerPixel
#     lon = pixelDistance*lonDegPerPixel

#     return [lat, lon]



# # Calculate the amount of degrees per pixel (conversion factor) in this frame
# # Parameters: pixelCord=nested list containing the pixel locations of Markers 1 and 2, lat=latitude coordinates of Markers 1 and 2, lon=longitude 
# #   coordinates of Markers 1 and 2     
# def calcDegPerPixel(pixelCoord1, pixelCoord2):

#     #print("in calcDegPerPixel")

#     # Calculate the Euclidean distance between the 2 given pixel coordinates (each formatted as a 2-item list)
#     pixelDistance = math.dist(pixelCoord1, pixelCoord2)

#     # Use Martin Plank's functions below to convert the latitude and longitude coordinates into meters
#     # Note: I don't know if I'm doing the parameters right here, may have to change depending on Martin's answer
#     latDistance = convert_lat_to_meter(MARKER_POS[0][0], MARKER_POS[1][0])
#     lonDistance = convert_lon_to_meter(MARKER_POS[0][1], MARKER_POS[1][1])

#     # Getting degree latitude and degree longitude per pixel
#     latPerPixel = pixelDistance / latDistance
#     lonPerPixel = pixelDistance / lonDistance

#     return latPerPixel, lonPerPixel



# #---------------------------------------Martin Plank's functions-------------------
# def convert_lat_to_meter(deg_lat, lat_mid):
#     """Convert Degree latitude to meters at the given lat_mid.
#         @arg deg_lat: The degree Latitude to convert.
#         @arg lat_mid: The latitude at which to do the conversion.
#     """
#     return deg_lat * get_m_per_deg_lat(lat_mid)


# def convert_lon_to_meter(deg_lon, lat_mid):
#     """Convert Degree longitude to meters at the given lat_mid.
#         @arg deg_lat: The degree longitude to convert.
#         @arg lat_mid: The latitude at which to do the conversion.
#     """
#     return deg_lon * get_m_per_deg_lon(lat_mid)


# def get_m_per_deg_lat(lat_mid):
#     """Determine the Meters in one degree latitude at the given lat_mid.
#         @arg lat_mid: The Latitude at which to do the conversion.
#     """
#     m_per_deg_lat = (
#         111132.954
#         - 559.822 * math.cos(math.radians(2.0 * lat_mid))
#         + 1.175 * math.cos(math.radians(4.0 * lat_mid))
#         - 0.0023 * math.cos(math.radians(6.0 * lat_mid)))
#     return abs(m_per_deg_lat)


# def get_m_per_deg_lon(lat_mid):
#     """Determine the Meters in one degree longitude at the given lat_mid.
#         @arg lat_mid: The Latitude at which to do the conversion.
#     """
#     m_per_deg_lon = (
#         111412.84 * math.cos(math.radians(lat_mid))
#         - 93.5 * math.cos(math.radians(3.0 * lat_mid))
#         + 0.118 * math.cos(math.radians(5.0 * lat_mid)))

#     return abs(m_per_deg_lon)


#----------------------------------------2nd Attempt w/ Martin's code------------------------------------------------
# Function to convert given pixel coordinates to Cartesian x/y coordinates 
# # Process from this link: 

# def pixelToXY(pixX, pixY):

#     # Calculate scale factors --> fit all pixels into our x/y plane
#     scaleX = X_MAX / WIDTH
#     scaleY = Y_MAX / HEIGHT

#     # Convert pixels
#     x = scaleX * pixX
#     y = scaleY * (HEIGHT-pixY)

#     return [x, y]



# # Function to convert given latitude/longitude to x/y coordinates in meters, to be used with GNSS coordinates to compare to what we calculate
# def geoToXY(coord):

#     if coord != []:
#         lat = coord[0]
#         lon = coord[1]
        
#     #     R = 6371
#     #     x = R * math.cos(math.radians(lat)) * math.cos(math.radians(lon))
#     #     y = R * math.cos(math.radians(lat)) * math.sin(math.radians(lon))

#     #     return [x, y]
#         return [convert_lat_to_meter(lat, lat), convert_lon_to_meter(lon, lat)]
#     else:
#         return ""


# #--------------------------------------------Martin Plank's functions-------------------------------------------------------------
# def get_m_per_deg_lat(lat_mid):
#     """Determine the Meters in one degree latitude at the given lat_mid.
#         @arg lat_mid: The Latitude at which to do the conversion.
#     """
#     m_per_deg_lat = (
#         111132.954
#         - 559.822 * math.cos(math.radians(2.0 * lat_mid))
#         + 1.175 * math.cos(math.radians(4.0 * lat_mid))
#         - 0.0023 * math.cos(math.radians(6.0 * lat_mid)))
#     return abs(m_per_deg_lat)


# def get_m_per_deg_lon(lat_mid):
#     """Determine the Meters in one degree longitude at the given lat_mid.
#         @arg lat_mid: The Latitude at which to do the conversion.
#     """
#     m_per_deg_lon = (
#         111412.84 * math.cos(math.radians(lat_mid))
#         - 93.5 * math.cos(math.radians(3.0 * lat_mid))
#         + 0.118 * math.cos(math.radians(5.0 * lat_mid)))

#     return abs(m_per_deg_lon)


# def convert_lat_to_meter(deg_lat, lat_mid):
#     """Convert Degree latitude to meters at the given lat_mid.
#         @arg deg_lat: The degree Latitude to convert.
#         @arg lat_mid: The latitude at which to do the conversion.
#     """
#     return deg_lat * get_m_per_deg_lat(lat_mid)


# def convert_lon_to_meter(deg_lon, lat_mid):
#     """Convert Degree longitude to meters at the given lat_mid.
#         @arg deg_lat: The degree longitude to convert.
#         @arg lat_mid: The latitude at which to do the conversion.
#     """
#     return deg_lon * get_m_per_deg_lon(lat_mid)



# # TODO: make this more accurate, right now I'm hard coding it to always have the same points 
# def getGNSS(markerNumber):

#     # Testing 
#     #print("markerNumber: ", markerNumber)

#     GNSS = [[], [50.92561868, 13.33166132], [50.92573552, 13.3317382], [50.92568494, 13.33170740], [50.92568242, 13.33166373], [50.92572391, 13.33168045]]

#     return GNSS[markerNumber-2]



# Code to display a frame with the detected markers outlined and labeled with their id, rejected markers also outlined
# Can use for testing:
    # # Takes frame from camera displays it with markers highlighted
    # cv.imshow('final frame', frame)
    # frameFinal = frame.copy()
    # cv.aruco.drawDetectedMarkers(frameFinal, corners, ids, rejected imgPoints)
    # cv.imshow("final frame", frameFinal)
    # cv.waitKey(0)  # Wait for any key press to close the window

    # # Close all windows
    # cv.destroyAllWindows()



# Probs don't need but don't want to delete yet just in case 
# LEFT OFF: got the example to work, but I don't know how to get that to work with our images, want to see if theres a way to check that the
    # images are even distorted, because if they're not then you don't have to do all of this, but we should figure it out in case a random
    # frame does end up being distorted b/c we don't want that to just break our code 

# Function to undistort the image we have at each frame before finding locations, taken from OpenCV documentation
# def undistort():
#     # print("in undistort")

#     # 1. Set-up:

#     # Termination criteria
#     criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

#     # Prepare object points like (0,0,0), (1,0,0), (2,0,0)...,(6,5,0)
#     objp = np.zeros((6*7,3), np.float32)
#     objp[:,:2] = np.mgrid[0:7, 0:6].T.reshape(-1,2)

#     # print("objp: ", objp)

#     # Arrays to store object points and image points from all the images 
#     objPoints = [] # 3D points in real world space
#     imgPoints = [] # 2D points in image plane
 
#     images = glob.glob('*.png')

#     for image in images:
#         img = cv.imread(image)
#         gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

#         # cv.imshow('img', gray)
#         # cv.waitKey(0)
#         # cv.destroyAllWindows

#         # Find the chess board corners
#         ret, corners = cv.findChessboardCorners(gray, (7,6), None)
#         # print("past ret")

#         # If found, add object points and image points (after refining them)
#         if ret == True:
#             # print("true")

#             objPoints.append(objp)
#         #else:
#             # print("false")
        
#         corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
#         imgPoints.append(corners2)

        # Draw and display the corners
    #     cv.drawChessboardCorners(img, (7,6), corners2, ret)
    #     cv.imshow('img', img)
    #     cv.waitKey(500)

    # cv.destroyAllWindows


    # 2. Calibration
    # mtx = camera matrix, dist = distortion coefficients, rvecs = rotation vectors, tvecs = translatiob vectors
    # ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objPoints, imgPoints, gray.shape[::-1], None, None)


    # # 3. Undistortion
    # img = cv.imread("left12.png")
    # h, w = img.shape[:2]
    # newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

    # # mapx, mapy = cv.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5)
    # # dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)

    # # Undistort
    # dst = cv.undistort(img, mtx, dist, None, newcameramtx)

    # # Crop image
    # x, y, w, h = roi
    # dst = dst[y:y+h, x:x+w]
    # cv.imwrite('calibresult.jpg', dst)