
# =====================================================

import math, json
from scipy import integrate

import numpy as np
import matplotlib.pyplot as plt

import geoutils
import geopandas as gpd
from geopandas import GeoSeries
from shapely.geometry import LineString

def piecewise_function(X,Y,x):
	if x < X[0] or x > X[len(X)-1]:
		print('ILLEGAL_ARGUMENT')
		return 0;
	i=1
	while i <= len(X)-1:
		if(x<=X[i]):
			y=(Y[i]-Y[i-1])*(x-X[i])/(X[i]-X[i-1])+Y[i]
			return y
		else:
			i=i+1
	return 0

def adjust_StartAndDirection(A):
	if not geoutils.is_CloseWise(A):
		A.reverse()
	OBB,_=geoutils.mininumAreaRectangle(A)
	start_index=OBB.pointTouchRectWithMaxX(A)
	max_X,max_index=A[0][0],0
	for j in range(1,len(A)):
		if A[j][0] > max_X:
			max_X,max_index=A[j][0],j
	distance=OBB.distanceOfPointFromRect(A[max_index])
	if (distance<0.05):
		start_index=max_index
	B=[]
	for i in range(max_index, max_index + len(A)):
		if i == len(A) - 1:
			continue
		B.append(A[i % len(A)])
	B.append(B[0])
	return B

def fourier_Series(X,S,series_size):
	coord_size=len(X)
	if (coord_size <= 2 or
		coord_size != len(S)):
		print('ILLEGAL_ARGUMENT')
		return [],[],[],[]
	period_size=S[coord_size-1]
	XA,XB=[],[]
	for i in range(0,series_size):
		XAi,XBi=0,0
		for j in range(0, coord_size-1):
			fax=lambda s:(X[j]+(X[j+1]-X[j])*(s-S[j])/(S[j+1]-S[j]))*math.cos(i*2*math.pi*s/period_size)*2/period_size
			if S[j] == S[j+1]:
				print(integrate.quad(fax, S[j], S[j+1]))
				exit()
			XAi+=integrate.quad(fax, S[j], S[j+1])[0]
			fbx=lambda s:(X[j]+(X[j+1]-X[j])*(s-S[j])/(S[j+1]-S[j]))*math.sin(i*2*math.pi*s/period_size)*2/period_size
			XBi+=integrate.quad(fbx, S[j], S[j+1])[0]
		XA.append(XAi)
		XB.append(XBi)
	return XA,XB

def constructObjectByFourierSeries(S,series_size,period_size=2*math.pi):
	arc_length=numpy.linspace(0,period_size,40*5-1)
	constructedCoords=[]
	for i in arc_length:
		coord_x,coord_y=S[0]/2,S[2*series_size]/2
		for j in range(1, series_size):
			coord_x+=S[j]*math.cos(j*i*2*math.pi/period_size)+S[series_size+j]*math.sin(j*i*2*math.pi/period_size)
			coord_y+=S[2*series_size+j]*math.cos(j*i*2*math.pi/period_size)+S[3*series_size+j]*math.sin(j*i*2*math.pi/period_size)
		constructedCoords.append([coord_x,coord_y])
	return constructedCoords

def convertGeoObjectToArcLengthFunction(A,is_adjusted=False,is_normalized=False):
	X,Y,S=[],[],[]
	coord_size=len(A)
	if (coord_size < 2):
		print('ILLEGAL_ARGUMENT')
		return [],[],[];
	if (A[0][0]!=A[len(A)-1][0] or A[0][1]!=A[len(A)-1][1]):
		A.append(A[0])
		coord_size=coord_size+1
	if (is_adjusted):
		A=adjust_StartAndDirection(A)
	for i in range(0,coord_size):
		X.append(A[i][0])
		Y.append(A[i][1])
		if (i==0):
			S.append(0)
		else:
			S.append(S[i-1]+
				math.sqrt(
					pow(A[i][0]-A[i-1][0],2)+
					pow(A[i][1]-A[i-1][1],2))
				)
	if (is_normalized):
		S=[i*2*math.pi/S[len(S)-1] for i in S]
	return X,Y,S

def do_FFT(A,series_size,is_adjusted=True,is_normalized=False):
	X, Y, S = convertGeoObjectToArcLengthFunction(A,is_adjusted=is_adjusted,is_normalized=is_normalized)
	coord_size = len(X)
	if(coord_size == 0):
		print('ILLEGAL_ARGUMENT')
		return [], [], [], []
	XA, XB = fourier_Series(X, S, series_size)
	YA, YB = fourier_Series(Y, S, series_size)
	return XA, XB, YA, YB

def do_Matching(A,B,series_size=4):
	[[ACX,ACY],Aarea,Aperi]=geoutils.get_basic_parametries_of_Poly(A)
	[[BCX,BCY],Barea,Bperi]=geoutils.get_basic_parametries_of_Poly(B)

	uniform_A_coords = [[(j[0]-ACX)*2*math.pi/Aperi, (j[1]-ACY)*2*math.pi/Aperi] for j in A]
	uniform_B_coords = [[(j[0]-BCX)*2*math.pi/Bperi, (j[1]-BCY)*2*math.pi/Bperi] for j in B]
	
	AXA,AXB,AYA,AYB  = do_FFT(uniform_A_coords,series_size,is_adjusted=True)
	composite_A_vector = AXA+AXB+AYA+AYB
	composite_A_vector = np.array(composite_A_vector)
	
	BXA,BXB,BYA,BYB  = do_FFT(uniform_B_coords,series_size,is_adjusted=True)
	composite_B_vector = BXA+BXB+BYA+BYB
	composite_B_vector = np.array(composite_B_vector)

	dis = 0
	for i in range(len(composite_A_vector)):
		dis += pow((composite_A_vector[i]-composite_B_vector[i]), 2)
	return dis # get_distance(composite_A_vector,composite_B_vector,1)

def main(argv=None):
	coords = []
	file=open('./data/FF_test.json','r',encoding='utf-8')
	data=json.load(file)
	feature_size=len(data['features'])
	for i in range(0,feature_size):
		ID = data['features'][i]['attributes']['type']        # nCohesion
		geome_dict = data['features'][i]['geometry']          # Get the geometry objects.
		geo_path   = geome_dict['rings']
		coord = []
		for j in range(0,len(geo_path)):
			# print(len(geo_path[j]))
			for k in range(0,len(geo_path[j])):
				coord.append([geo_path[j][k][0], geo_path[j][k][1]])
			break
		coords.append([coord, ID])

	for i in range(0, len(coords)):
		for j in range(i+1, len(coords)):
			print('matching_degres of {0} and {1} is :   {2}'.format(coords[i][1], coords[j][1], do_Matching(coords[i][0], coords[j][0])))
			# print('')
	exit()

def main_(argv=None):
	Xs, Ys, Ss = [], [], []
	file=open('data/alldata/FFT_bu_test1.json','r',encoding='utf-8')
	data=json.load(file)
	feature_size=len(data['features'])
	print('feature_size: ', feature_size)

	for i in range(0,feature_size):
		X, Y, S = [], [], []
		attri_dict=data['features'][i]['attributes']        # Get the attributes.
		geome_dict=data['features'][i]['geometry']          # Get the geometry objects.
		geo_path=geome_dict['paths']
		for j in range(0, len(geo_path)):
			print(len(geo_path[j]))
			for k in range(0,len(geo_path[j])):
				X.append(geo_path[j][k][0])
				Y.append(geo_path[j][k][1])
				if k==0:
					S.append(0)
				else:
					S.append(S[k-1]+
						math.sqrt(
							pow(geo_path[j][k][0]-geo_path[j][k-1][0],2)+
							pow(geo_path[j][k][1]-geo_path[j][k-1][1],2)
							)
						)
			print(len(X),len(Y),len(S))
			Xs.append(X)
			Ys.append(Y)
			Ss.append(S)
			break
	
	out_shp = []
	for i in range(0, feature_size):
		X, Y, S = Xs[i], Ys[i], Ss[i]
		series_size = 16        # 8 16
		coord_size  = len(X)
		period_size = S[coord_size-1]
		print(coord_size, period_size)
		XA,XB = fourier_Series(X,S,series_size)
		YA,YB = fourier_Series(Y,S,series_size)

		arc_length = np.linspace(0,period_size,100*5-1)
		origin_coords_x,origin_coords_y,transform_coords_x,transform_coords_y = [],[],[],[]
		for i in arc_length:
			origin_coord_x, transfor_coord_x = piecewise_function(S,X,i), XA[0]/2
			origin_coord_y, transfor_coord_y = piecewise_function(S,Y,i), YA[0]/2
			for j in range(1, series_size):
				transfor_coord_x += XA[j]*math.cos(j*i*2*math.pi/period_size) + XB[j]*math.sin(j*i*2*math.pi/period_size)
				transfor_coord_y += YA[j]*math.cos(j*i*2*math.pi/period_size) + YB[j]*math.sin(j*i*2*math.pi/period_size)
			origin_coords_x.append(origin_coord_x)
			origin_coords_y.append(origin_coord_y)
			transform_coords_x.append(transfor_coord_x)
			transform_coords_y.append(transfor_coord_y)
		#'''
		plt.subplot(311)
		plt.plot(arc_length,origin_coords_x,color='blue',label='Origin coordinate Y')
		plt.plot(arc_length,transform_coords_x,color='red',label='Fourier series approximation')
		plt.subplot(312)
		plt.plot(arc_length,origin_coords_y,color='blue',label='Origin coordinate Y')
		plt.plot(arc_length,transform_coords_y,color='red',label='Fourier series approximation')
		plt.subplot(313)
		#'''
		
		plt.plot(X, Y, color='blue',marker='o')
		plt.plot(transform_coords_x, transform_coords_y, color='red',marker='o')
		plt.yticks(Y)
		plt.legend()
		plt.gca().set_aspect(1)
		
		line_t = LineString(list(zip(transform_coords_x, transform_coords_y)))
		out_shp.append(line_t)

		#break

	plt.show()

	g = GeoSeries(out_shp)
	g.to_file('data/alldata/fourier_bu_16.shp')
	plt.title('fourier_simplification')

if __name__ == '__main__':
	main_()

