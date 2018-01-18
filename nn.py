import numpy as np
import h5py
import sys
import sklearn
from sklearn.neighbors import KDTree
import pdb
import networkx as nx
from networkx.algorithms.components.connected import connected_components
import matplotlib.pyplot as plt
import random
from itertools import cycle
import time
import hdbscan
from collections import defaultdict

groundLevel = -1.64
distance = 0.5
pointLimit = 128

def readFile(fileName):
	pointCloud = []
	inputFile = open(fileName,"r")
	for line in inputFile:
		try:
			floats=[float(x) for x in line.split(" ")]
			if floats[2]>groundLevel and floats[0] < 40 and floats[0] > -40 and floats[1] < 40 and floats[1] > -40:
				pointCloud.append(floats[0:3])
		except Exception as e:
			pass
	return pointCloud


def getNeighbors(tree,pointCloud):
	clusters = []
	neighborhoods = tree.query_radius(pointCloud,r=distance)
	for array in neighborhoods:
		clusters.append(array.tolist())
	return clusters

def to_graph(neighbors_clusters):
	G=nx.Graph()
	for neighbors in neighbors_clusters:
		G.add_nodes_from(neighbors)
		G.add_edges_from(to_edges(neighbors))
	return G
	
def to_edges(neighbors):
	it = iter(neighbors)
	last = next(it)
	for current in it:
		yield last,current
		last = current

def formatData(pointClusters,pointCloud):
	formatedClusters = []
	labels = []
	for points in pointClusters:
		if len(points) >= 128:
			formatedClusters.append(resize(points,pointCloud))
			labels.append(0)
		elif 128 > len(points) and len(points) > 20:
			formatedClusters.append(resize(points,pointCloud))
			labels.append(0)
		else:
			pass	
	return (formatedClusters, labels)

def resize(model,pointCloud):
	if len(model) > 128:
		for x in range(int(len(model)/pointLimit)):
			newSet = random.sample(model,pointLimit)
	else:
		newSet = model
	formatedCloud = []
	for point in newSet:
		formatedCloud.append(pointCloud[point])

	return formatedCloud

def NN_segmentation(inputFile):
	pointCloud = readFile(inputFile)
	current_time = time.time()
	tree = KDTree(pointCloud)
	print("Tree build time:%f",(time.time()-current_time))
	current_time = time.time()
	Graph=to_graph(getNeighbors(tree,pointCloud))
	print("Graph build time:%f",(time.time()-current_time))
	subGraph = list(nx.connected_components(Graph))
	return formatData(subGraph,pointCloud)

def importData(inputFile,seg_alg):
	formatedList = []
	orderList = []
	allClusters = []
	objLen = []
	if seg_alg =='NN':
		data,labels = NN_segmentation(inputFile)
	elif seg_alg == 'HDBSCAN':
		resizedData,labels, pointCloud, allSegmentedPoints,objLength = HDBSCAN(inputFile) 
		for cluster in resizedData:
			if len(cluster) == 128:
				X = (max([point[0] for point in cluster]) + min([point[0] for point in cluster])) / 2
				Y = (max([point[1] for point in cluster]) + min([point[1] for point in cluster])) /2
				Z = (max([point[2] for point in cluster]) + min([point[2] for point in cluster])) / 2
				centeredCoordinates = [(point[0]-X,point[1]-Y,point[2]-Z) for point in cluster]
				formatedList.append(centeredCoordinates)
				orderList.append(resizedData.index(cluster))
			allClusters += cluster
	else:
		print('No algorithm choosen') 
	return np.array(formatedList), np.array(orderList), np.array(pointCloud), np.array(objLength), np.array(allSegmentedPoints)

def HDBSCAN(inputFile):
	dictonary = defaultdict(list)
	pointCloud = readFile(inputFile)
	array = np.asarray(pointCloud)
	clusterer = hdbscan.HDBSCAN(min_cluster_size=200,approx_min_span_tree=True,leaf_size=50,gen_min_span_tree=True)
	clusterer.fit(array)
	allSegmentedClusters = []
	allSegmentedPoints = []
	objLength = []
	for i in range(len(clusterer.labels_)):
		dictonary[clusterer.labels_[i]].append(pointCloud[i])
	for index in range(len(dictonary)):	
		points = []
		for node in dictonary[index]:
			points.append(node)
		allSegmentedPoints = allSegmentedPoints + points
		allSegmentedClusters.append(points)
		objLength.append(len(points))
	resizedData = []
	labels = []
	for model in allSegmentedClusters:
		if len(model) > 128:
			resizedData.append(random.sample(model,pointLimit))
			labels.append(0)
		else:
			resizedData.append(model)
			labels.append(0)
	return resizedData,labels,pointCloud, allSegmentedPoints, objLength

	#fileName = 0
#for cluster in subGraph:
#	if len(cluster) > 30:	
#		list(cluster)
#		fileName += 1
#		output=open((str(fileName)+".pcd"),"w")
#		output.write("# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\nWIDTH "+str(len(cluster)) +"\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS "+str(len(cluster))+"\nDATA ascii\n")
#		for node in cluster:
#			output.write(str(pointCloud[node][0])+" "+str(pointCloud[node][1])+" "+str(pointCloud[node][2])+"\n")
#		output.close()



def printInfo(fileName):
	start_time = time.time()
	data,labels = NN_segmentation(fileName)
	print("Total time for retrieve:%f",(time.time()-start_time))
	a=0
	for x in data:
		a= a+ len(x)		
	print(a)
	print(len(data))

def convertToNumpy2D(data):
	#array = np.array([[0,0,0]])
	array = []
	objLen = []
	for x in range(len(data)):
		objLen.append(len(data[x]))
		array.append(data[x])
		#temp=np.asarray(data[x],dtype=np.float32)
		#array = np.concatenate((array,temp),axis=0)
	return np.array(array), objLen


# Create the scatter plot
def colorMaping(predLabels,data):
	
	colorMap = np.array([[1,1,1]])
	n=0	
	for obj in data:
		if len(obj) == 128:
			if predLabels[n] == 1:
				color = np.array([[1, 1, 0]] * len(obj))
				colorMap = np.concatenate((colorMap,color),axis=0)
			elif predLabels[n] == 2:
				color = np.array([[0, 1, 1]] * len(obj))
				colorMap = np.concatenate((colorMap,color),axis=0)
			elif predLabels[n] == 3:
				color = np.array([[1, 0, 1]] * len(obj))
				colorMap = np.concatenate((colorMap,color),axis=0)
			else:
				color = np.array([[1, 1, 1]] * len(obj))
				colorMap = np.concatenate((colorMap,color),axis=0)
			n += 1
		else:
			color = np.array([[1, 1, 1]] * len(obj))
			colorMap = np.concatenate((colorMap,color),axis=0)
	return colorMap

#data1,label1 = NN_segmentation(sys.argv[1])
#data2,label2 = HDBSCAN(sys.argv[1])
