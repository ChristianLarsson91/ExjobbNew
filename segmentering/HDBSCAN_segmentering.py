import numpy as np
import lib
import hdbscan
import sys
import pdb
from collections import defaultdict

dictonary = defaultdict(list)

pointCloud = lib.readFile(sys.argv[1])
array = np.asarray(pointCloud)

clusterer = hdbscan.HDBSCAN(min_cluster_size=128)
clusterer.fit(array)
for i in range(len(clusterer.labels_)):
	dictonary[clusterer.labels_[i]].append(pointCloud[i])

fileName = 0
for index in range(len(dictonary)):
	pdb.set_trace()
	if len(dictonary[index]) > 30:
		fileName += 1
		output=open((str(fileName)+".pcd"),"w")
		output.write("# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\nWIDTH "+str(len(dictonary[index])) +"\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS "+str(len(dictonary[index]))+"\nDATA ascii\n")
		for node in dictonary[index]:
			output.write(str(node[0])+" "+str(node[1])+" "+str(node[2])+"\n")
		output.close()
