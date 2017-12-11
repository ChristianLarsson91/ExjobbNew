import numpy as np
from vispy import scene, visuals, app, gloo
import matplotlib.pyplot as plt
import sys
#import nn


pointCloud = []
inputFile = open("set2.pcd","r")
for line in inputFile:
	try:
		floats=[float(x) for x in line.split(" ")]
		#if floats[2]>groundLevel and floats[0] < 40 and floats[0] > -40 and floats[1] < 40 and floats[1] > -40 :
		#print(floats)
		pointCloud.append(floats)
	except Exception as e:
		pass

data = np.asarray(pointCloud,dtype=np.float32)

canvas = scene.SceneCanvas(keys='interactive', show=True)
view = canvas.central_widget.add_view()

color1 = np.array([[1, 0, 0.4]] * 123398)
color2 = np.array([[0.4, 1, 0]] * 128)
color3 = np.array([[0, 0.4, 1]] * 128)



# Create the scatter plot
scatter = scene.visuals.Markers()
print(data.shape)
scatter.set_data(data[:,:2], face_color=color1,size=0.5)
view.add(scatter)
view.camera = scene.PanZoomCamera(aspect=1)
view.camera.set_range()
app.run()

#plt.cm.jet(data[:,2]