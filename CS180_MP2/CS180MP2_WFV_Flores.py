import numpy as np
import cv2
from sklearn.cluster import KMeans

choosy = int(input("1 - 2(+1 for black area) random centroids\n2 - Manually selected centroids\n3 - 5 Centroids\n4 - HSV\n5 - Cielab\n"))

if choosy == 1:
    #FILARIODIEA

    #Load the images for training
    img = cv2.imread('filarioidea/filaria.jpg')
    img2 = cv2.imread('filarioidea/filaria2.jpg')
    trainer = np.vstack((img, img2))

    #Save image dimensions
    dim1 = img.shape[0]
    dim2 = img.shape[1]

    #Reshape the image from 3D to 2D
    trainer = np.reshape(trainer, (-1,3))

    #Perform KMeans (extra cluster for black portions)
    kmeans = KMeans(n_clusters=3).fit(trainer)

    #Upscale the labels
    labs = (kmeans.labels_/2)*255

    #Reshape the upscaled labels from 1D to 2D
    labs = np.reshape(labs, (dim1*2,dim2))

    #Predicting part
    pred1 = cv2.imread('filarioidea/filaria.jpg')
    pred2 = cv2.imread('filarioidea/filaria2.jpg')
    pred3 = cv2.imread('filarioidea/filaria3.jpg')
    pred4 = cv2.imread('filarioidea/filaria4.jpg')
    pred5 = cv2.imread('filarioidea/filaria5.jpg')
    pred6 = cv2.imread('filarioidea/filaria6.jpg')
    pred7 = cv2.imread('filarioidea/filaria7.jpg')
    pred8 = cv2.imread('filarioidea/filaria8.jpg')
    pred9 = cv2.imread('filarioidea/filaria9.jpg')
    pred10 = cv2.imread('filarioidea/filaria10.jpg')

    dim1_1 = pred1.shape[0]
    dim1_2 = pred1.shape[1]
    dim2_1 = pred2.shape[0]
    dim2_2 = pred2.shape[1]
    dim3_1 = pred3.shape[0]
    dim3_2 = pred3.shape[1]
    dim4_1 = pred4.shape[0]
    dim4_2 = pred4.shape[1]
    dim5_1 = pred5.shape[0]
    dim5_2 = pred5.shape[1]
    dim6_1 = pred6.shape[0]
    dim6_2 = pred6.shape[1]
    dim7_1 = pred7.shape[0]
    dim7_2 = pred7.shape[1]
    dim8_1 = pred8.shape[0]
    dim8_2 = pred8.shape[1]
    dim9_1 = pred9.shape[0]
    dim9_2 = pred9.shape[1]
    dim10_1 = pred10.shape[0]
    dim10_2 = pred10.shape[1]

    pred1 = np.reshape(pred1, (-1,3))
    pred2 = np.reshape(pred2, (-1,3))
    pred3 = np.reshape(pred3, (-1,3))
    pred4 = np.reshape(pred4, (-1,3))
    pred5 = np.reshape(pred5, (-1,3))
    pred6 = np.reshape(pred6, (-1,3))
    pred7 = np.reshape(pred7, (-1,3))
    pred8 = np.reshape(pred8, (-1,3))
    pred9 = np.reshape(pred9, (-1,3))
    pred10 = np.reshape(pred10, (-1,3))

    output = kmeans.predict(pred1)
    output = (output/2)*255
    output = np.reshape(output, (dim1_1, dim1_2))
    cv2.imwrite('filaria_pred_1_1.png', output)

    output = kmeans.predict(pred2)
    output = (output/2)*255
    output = np.reshape(output, (dim2_1, dim2_2))
    cv2.imwrite('filaria_pred_1_2.png', output)

    output = kmeans.predict(pred3)
    output = (output/2)*255
    output = np.reshape(output, (dim3_1, dim3_2))
    cv2.imwrite('filaria_pred_1_3.png', output)

    output = kmeans.predict(pred4)
    output = (output/2)*255
    output = np.reshape(output, (dim4_1, dim4_2))
    cv2.imwrite('filaria_pred_1_4.png', output)

    output = kmeans.predict(pred5)
    output = (output/2)*255
    output = np.reshape(output, (dim5_1, dim5_2))
    cv2.imwrite('filaria_pred_1_5.png', output)

    output = kmeans.predict(pred6)
    output = (output/2)*255
    output = np.reshape(output, (dim6_1, dim6_2))
    cv2.imwrite('filaria_pred_1_6.png', output)

    output = kmeans.predict(pred7)
    output = (output/2)*255
    output = np.reshape(output, (dim7_1, dim7_2))
    cv2.imwrite('filaria_pred_1_7.png', output)

    output = kmeans.predict(pred8)
    output = (output/2)*255
    output = np.reshape(output, (dim8_1, dim8_2))
    cv2.imwrite('filaria_pred_1_8.png', output)

    output = kmeans.predict(pred9)
    output = (output/2)*255
    output = np.reshape(output, (dim9_1, dim9_2))
    cv2.imwrite('filaria_pred_1_9.png', output)

    output = kmeans.predict(pred10)
    output = (output/2)*255
    output = np.reshape(output, (dim10_1, dim10_2))
    cv2.imwrite('filaria_pred_1_10.png', output)

    #PLASMODIUM

    #Load the images for training
    img = cv2.imread('plasmodium/1c.JPG')
    img2 = cv2.imread('plasmodium/6c.JPG')
    trainer = np.vstack((img, img2))

    #Save image dimensions
    dim1 = img.shape[0]
    dim2 = img.shape[1]

    #Reshape the image from 3D to 2D
    trainer = np.reshape(trainer, (-1,3))

    #Perform KMeans (no extra cluster for black portions)
    kmeans = KMeans(n_clusters=2).fit(trainer)

    #Upscale the labels
    labs = (kmeans.labels_)*255

    #Reshape the upscaled labels from 1D to 2D
    labs = np.reshape(labs, (dim1*2,dim2))

    #Predicting part
    pred1 = cv2.imread('plasmodium/1c.JPG')
    pred2 = cv2.imread('plasmodium/3c.JPG')
    pred3 = cv2.imread('plasmodium/6c.JPG')
    pred4 = cv2.imread('plasmodium/7c.JPG')
    pred5 = cv2.imread('plasmodium/11c.JPG')
    pred6 = cv2.imread('plasmodium/19c.JPG')
    pred7 = cv2.imread('plasmodium/55c.JPG')
    pred8 = cv2.imread('plasmodium/79c.JPG')
    pred9 = cv2.imread('plasmodium/94c.JPG')
    pred10 = cv2.imread('plasmodium/105c.JPG')

    dim1_1 = pred1.shape[0]
    dim1_2 = pred1.shape[1]
    dim2_1 = pred2.shape[0]
    dim2_2 = pred2.shape[1]
    dim3_1 = pred3.shape[0]
    dim3_2 = pred3.shape[1]
    dim4_1 = pred4.shape[0]
    dim4_2 = pred4.shape[1]
    dim5_1 = pred5.shape[0]
    dim5_2 = pred5.shape[1]
    dim6_1 = pred6.shape[0]
    dim6_2 = pred6.shape[1]
    dim7_1 = pred7.shape[0]
    dim7_2 = pred7.shape[1]
    dim8_1 = pred8.shape[0]
    dim8_2 = pred8.shape[1]
    dim9_1 = pred9.shape[0]
    dim9_2 = pred9.shape[1]
    dim10_1 = pred10.shape[0]
    dim10_2 = pred10.shape[1]

    pred1 = np.reshape(pred1, (-1,3))
    pred2 = np.reshape(pred2, (-1,3))
    pred3 = np.reshape(pred3, (-1,3))
    pred4 = np.reshape(pred4, (-1,3))
    pred5 = np.reshape(pred5, (-1,3))
    pred6 = np.reshape(pred6, (-1,3))
    pred7 = np.reshape(pred7, (-1,3))
    pred8 = np.reshape(pred8, (-1,3))
    pred9 = np.reshape(pred9, (-1,3))
    pred10 = np.reshape(pred10, (-1,3))

    output = kmeans.predict(pred1)
    output = (output)*255
    output = np.reshape(output, (dim1_1, dim1_2))
    cv2.imwrite('plasmodium_pred_1_1.png', output)

    output = kmeans.predict(pred2)
    output = (output)*255
    output = np.reshape(output, (dim2_1, dim2_2))
    cv2.imwrite('plasmodium_pred_1_2.png', output)

    output = kmeans.predict(pred3)
    output = (output)*255
    output = np.reshape(output, (dim3_1, dim3_2))
    cv2.imwrite('plasmodium_pred_1_3.png', output)

    output = kmeans.predict(pred4)
    output = (output)*255
    output = np.reshape(output, (dim4_1, dim4_2))
    cv2.imwrite('plasmodium_pred_1_4.png', output)

    output = kmeans.predict(pred5)
    output = (output)*255
    output = np.reshape(output, (dim5_1, dim5_2))
    cv2.imwrite('plasmodium_pred_1_5.png', output)

    output = kmeans.predict(pred6)
    output = (output)*255
    output = np.reshape(output, (dim6_1, dim6_2))
    cv2.imwrite('plasmodium_pred_1_6.png', output)

    output = kmeans.predict(pred7)
    output = (output)*255
    output = np.reshape(output, (dim7_1, dim7_2))
    cv2.imwrite('plasmodium_pred_1_7.png', output)

    output = kmeans.predict(pred8)
    output = (output)*255
    output = np.reshape(output, (dim8_1, dim8_2))
    cv2.imwrite('plasmodium_pred_1_8.png', output)

    output = kmeans.predict(pred9)
    output = (output)*255
    output = np.reshape(output, (dim9_1, dim9_2))
    cv2.imwrite('plasmodium_pred_1_9.png', output)

    output = kmeans.predict(pred10)
    output = (output)*255
    output = np.reshape(output, (dim10_1, dim10_2))
    cv2.imwrite('plasmodium_pred_1_10.png', output)

    #SCHISTOSOMA

    #Load the images for training
    img = cv2.imread('schistosoma/schistosoma.jpg')
    img2 = cv2.imread('schistosoma/schistosoma2.jpg')
    trainer = np.vstack((img, img2))

    #Save image dimensions
    dim1 = img.shape[0]
    dim2 = img.shape[1]

    #Reshape the image from 3D to 2D
    trainer = np.reshape(trainer, (-1,3))

    #Perform KMeans (extra cluster for black portions)
    kmeans = KMeans(n_clusters=3).fit(trainer)

    #Upscale the labels
    labs = (kmeans.labels_/2)*255

    #Reshape the upscaled labels from 1D to 2D
    labs = np.reshape(labs, (dim1*2,dim2))

    #Predicting part
    pred1 = cv2.imread('schistosoma/schistosoma.jpg')
    pred2 = cv2.imread('schistosoma/schistosoma2.jpg')
    pred3 = cv2.imread('schistosoma/schistosoma3.jpg')
    pred4 = cv2.imread('schistosoma/schistosoma4.jpg')
    pred5 = cv2.imread('schistosoma/schistosoma5.jpg')
    pred6 = cv2.imread('schistosoma/schistosoma6.jpg')
    pred7 = cv2.imread('schistosoma/schistosoma7.jpg')
    pred8 = cv2.imread('schistosoma/schistosoma8.jpg')
    pred9 = cv2.imread('schistosoma/schistosoma9.jpg')
    pred10 = cv2.imread('schistosoma/schistosoma10.jpg')

    dim1_1 = pred1.shape[0]
    dim1_2 = pred1.shape[1]
    dim2_1 = pred2.shape[0]
    dim2_2 = pred2.shape[1]
    dim3_1 = pred3.shape[0]
    dim3_2 = pred3.shape[1]
    dim4_1 = pred4.shape[0]
    dim4_2 = pred4.shape[1]
    dim5_1 = pred5.shape[0]
    dim5_2 = pred5.shape[1]
    dim6_1 = pred6.shape[0]
    dim6_2 = pred6.shape[1]
    dim7_1 = pred7.shape[0]
    dim7_2 = pred7.shape[1]
    dim8_1 = pred8.shape[0]
    dim8_2 = pred8.shape[1]
    dim9_1 = pred9.shape[0]
    dim9_2 = pred9.shape[1]
    dim10_1 = pred10.shape[0]
    dim10_2 = pred10.shape[1]

    pred1 = np.reshape(pred1, (-1,3))
    pred2 = np.reshape(pred2, (-1,3))
    pred3 = np.reshape(pred3, (-1,3))
    pred4 = np.reshape(pred4, (-1,3))
    pred5 = np.reshape(pred5, (-1,3))
    pred6 = np.reshape(pred6, (-1,3))
    pred7 = np.reshape(pred7, (-1,3))
    pred8 = np.reshape(pred8, (-1,3))
    pred9 = np.reshape(pred9, (-1,3))
    pred10 = np.reshape(pred10, (-1,3))

    output = kmeans.predict(pred1)
    output = (output/2)*255
    output = np.reshape(output, (dim1_1, dim1_2))
    cv2.imwrite('schistosoma_pred_1_1.png', output)

    output = kmeans.predict(pred2)
    output = (output/2)*255
    output = np.reshape(output, (dim2_1, dim2_2))
    cv2.imwrite('schistosoma_pred_1_2.png', output)

    output = kmeans.predict(pred3)
    output = (output/2)*255
    output = np.reshape(output, (dim3_1, dim3_2))
    cv2.imwrite('schistosoma_pred_1_3.png', output)

    output = kmeans.predict(pred4)
    output = (output/2)*255
    output = np.reshape(output, (dim4_1, dim4_2))
    cv2.imwrite('schistosoma_pred_1_4.png', output)

    output = kmeans.predict(pred5)
    output = (output/2)*255
    output = np.reshape(output, (dim5_1, dim5_2))
    cv2.imwrite('schistosoma_pred_1_5.png', output)

    output = kmeans.predict(pred6)
    output = (output/2)*255
    output = np.reshape(output, (dim6_1, dim6_2))
    cv2.imwrite('schistosoma_pred_1_6.png', output)

    output = kmeans.predict(pred7)
    output = (output/2)*255
    output = np.reshape(output, (dim7_1, dim7_2))
    cv2.imwrite('schistosoma_pred_1_7.png', output)

    output = kmeans.predict(pred8)
    output = (output/2)*255
    output = np.reshape(output, (dim8_1, dim8_2))
    cv2.imwrite('schistosoma_pred_1_8.png', output)

    output = kmeans.predict(pred9)
    output = (output/2)*255
    output = np.reshape(output, (dim9_1, dim9_2))
    cv2.imwrite('schistosoma_pred_1_9.png', output)

    output = kmeans.predict(pred10)
    output = (output/2)*255
    output = np.reshape(output, (dim10_1, dim10_2))
    cv2.imwrite('schistosoma_pred_1_10.png', output)

elif choosy == 2:

    #FILARIODIEA

    #Load the images for training
    img = cv2.imread('filarioidea/filaria.jpg')
    img2 = cv2.imread('filarioidea/filaria2.jpg')
    trainer = np.vstack((img, img2))

    #Save image dimensions
    dim1 = img.shape[0]
    dim2 = img.shape[1]

    #Reshape the image from 3D to 2D
    trainer = np.reshape(trainer, (-1,3))

    #Manually select centroids
    pick = np.array([[146,118,127],[195,197,197],[2,2,2]])

    #Perform KMeans (extra cluster for black portions)
    kmeans = KMeans(n_clusters=3, init=pick).fit(trainer)

    #Upscale the labels
    labs = (kmeans.labels_/2)*255

    #Reshape the upscaled labels from 1D to 2D
    labs = np.reshape(labs, (dim1*2,dim2))

    #Predicting part
    pred1 = cv2.imread('filarioidea/filaria.jpg')
    pred2 = cv2.imread('filarioidea/filaria2.jpg')
    pred3 = cv2.imread('filarioidea/filaria3.jpg')
    pred4 = cv2.imread('filarioidea/filaria4.jpg')
    pred5 = cv2.imread('filarioidea/filaria5.jpg')
    pred6 = cv2.imread('filarioidea/filaria6.jpg')
    pred7 = cv2.imread('filarioidea/filaria7.jpg')
    pred8 = cv2.imread('filarioidea/filaria8.jpg')
    pred9 = cv2.imread('filarioidea/filaria9.jpg')
    pred10 = cv2.imread('filarioidea/filaria10.jpg')

    dim1_1 = pred1.shape[0]
    dim1_2 = pred1.shape[1]
    dim2_1 = pred2.shape[0]
    dim2_2 = pred2.shape[1]
    dim3_1 = pred3.shape[0]
    dim3_2 = pred3.shape[1]
    dim4_1 = pred4.shape[0]
    dim4_2 = pred4.shape[1]
    dim5_1 = pred5.shape[0]
    dim5_2 = pred5.shape[1]
    dim6_1 = pred6.shape[0]
    dim6_2 = pred6.shape[1]
    dim7_1 = pred7.shape[0]
    dim7_2 = pred7.shape[1]
    dim8_1 = pred8.shape[0]
    dim8_2 = pred8.shape[1]
    dim9_1 = pred9.shape[0]
    dim9_2 = pred9.shape[1]
    dim10_1 = pred10.shape[0]
    dim10_2 = pred10.shape[1]

    pred1 = np.reshape(pred1, (-1,3))
    pred2 = np.reshape(pred2, (-1,3))
    pred3 = np.reshape(pred3, (-1,3))
    pred4 = np.reshape(pred4, (-1,3))
    pred5 = np.reshape(pred5, (-1,3))
    pred6 = np.reshape(pred6, (-1,3))
    pred7 = np.reshape(pred7, (-1,3))
    pred8 = np.reshape(pred8, (-1,3))
    pred9 = np.reshape(pred9, (-1,3))
    pred10 = np.reshape(pred10, (-1,3))

    output = kmeans.predict(pred1)
    output = (output/2)*255
    output = np.reshape(output, (dim1_1, dim1_2))
    cv2.imwrite('filaria_pred_2_1.png', output)

    output = kmeans.predict(pred2)
    output = (output/2)*255
    output = np.reshape(output, (dim2_1, dim2_2))
    cv2.imwrite('filaria_pred_2_2.png', output)

    output = kmeans.predict(pred3)
    output = (output/2)*255
    output = np.reshape(output, (dim3_1, dim3_2))
    cv2.imwrite('filaria_pred_2_3.png', output)

    output = kmeans.predict(pred4)
    output = (output/2)*255
    output = np.reshape(output, (dim4_1, dim4_2))
    cv2.imwrite('filaria_pred_2_4.png', output)

    output = kmeans.predict(pred5)
    output = (output/2)*255
    output = np.reshape(output, (dim5_1, dim5_2))
    cv2.imwrite('filaria_pred_2_5.png', output)

    output = kmeans.predict(pred6)
    output = (output/2)*255
    output = np.reshape(output, (dim6_1, dim6_2))
    cv2.imwrite('filaria_pred_2_6.png', output)

    output = kmeans.predict(pred7)
    output = (output/2)*255
    output = np.reshape(output, (dim7_1, dim7_2))
    cv2.imwrite('filaria_pred_2_7.png', output)

    output = kmeans.predict(pred8)
    output = (output/2)*255
    output = np.reshape(output, (dim8_1, dim8_2))
    cv2.imwrite('filaria_pred_2_8.png', output)

    output = kmeans.predict(pred9)
    output = (output/2)*255
    output = np.reshape(output, (dim9_1, dim9_2))
    cv2.imwrite('filaria_pred_2_9.png', output)

    output = kmeans.predict(pred10)
    output = (output/2)*255
    output = np.reshape(output, (dim10_1, dim10_2))
    cv2.imwrite('filaria_pred_2_10.png', output)

    #PLASMODIUM

    #Load the images for training
    img = cv2.imread('plasmodium/1c.JPG')
    img2 = cv2.imread('plasmodium/6c.JPG')
    trainer = np.vstack((img, img2))

    #Save image dimensions
    dim1 = img.shape[0]
    dim2 = img.shape[1]

    #Reshape the image from 3D to 2D
    trainer = np.reshape(trainer, (-1,3))

    #Manually select centroids

    pick = np.array([[62,54,94],[187,175,193]])

    #Perform KMeans (extra cluster for black portions)
    kmeans = KMeans(n_clusters=2, init=pick).fit(trainer)

    #Upscale the labels
    labs = (kmeans.labels_)*255

    #Reshape the upscaled labels from 1D to 2D
    labs = np.reshape(labs, (dim1*2,dim2))

    #Predicting part
    pred1 = cv2.imread('plasmodium/1c.JPG')
    pred2 = cv2.imread('plasmodium/3c.JPG')
    pred3 = cv2.imread('plasmodium/6c.JPG')
    pred4 = cv2.imread('plasmodium/7c.JPG')
    pred5 = cv2.imread('plasmodium/11c.JPG')
    pred6 = cv2.imread('plasmodium/19c.JPG')
    pred7 = cv2.imread('plasmodium/55c.JPG')
    pred8 = cv2.imread('plasmodium/79c.JPG')
    pred9 = cv2.imread('plasmodium/94c.JPG')
    pred10 = cv2.imread('plasmodium/105c.JPG')

    dim1_1 = pred1.shape[0]
    dim1_2 = pred1.shape[1]
    dim2_1 = pred2.shape[0]
    dim2_2 = pred2.shape[1]
    dim3_1 = pred3.shape[0]
    dim3_2 = pred3.shape[1]
    dim4_1 = pred4.shape[0]
    dim4_2 = pred4.shape[1]
    dim5_1 = pred5.shape[0]
    dim5_2 = pred5.shape[1]
    dim6_1 = pred6.shape[0]
    dim6_2 = pred6.shape[1]
    dim7_1 = pred7.shape[0]
    dim7_2 = pred7.shape[1]
    dim8_1 = pred8.shape[0]
    dim8_2 = pred8.shape[1]
    dim9_1 = pred9.shape[0]
    dim9_2 = pred9.shape[1]
    dim10_1 = pred10.shape[0]
    dim10_2 = pred10.shape[1]

    pred1 = np.reshape(pred1, (-1,3))
    pred2 = np.reshape(pred2, (-1,3))
    pred3 = np.reshape(pred3, (-1,3))
    pred4 = np.reshape(pred4, (-1,3))
    pred5 = np.reshape(pred5, (-1,3))
    pred6 = np.reshape(pred6, (-1,3))
    pred7 = np.reshape(pred7, (-1,3))
    pred8 = np.reshape(pred8, (-1,3))
    pred9 = np.reshape(pred9, (-1,3))
    pred10 = np.reshape(pred10, (-1,3))

    output = kmeans.predict(pred1)
    output = (output)*255
    output = np.reshape(output, (dim1_1, dim1_2))
    cv2.imwrite('plasmodium_pred_2_1.png', output)

    output = kmeans.predict(pred2)
    output = (output)*255
    output = np.reshape(output, (dim2_1, dim2_2))
    cv2.imwrite('plasmodium_pred_2_2.png', output)

    output = kmeans.predict(pred3)
    output = (output)*255
    output = np.reshape(output, (dim3_1, dim3_2))
    cv2.imwrite('plasmodium_pred_2_3.png', output)

    output = kmeans.predict(pred4)
    output = (output)*255
    output = np.reshape(output, (dim4_1, dim4_2))
    cv2.imwrite('plasmodium_pred_2_4.png', output)

    output = kmeans.predict(pred5)
    output = (output)*255
    output = np.reshape(output, (dim5_1, dim5_2))
    cv2.imwrite('plasmodium_pred_2_5.png', output)

    output = kmeans.predict(pred6)
    output = (output)*255
    output = np.reshape(output, (dim6_1, dim6_2))
    cv2.imwrite('plasmodium_pred_2_6.png', output)

    output = kmeans.predict(pred7)
    output = (output)*255
    output = np.reshape(output, (dim7_1, dim7_2))
    cv2.imwrite('plasmodium_pred_2_7.png', output)

    output = kmeans.predict(pred8)
    output = (output)*255
    output = np.reshape(output, (dim8_1, dim8_2))
    cv2.imwrite('plasmodium_pred_2_8.png', output)

    output = kmeans.predict(pred9)
    output = (output)*255
    output = np.reshape(output, (dim9_1, dim9_2))
    cv2.imwrite('plasmodium_pred_2_9.png', output)

    output = kmeans.predict(pred10)
    output = (output)*255
    output = np.reshape(output, (dim10_1, dim10_2))
    cv2.imwrite('plasmodium_pred_2_10.png', output)

    #SCHISTOSOMA

    #Load the images for training
    img = cv2.imread('schistosoma/schistosoma.jpg')
    img2 = cv2.imread('schistosoma/schistosoma2.jpg')
    trainer = np.vstack((img, img2))

    #Save image dimensions
    dim1 = img.shape[0]
    dim2 = img.shape[1]

    #Reshape the image from 3D to 2D
    trainer = np.reshape(trainer, (-1,3))

    #Manually select centroids

    pick = np.array([[18,131,187],[83,188,227],[2,2,2]])

    #Perform KMeans (extra cluster for black portions)
    kmeans = KMeans(n_clusters=3, init=pick).fit(trainer)

    #Upscale the labels
    labs = (kmeans.labels_/2)*255

    #Reshape the upscaled labels from 1D to 2D
    labs = np.reshape(labs, (dim1*2,dim2))

    #Predicting part
    pred1 = cv2.imread('schistosoma/schistosoma.jpg')
    pred2 = cv2.imread('schistosoma/schistosoma2.jpg')
    pred3 = cv2.imread('schistosoma/schistosoma3.jpg')
    pred4 = cv2.imread('schistosoma/schistosoma4.jpg')
    pred5 = cv2.imread('schistosoma/schistosoma5.jpg')
    pred6 = cv2.imread('schistosoma/schistosoma6.jpg')
    pred7 = cv2.imread('schistosoma/schistosoma7.jpg')
    pred8 = cv2.imread('schistosoma/schistosoma8.jpg')
    pred9 = cv2.imread('schistosoma/schistosoma9.jpg')
    pred10 = cv2.imread('schistosoma/schistosoma10.jpg')

    dim1_1 = pred1.shape[0]
    dim1_2 = pred1.shape[1]
    dim2_1 = pred2.shape[0]
    dim2_2 = pred2.shape[1]
    dim3_1 = pred3.shape[0]
    dim3_2 = pred3.shape[1]
    dim4_1 = pred4.shape[0]
    dim4_2 = pred4.shape[1]
    dim5_1 = pred5.shape[0]
    dim5_2 = pred5.shape[1]
    dim6_1 = pred6.shape[0]
    dim6_2 = pred6.shape[1]
    dim7_1 = pred7.shape[0]
    dim7_2 = pred7.shape[1]
    dim8_1 = pred8.shape[0]
    dim8_2 = pred8.shape[1]
    dim9_1 = pred9.shape[0]
    dim9_2 = pred9.shape[1]
    dim10_1 = pred10.shape[0]
    dim10_2 = pred10.shape[1]

    pred1 = np.reshape(pred1, (-1,3))
    pred2 = np.reshape(pred2, (-1,3))
    pred3 = np.reshape(pred3, (-1,3))
    pred4 = np.reshape(pred4, (-1,3))
    pred5 = np.reshape(pred5, (-1,3))
    pred6 = np.reshape(pred6, (-1,3))
    pred7 = np.reshape(pred7, (-1,3))
    pred8 = np.reshape(pred8, (-1,3))
    pred9 = np.reshape(pred9, (-1,3))
    pred10 = np.reshape(pred10, (-1,3))

    output = kmeans.predict(pred1)
    output = (output/2)*255
    output = np.reshape(output, (dim1_1, dim1_2))
    cv2.imwrite('schistosoma_pred_2_1.png', output)

    output = kmeans.predict(pred2)
    output = (output/2)*255
    output = np.reshape(output, (dim2_1, dim2_2))
    cv2.imwrite('schistosoma_pred_2_2.png', output)

    output = kmeans.predict(pred3)
    output = (output/2)*255
    output = np.reshape(output, (dim3_1, dim3_2))
    cv2.imwrite('schistosoma_pred_2_3.png', output)

    output = kmeans.predict(pred4)
    output = (output/2)*255
    output = np.reshape(output, (dim4_1, dim4_2))
    cv2.imwrite('schistosoma_pred_2_4.png', output)

    output = kmeans.predict(pred5)
    output = (output/2)*255
    output = np.reshape(output, (dim5_1, dim5_2))
    cv2.imwrite('schistosoma_pred_2_5.png', output)

    output = kmeans.predict(pred6)
    output = (output/2)*255
    output = np.reshape(output, (dim6_1, dim6_2))
    cv2.imwrite('schistosoma_pred_2_6.png', output)

    output = kmeans.predict(pred7)
    output = (output/2)*255
    output = np.reshape(output, (dim7_1, dim7_2))
    cv2.imwrite('schistosoma_pred_2_7.png', output)

    output = kmeans.predict(pred8)
    output = (output/2)*255
    output = np.reshape(output, (dim8_1, dim8_2))
    cv2.imwrite('schistosoma_pred_2_8.png', output)

    output = kmeans.predict(pred9)
    output = (output/2)*255
    output = np.reshape(output, (dim9_1, dim9_2))
    cv2.imwrite('schistosoma_pred_2_9.png', output)

    output = kmeans.predict(pred10)
    output = (output/2)*255
    output = np.reshape(output, (dim10_1, dim10_2))
    cv2.imwrite('schistosoma_pred_2_10.png', output)

elif choosy == 3: #5 CENTROIDS

    #FILARIODIEA

    #Load the images for training
    img = cv2.imread('filarioidea/filaria.jpg')
    img2 = cv2.imread('filarioidea/filaria2.jpg')
    trainer = np.vstack((img, img2))

    #Save image dimensions
    dim1 = img.shape[0]
    dim2 = img.shape[1]

    #Reshape the image from 3D to 2D
    trainer = np.reshape(trainer, (-1,3))

    #Perform KMeans (extra cluster for black portions)
    kmeans = KMeans(n_clusters=5).fit(trainer)

    #Upscale the labels
    labs = (kmeans.labels_/4)*255

    #Reshape the upscaled labels from 1D to 2D
    labs = np.reshape(labs, (dim1*2,dim2))

    #Predicting part
    pred1 = cv2.imread('filarioidea/filaria.jpg')
    pred2 = cv2.imread('filarioidea/filaria2.jpg')
    pred3 = cv2.imread('filarioidea/filaria3.jpg')
    pred4 = cv2.imread('filarioidea/filaria4.jpg')
    pred5 = cv2.imread('filarioidea/filaria5.jpg')
    pred6 = cv2.imread('filarioidea/filaria6.jpg')
    pred7 = cv2.imread('filarioidea/filaria7.jpg')
    pred8 = cv2.imread('filarioidea/filaria8.jpg')
    pred9 = cv2.imread('filarioidea/filaria9.jpg')
    pred10 = cv2.imread('filarioidea/filaria10.jpg')

    dim1_1 = pred1.shape[0]
    dim1_2 = pred1.shape[1]
    dim2_1 = pred2.shape[0]
    dim2_2 = pred2.shape[1]
    dim3_1 = pred3.shape[0]
    dim3_2 = pred3.shape[1]
    dim4_1 = pred4.shape[0]
    dim4_2 = pred4.shape[1]
    dim5_1 = pred5.shape[0]
    dim5_2 = pred5.shape[1]
    dim6_1 = pred6.shape[0]
    dim6_2 = pred6.shape[1]
    dim7_1 = pred7.shape[0]
    dim7_2 = pred7.shape[1]
    dim8_1 = pred8.shape[0]
    dim8_2 = pred8.shape[1]
    dim9_1 = pred9.shape[0]
    dim9_2 = pred9.shape[1]
    dim10_1 = pred10.shape[0]
    dim10_2 = pred10.shape[1]

    pred1 = np.reshape(pred1, (-1,3))
    pred2 = np.reshape(pred2, (-1,3))
    pred3 = np.reshape(pred3, (-1,3))
    pred4 = np.reshape(pred4, (-1,3))
    pred5 = np.reshape(pred5, (-1,3))
    pred6 = np.reshape(pred6, (-1,3))
    pred7 = np.reshape(pred7, (-1,3))
    pred8 = np.reshape(pred8, (-1,3))
    pred9 = np.reshape(pred9, (-1,3))
    pred10 = np.reshape(pred10, (-1,3))

    output = kmeans.predict(pred1)
    output = (output/4)*255
    output = np.reshape(output, (dim1_1, dim1_2))
    cv2.imwrite('filaria_pred_3_1.png', output)

    output = kmeans.predict(pred2)
    output = (output/4)*255
    output = np.reshape(output, (dim2_1, dim2_2))
    cv2.imwrite('filaria_pred_3_2.png', output)

    output = kmeans.predict(pred3)
    output = (output/4)*255
    output = np.reshape(output, (dim3_1, dim3_2))
    cv2.imwrite('filaria_pred_3_3.png', output)

    output = kmeans.predict(pred4)
    output = (output/4)*255
    output = np.reshape(output, (dim4_1, dim4_2))
    cv2.imwrite('filaria_pred_3_4.png', output)

    output = kmeans.predict(pred5)
    output = (output/4)*255
    output = np.reshape(output, (dim5_1, dim5_2))
    cv2.imwrite('filaria_pred_3_5.png', output)

    output = kmeans.predict(pred6)
    output = (output/4)*255
    output = np.reshape(output, (dim6_1, dim6_2))
    cv2.imwrite('filaria_pred_3_6.png', output)

    output = kmeans.predict(pred7)
    output = (output/4)*255
    output = np.reshape(output, (dim7_1, dim7_2))
    cv2.imwrite('filaria_pred_3_7.png', output)

    output = kmeans.predict(pred8)
    output = (output/4)*255
    output = np.reshape(output, (dim8_1, dim8_2))
    cv2.imwrite('filaria_pred_3_8.png', output)

    output = kmeans.predict(pred9)
    output = (output/4)*255
    output = np.reshape(output, (dim9_1, dim9_2))
    cv2.imwrite('filaria_pred_3_9.png', output)

    output = kmeans.predict(pred10)
    output = (output/4)*255
    output = np.reshape(output, (dim10_1, dim10_2))
    cv2.imwrite('filaria_pred_3_10.png', output)

    #PLASMODIUM

    #Load the images for training
    img = cv2.imread('plasmodium/1c.JPG')
    img2 = cv2.imread('plasmodium/6c.JPG')
    trainer = np.vstack((img, img2))

    #Save image dimensions
    dim1 = img.shape[0]
    dim2 = img.shape[1]

    #Reshape the image from 3D to 2D
    trainer = np.reshape(trainer, (-1,3))

    #Perform KMeans (no extra cluster for black portions)
    kmeans = KMeans(n_clusters=4).fit(trainer)

    #Upscale the labels
    labs = (kmeans.labels_/3)*255

    #Reshape the upscaled labels from 1D to 2D
    labs = np.reshape(labs, (dim1*2,dim2))

    #Predicting part
    pred1 = cv2.imread('plasmodium/1c.JPG')
    pred2 = cv2.imread('plasmodium/3c.JPG')
    pred3 = cv2.imread('plasmodium/6c.JPG')
    pred4 = cv2.imread('plasmodium/7c.JPG')
    pred5 = cv2.imread('plasmodium/11c.JPG')
    pred6 = cv2.imread('plasmodium/19c.JPG')
    pred7 = cv2.imread('plasmodium/55c.JPG')
    pred8 = cv2.imread('plasmodium/79c.JPG')
    pred9 = cv2.imread('plasmodium/94c.JPG')
    pred10 = cv2.imread('plasmodium/105c.JPG')

    dim1_1 = pred1.shape[0]
    dim1_2 = pred1.shape[1]
    dim2_1 = pred2.shape[0]
    dim2_2 = pred2.shape[1]
    dim3_1 = pred3.shape[0]
    dim3_2 = pred3.shape[1]
    dim4_1 = pred4.shape[0]
    dim4_2 = pred4.shape[1]
    dim5_1 = pred5.shape[0]
    dim5_2 = pred5.shape[1]
    dim6_1 = pred6.shape[0]
    dim6_2 = pred6.shape[1]
    dim7_1 = pred7.shape[0]
    dim7_2 = pred7.shape[1]
    dim8_1 = pred8.shape[0]
    dim8_2 = pred8.shape[1]
    dim9_1 = pred9.shape[0]
    dim9_2 = pred9.shape[1]
    dim10_1 = pred10.shape[0]
    dim10_2 = pred10.shape[1]

    pred1 = np.reshape(pred1, (-1,3))
    pred2 = np.reshape(pred2, (-1,3))
    pred3 = np.reshape(pred3, (-1,3))
    pred4 = np.reshape(pred4, (-1,3))
    pred5 = np.reshape(pred5, (-1,3))
    pred6 = np.reshape(pred6, (-1,3))
    pred7 = np.reshape(pred7, (-1,3))
    pred8 = np.reshape(pred8, (-1,3))
    pred9 = np.reshape(pred9, (-1,3))
    pred10 = np.reshape(pred10, (-1,3))

    output = kmeans.predict(pred1)
    output = (output/3)*255
    output = np.reshape(output, (dim1_1, dim1_2))
    cv2.imwrite('plasmodium_pred_3_1.png', output)

    output = kmeans.predict(pred2)
    output = (output/3)*255
    output = np.reshape(output, (dim2_1, dim2_2))
    cv2.imwrite('plasmodium_pred_3_2.png', output)

    output = kmeans.predict(pred3)
    output = (output/3)*255
    output = np.reshape(output, (dim3_1, dim3_2))
    cv2.imwrite('plasmodium_pred_3_3.png', output)

    output = kmeans.predict(pred4)
    output = (output/3)*255
    output = np.reshape(output, (dim4_1, dim4_2))
    cv2.imwrite('plasmodium_pred_3_4.png', output)

    output = kmeans.predict(pred5)
    output = (output/3)*255
    output = np.reshape(output, (dim5_1, dim5_2))
    cv2.imwrite('plasmodium_pred_3_5.png', output)

    output = kmeans.predict(pred6)
    output = (output/3)*255
    output = np.reshape(output, (dim6_1, dim6_2))
    cv2.imwrite('plasmodium_pred_3_6.png', output)

    output = kmeans.predict(pred7)
    output = (output/3)*255
    output = np.reshape(output, (dim7_1, dim7_2))
    cv2.imwrite('plasmodium_pred_3_7.png', output)

    output = kmeans.predict(pred8)
    output = (output/3)*255
    output = np.reshape(output, (dim8_1, dim8_2))
    cv2.imwrite('plasmodium_pred_3_8.png', output)

    output = kmeans.predict(pred9)
    output = (output/3)*255
    output = np.reshape(output, (dim9_1, dim9_2))
    cv2.imwrite('plasmodium_pred_3_9.png', output)

    output = kmeans.predict(pred10)
    output = (output/3)*255
    output = np.reshape(output, (dim10_1, dim10_2))
    cv2.imwrite('plasmodium_pred_3_10.png', output)

    #SCHISTOSOMA

    #Load the images for training
    img = cv2.imread('schistosoma/schistosoma.jpg')
    img2 = cv2.imread('schistosoma/schistosoma2.jpg')
    trainer = np.vstack((img, img2))

    #Save image dimensions
    dim1 = img.shape[0]
    dim2 = img.shape[1]

    #Reshape the image from 3D to 2D
    trainer = np.reshape(trainer, (-1,3))

    #Perform KMeans (extra cluster for black portions)
    kmeans = KMeans(n_clusters=5).fit(trainer)

    #Upscale the labels
    labs = (kmeans.labels_/4)*255

    #Reshape the upscaled labels from 1D to 2D
    labs = np.reshape(labs, (dim1*2,dim2))

    #Predicting part
    pred1 = cv2.imread('schistosoma/schistosoma.jpg')
    pred2 = cv2.imread('schistosoma/schistosoma2.jpg')
    pred3 = cv2.imread('schistosoma/schistosoma3.jpg')
    pred4 = cv2.imread('schistosoma/schistosoma4.jpg')
    pred5 = cv2.imread('schistosoma/schistosoma5.jpg')
    pred6 = cv2.imread('schistosoma/schistosoma6.jpg')
    pred7 = cv2.imread('schistosoma/schistosoma7.jpg')
    pred8 = cv2.imread('schistosoma/schistosoma8.jpg')
    pred9 = cv2.imread('schistosoma/schistosoma9.jpg')
    pred10 = cv2.imread('schistosoma/schistosoma10.jpg')

    dim1_1 = pred1.shape[0]
    dim1_2 = pred1.shape[1]
    dim2_1 = pred2.shape[0]
    dim2_2 = pred2.shape[1]
    dim3_1 = pred3.shape[0]
    dim3_2 = pred3.shape[1]
    dim4_1 = pred4.shape[0]
    dim4_2 = pred4.shape[1]
    dim5_1 = pred5.shape[0]
    dim5_2 = pred5.shape[1]
    dim6_1 = pred6.shape[0]
    dim6_2 = pred6.shape[1]
    dim7_1 = pred7.shape[0]
    dim7_2 = pred7.shape[1]
    dim8_1 = pred8.shape[0]
    dim8_2 = pred8.shape[1]
    dim9_1 = pred9.shape[0]
    dim9_2 = pred9.shape[1]
    dim10_1 = pred10.shape[0]
    dim10_2 = pred10.shape[1]

    pred1 = np.reshape(pred1, (-1,3))
    pred2 = np.reshape(pred2, (-1,3))
    pred3 = np.reshape(pred3, (-1,3))
    pred4 = np.reshape(pred4, (-1,3))
    pred5 = np.reshape(pred5, (-1,3))
    pred6 = np.reshape(pred6, (-1,3))
    pred7 = np.reshape(pred7, (-1,3))
    pred8 = np.reshape(pred8, (-1,3))
    pred9 = np.reshape(pred9, (-1,3))
    pred10 = np.reshape(pred10, (-1,3))

    output = kmeans.predict(pred1)
    output = (output/4)*255
    output = np.reshape(output, (dim1_1, dim1_2))
    cv2.imwrite('schistosoma_pred_3_1.png', output)

    output = kmeans.predict(pred2)
    output = (output/4)*255
    output = np.reshape(output, (dim2_1, dim2_2))
    cv2.imwrite('schistosoma_pred_3_2.png', output)

    output = kmeans.predict(pred3)
    output = (output/4)*255
    output = np.reshape(output, (dim3_1, dim3_2))
    cv2.imwrite('schistosoma_pred_3_3.png', output)

    output = kmeans.predict(pred4)
    output = (output/4)*255
    output = np.reshape(output, (dim4_1, dim4_2))
    cv2.imwrite('schistosoma_pred_3_4.png', output)

    output = kmeans.predict(pred5)
    output = (output/4)*255
    output = np.reshape(output, (dim5_1, dim5_2))
    cv2.imwrite('schistosoma_pred_3_5.png', output)

    output = kmeans.predict(pred6)
    output = (output/4)*255
    output = np.reshape(output, (dim6_1, dim6_2))
    cv2.imwrite('schistosoma_pred_3_6.png', output)

    output = kmeans.predict(pred7)
    output = (output/4)*255
    output = np.reshape(output, (dim7_1, dim7_2))
    cv2.imwrite('schistosoma_pred_3_7.png', output)

    output = kmeans.predict(pred8)
    output = (output/4)*255
    output = np.reshape(output, (dim8_1, dim8_2))
    cv2.imwrite('schistosoma_pred_3_8.png', output)

    output = kmeans.predict(pred9)
    output = (output/4)*255
    output = np.reshape(output, (dim9_1, dim9_2))
    cv2.imwrite('schistosoma_pred_3_9.png', output)

    output = kmeans.predict(pred10)
    output = (output/4)*255
    output = np.reshape(output, (dim10_1, dim10_2))
    cv2.imwrite('schistosoma_pred_3_10.png', output)

elif choosy == 4:

    #FILARIODIEA

    #Load the images for training
    img = cv2.imread('filarioidea/filaria.jpg')
    img2 = cv2.imread('filarioidea/filaria2.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
    trainer = np.vstack((img, img2))

    #Save image dimensions
    dim1 = img.shape[0]
    dim2 = img.shape[1]

    #Reshape the image from 3D to 2D
    trainer = np.reshape(trainer, (-1,3))

    #Perform KMeans (extra cluster for black portions)
    kmeans = KMeans(n_clusters=3).fit(trainer)

    #Upscale the labels
    labs = (kmeans.labels_/2)*255

    #Reshape the upscaled labels from 1D to 2D
    labs = np.reshape(labs, (dim1*2,dim2))

    #Predicting part
    pred1 = cv2.imread('filarioidea/filaria.jpg')
    pred2 = cv2.imread('filarioidea/filaria2.jpg')
    pred3 = cv2.imread('filarioidea/filaria3.jpg')
    pred4 = cv2.imread('filarioidea/filaria4.jpg')
    pred5 = cv2.imread('filarioidea/filaria5.jpg')
    pred6 = cv2.imread('filarioidea/filaria6.jpg')
    pred7 = cv2.imread('filarioidea/filaria7.jpg')
    pred8 = cv2.imread('filarioidea/filaria8.jpg')
    pred9 = cv2.imread('filarioidea/filaria9.jpg')
    pred10 = cv2.imread('filarioidea/filaria10.jpg')

    pred1 = cv2.cvtColor(pred1, cv2.COLOR_BGR2HSV)
    pred2 = cv2.cvtColor(pred2, cv2.COLOR_BGR2HSV)
    pred3 = cv2.cvtColor(pred3, cv2.COLOR_BGR2HSV)
    pred4 = cv2.cvtColor(pred4, cv2.COLOR_BGR2HSV)
    pred5 = cv2.cvtColor(pred5, cv2.COLOR_BGR2HSV)
    pred6 = cv2.cvtColor(pred6, cv2.COLOR_BGR2HSV)
    pred7 = cv2.cvtColor(pred7, cv2.COLOR_BGR2HSV)
    pred8 = cv2.cvtColor(pred8, cv2.COLOR_BGR2HSV)
    pred9 = cv2.cvtColor(pred9, cv2.COLOR_BGR2HSV)
    pred10 = cv2.cvtColor(pred10, cv2.COLOR_BGR2HSV)

    dim1_1 = pred1.shape[0]
    dim1_2 = pred1.shape[1]
    dim2_1 = pred2.shape[0]
    dim2_2 = pred2.shape[1]
    dim3_1 = pred3.shape[0]
    dim3_2 = pred3.shape[1]
    dim4_1 = pred4.shape[0]
    dim4_2 = pred4.shape[1]
    dim5_1 = pred5.shape[0]
    dim5_2 = pred5.shape[1]
    dim6_1 = pred6.shape[0]
    dim6_2 = pred6.shape[1]
    dim7_1 = pred7.shape[0]
    dim7_2 = pred7.shape[1]
    dim8_1 = pred8.shape[0]
    dim8_2 = pred8.shape[1]
    dim9_1 = pred9.shape[0]
    dim9_2 = pred9.shape[1]
    dim10_1 = pred10.shape[0]
    dim10_2 = pred10.shape[1]

    pred1 = np.reshape(pred1, (-1,3))
    pred2 = np.reshape(pred2, (-1,3))
    pred3 = np.reshape(pred3, (-1,3))
    pred4 = np.reshape(pred4, (-1,3))
    pred5 = np.reshape(pred5, (-1,3))
    pred6 = np.reshape(pred6, (-1,3))
    pred7 = np.reshape(pred7, (-1,3))
    pred8 = np.reshape(pred8, (-1,3))
    pred9 = np.reshape(pred9, (-1,3))
    pred10 = np.reshape(pred10, (-1,3))

    output = kmeans.predict(pred1)
    output = (output/2)*255
    output = np.reshape(output, (dim1_1, dim1_2))
    cv2.imwrite('filaria_pred_4_1.png', output)

    output = kmeans.predict(pred2)
    output = (output/2)*255
    output = np.reshape(output, (dim2_1, dim2_2))
    cv2.imwrite('filaria_pred_4_2.png', output)

    output = kmeans.predict(pred3)
    output = (output/2)*255
    output = np.reshape(output, (dim3_1, dim3_2))
    cv2.imwrite('filaria_pred_4_3.png', output)

    output = kmeans.predict(pred4)
    output = (output/2)*255
    output = np.reshape(output, (dim4_1, dim4_2))
    cv2.imwrite('filaria_pred_4_4.png', output)

    output = kmeans.predict(pred5)
    output = (output/2)*255
    output = np.reshape(output, (dim5_1, dim5_2))
    cv2.imwrite('filaria_pred_4_5.png', output)

    output = kmeans.predict(pred6)
    output = (output/2)*255
    output = np.reshape(output, (dim6_1, dim6_2))
    cv2.imwrite('filaria_pred_4_6.png', output)

    output = kmeans.predict(pred7)
    output = (output/2)*255
    output = np.reshape(output, (dim7_1, dim7_2))
    cv2.imwrite('filaria_pred_4_7.png', output)

    output = kmeans.predict(pred8)
    output = (output/2)*255
    output = np.reshape(output, (dim8_1, dim8_2))
    cv2.imwrite('filaria_pred_4_8.png', output)

    output = kmeans.predict(pred9)
    output = (output/2)*255
    output = np.reshape(output, (dim9_1, dim9_2))
    cv2.imwrite('filaria_pred_4_9.png', output)

    output = kmeans.predict(pred10)
    output = (output/2)*255
    output = np.reshape(output, (dim10_1, dim10_2))
    cv2.imwrite('filaria_pred_4_10.png', output)

    #PLASMODIUM

    #Load the images for training
    img = cv2.imread('plasmodium/1c.JPG')
    img2 = cv2.imread('plasmodium/6c.JPG')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
    trainer = np.vstack((img, img2))

    #Save image dimensions
    dim1 = img.shape[0]
    dim2 = img.shape[1]

    #Reshape the image from 3D to 2D
    trainer = np.reshape(trainer, (-1,3))

    #Perform KMeans (no extra cluster for black portions)
    kmeans = KMeans(n_clusters=2).fit(trainer)

    #Upscale the labels
    labs = (kmeans.labels_)*255

    #Reshape the upscaled labels from 1D to 2D
    labs = np.reshape(labs, (dim1*2,dim2))

    #Predicting part
    pred1 = cv2.imread('plasmodium/1c.JPG')
    pred2 = cv2.imread('plasmodium/3c.JPG')
    pred3 = cv2.imread('plasmodium/6c.JPG')
    pred4 = cv2.imread('plasmodium/7c.JPG')
    pred5 = cv2.imread('plasmodium/11c.JPG')
    pred6 = cv2.imread('plasmodium/19c.JPG')
    pred7 = cv2.imread('plasmodium/55c.JPG')
    pred8 = cv2.imread('plasmodium/79c.JPG')
    pred9 = cv2.imread('plasmodium/94c.JPG')
    pred10 = cv2.imread('plasmodium/105c.JPG')

    pred1 = cv2.cvtColor(pred1, cv2.COLOR_BGR2HSV)
    pred2 = cv2.cvtColor(pred2, cv2.COLOR_BGR2HSV)
    pred3 = cv2.cvtColor(pred3, cv2.COLOR_BGR2HSV)
    pred4 = cv2.cvtColor(pred4, cv2.COLOR_BGR2HSV)
    pred5 = cv2.cvtColor(pred5, cv2.COLOR_BGR2HSV)
    pred6 = cv2.cvtColor(pred6, cv2.COLOR_BGR2HSV)
    pred7 = cv2.cvtColor(pred7, cv2.COLOR_BGR2HSV)
    pred8 = cv2.cvtColor(pred8, cv2.COLOR_BGR2HSV)
    pred9 = cv2.cvtColor(pred9, cv2.COLOR_BGR2HSV)
    pred10 = cv2.cvtColor(pred10, cv2.COLOR_BGR2HSV)

    dim1_1 = pred1.shape[0]
    dim1_2 = pred1.shape[1]
    dim2_1 = pred2.shape[0]
    dim2_2 = pred2.shape[1]
    dim3_1 = pred3.shape[0]
    dim3_2 = pred3.shape[1]
    dim4_1 = pred4.shape[0]
    dim4_2 = pred4.shape[1]
    dim5_1 = pred5.shape[0]
    dim5_2 = pred5.shape[1]
    dim6_1 = pred6.shape[0]
    dim6_2 = pred6.shape[1]
    dim7_1 = pred7.shape[0]
    dim7_2 = pred7.shape[1]
    dim8_1 = pred8.shape[0]
    dim8_2 = pred8.shape[1]
    dim9_1 = pred9.shape[0]
    dim9_2 = pred9.shape[1]
    dim10_1 = pred10.shape[0]
    dim10_2 = pred10.shape[1]

    pred1 = np.reshape(pred1, (-1,3))
    pred2 = np.reshape(pred2, (-1,3))
    pred3 = np.reshape(pred3, (-1,3))
    pred4 = np.reshape(pred4, (-1,3))
    pred5 = np.reshape(pred5, (-1,3))
    pred6 = np.reshape(pred6, (-1,3))
    pred7 = np.reshape(pred7, (-1,3))
    pred8 = np.reshape(pred8, (-1,3))
    pred9 = np.reshape(pred9, (-1,3))
    pred10 = np.reshape(pred10, (-1,3))

    output = kmeans.predict(pred1)
    output = (output)*255
    output = np.reshape(output, (dim1_1, dim1_2))
    cv2.imwrite('plasmodium_pred_4_1.png', output)

    output = kmeans.predict(pred2)
    output = (output)*255
    output = np.reshape(output, (dim2_1, dim2_2))
    cv2.imwrite('plasmodium_pred_4_2.png', output)

    output = kmeans.predict(pred3)
    output = (output)*255
    output = np.reshape(output, (dim3_1, dim3_2))
    cv2.imwrite('plasmodium_pred_4_3.png', output)

    output = kmeans.predict(pred4)
    output = (output)*255
    output = np.reshape(output, (dim4_1, dim4_2))
    cv2.imwrite('plasmodium_pred_4_4.png', output)

    output = kmeans.predict(pred5)
    output = (output)*255
    output = np.reshape(output, (dim5_1, dim5_2))
    cv2.imwrite('plasmodium_pred_4_5.png', output)

    output = kmeans.predict(pred6)
    output = (output)*255
    output = np.reshape(output, (dim6_1, dim6_2))
    cv2.imwrite('plasmodium_pred_4_6.png', output)

    output = kmeans.predict(pred7)
    output = (output)*255
    output = np.reshape(output, (dim7_1, dim7_2))
    cv2.imwrite('plasmodium_pred_4_7.png', output)

    output = kmeans.predict(pred8)
    output = (output)*255
    output = np.reshape(output, (dim8_1, dim8_2))
    cv2.imwrite('plasmodium_pred_4_8.png', output)

    output = kmeans.predict(pred9)
    output = (output)*255
    output = np.reshape(output, (dim9_1, dim9_2))
    cv2.imwrite('plasmodium_pred_4_9.png', output)

    output = kmeans.predict(pred10)
    output = (output)*255
    output = np.reshape(output, (dim10_1, dim10_2))
    cv2.imwrite('plasmodium_pred_4_10.png', output)

    #SCHISTOSOMA

    #Load the images for training
    img = cv2.imread('schistosoma/schistosoma.jpg')
    img2 = cv2.imread('schistosoma/schistosoma2.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
    trainer = np.vstack((img, img2))

    #Save image dimensions
    dim1 = img.shape[0]
    dim2 = img.shape[1]

    #Reshape the image from 3D to 2D
    trainer = np.reshape(trainer, (-1,3))

    #Perform KMeans (extra cluster for black portions)
    kmeans = KMeans(n_clusters=3).fit(trainer)

    #Upscale the labels
    labs = (kmeans.labels_/2)*255

    #Reshape the upscaled labels from 1D to 2D
    labs = np.reshape(labs, (dim1*2,dim2))

    #Predicting part
    pred1 = cv2.imread('schistosoma/schistosoma.jpg')
    pred2 = cv2.imread('schistosoma/schistosoma2.jpg')
    pred3 = cv2.imread('schistosoma/schistosoma3.jpg')
    pred4 = cv2.imread('schistosoma/schistosoma4.jpg')
    pred5 = cv2.imread('schistosoma/schistosoma5.jpg')
    pred6 = cv2.imread('schistosoma/schistosoma6.jpg')
    pred7 = cv2.imread('schistosoma/schistosoma7.jpg')
    pred8 = cv2.imread('schistosoma/schistosoma8.jpg')
    pred9 = cv2.imread('schistosoma/schistosoma9.jpg')
    pred10 = cv2.imread('schistosoma/schistosoma10.jpg')

    pred1 = cv2.cvtColor(pred1, cv2.COLOR_BGR2HSV)
    pred2 = cv2.cvtColor(pred2, cv2.COLOR_BGR2HSV)
    pred3 = cv2.cvtColor(pred3, cv2.COLOR_BGR2HSV)
    pred4 = cv2.cvtColor(pred4, cv2.COLOR_BGR2HSV)
    pred5 = cv2.cvtColor(pred5, cv2.COLOR_BGR2HSV)
    pred6 = cv2.cvtColor(pred6, cv2.COLOR_BGR2HSV)
    pred7 = cv2.cvtColor(pred7, cv2.COLOR_BGR2HSV)
    pred8 = cv2.cvtColor(pred8, cv2.COLOR_BGR2HSV)
    pred9 = cv2.cvtColor(pred9, cv2.COLOR_BGR2HSV)
    pred10 = cv2.cvtColor(pred10, cv2.COLOR_BGR2HSV)

    dim1_1 = pred1.shape[0]
    dim1_2 = pred1.shape[1]
    dim2_1 = pred2.shape[0]
    dim2_2 = pred2.shape[1]
    dim3_1 = pred3.shape[0]
    dim3_2 = pred3.shape[1]
    dim4_1 = pred4.shape[0]
    dim4_2 = pred4.shape[1]
    dim5_1 = pred5.shape[0]
    dim5_2 = pred5.shape[1]
    dim6_1 = pred6.shape[0]
    dim6_2 = pred6.shape[1]
    dim7_1 = pred7.shape[0]
    dim7_2 = pred7.shape[1]
    dim8_1 = pred8.shape[0]
    dim8_2 = pred8.shape[1]
    dim9_1 = pred9.shape[0]
    dim9_2 = pred9.shape[1]
    dim10_1 = pred10.shape[0]
    dim10_2 = pred10.shape[1]

    pred1 = np.reshape(pred1, (-1,3))
    pred2 = np.reshape(pred2, (-1,3))
    pred3 = np.reshape(pred3, (-1,3))
    pred4 = np.reshape(pred4, (-1,3))
    pred5 = np.reshape(pred5, (-1,3))
    pred6 = np.reshape(pred6, (-1,3))
    pred7 = np.reshape(pred7, (-1,3))
    pred8 = np.reshape(pred8, (-1,3))
    pred9 = np.reshape(pred9, (-1,3))
    pred10 = np.reshape(pred10, (-1,3))

    output = kmeans.predict(pred1)
    output = (output/2)*255
    output = np.reshape(output, (dim1_1, dim1_2))
    cv2.imwrite('schistosoma_pred_4_1.png', output)

    output = kmeans.predict(pred2)
    output = (output/2)*255
    output = np.reshape(output, (dim2_1, dim2_2))
    cv2.imwrite('schistosoma_pred_4_2.png', output)

    output = kmeans.predict(pred3)
    output = (output/2)*255
    output = np.reshape(output, (dim3_1, dim3_2))
    cv2.imwrite('schistosoma_pred_4_3.png', output)

    output = kmeans.predict(pred4)
    output = (output/2)*255
    output = np.reshape(output, (dim4_1, dim4_2))
    cv2.imwrite('schistosoma_pred_4_4.png', output)

    output = kmeans.predict(pred5)
    output = (output/2)*255
    output = np.reshape(output, (dim5_1, dim5_2))
    cv2.imwrite('schistosoma_pred_4_5.png', output)

    output = kmeans.predict(pred6)
    output = (output/2)*255
    output = np.reshape(output, (dim6_1, dim6_2))
    cv2.imwrite('schistosoma_pred_4_6.png', output)

    output = kmeans.predict(pred7)
    output = (output/2)*255
    output = np.reshape(output, (dim7_1, dim7_2))
    cv2.imwrite('schistosoma_pred_4_7.png', output)

    output = kmeans.predict(pred8)
    output = (output/2)*255
    output = np.reshape(output, (dim8_1, dim8_2))
    cv2.imwrite('schistosoma_pred_4_8.png', output)

    output = kmeans.predict(pred9)
    output = (output/2)*255
    output = np.reshape(output, (dim9_1, dim9_2))
    cv2.imwrite('schistosoma_pred_4_9.png', output)

    output = kmeans.predict(pred10)
    output = (output/2)*255
    output = np.reshape(output, (dim10_1, dim10_2))
    cv2.imwrite('schistosoma_pred_4_10.png', output)

elif choosy == 5:

    #FILARIODIEA

    #Load the images for training
    img = cv2.imread('filarioidea/filaria.jpg')
    img2 = cv2.imread('filarioidea/filaria2.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2Lab)
    trainer = np.vstack((img, img2))

    #Save image dimensions
    dim1 = img.shape[0]
    dim2 = img.shape[1]

    #Reshape the image from 3D to 2D
    trainer = np.reshape(trainer, (-1,3))

    #Perform KMeans (extra cluster for black portions)
    kmeans = KMeans(n_clusters=3).fit(trainer)

    #Upscale the labels
    labs = (kmeans.labels_/2)*255

    #Reshape the upscaled labels from 1D to 2D
    labs = np.reshape(labs, (dim1*2,dim2))

    #Predicting part
    pred1 = cv2.imread('filarioidea/filaria.jpg')
    pred2 = cv2.imread('filarioidea/filaria2.jpg')
    pred3 = cv2.imread('filarioidea/filaria3.jpg')
    pred4 = cv2.imread('filarioidea/filaria4.jpg')
    pred5 = cv2.imread('filarioidea/filaria5.jpg')
    pred6 = cv2.imread('filarioidea/filaria6.jpg')
    pred7 = cv2.imread('filarioidea/filaria7.jpg')
    pred8 = cv2.imread('filarioidea/filaria8.jpg')
    pred9 = cv2.imread('filarioidea/filaria9.jpg')
    pred10 = cv2.imread('filarioidea/filaria10.jpg')

    pred1 = cv2.cvtColor(pred1, cv2.COLOR_BGR2Lab)
    pred2 = cv2.cvtColor(pred2, cv2.COLOR_BGR2Lab)
    pred3 = cv2.cvtColor(pred3, cv2.COLOR_BGR2Lab)
    pred4 = cv2.cvtColor(pred4, cv2.COLOR_BGR2Lab)
    pred5 = cv2.cvtColor(pred5, cv2.COLOR_BGR2Lab)
    pred6 = cv2.cvtColor(pred6, cv2.COLOR_BGR2Lab)
    pred7 = cv2.cvtColor(pred7, cv2.COLOR_BGR2Lab)
    pred8 = cv2.cvtColor(pred8, cv2.COLOR_BGR2Lab)
    pred9 = cv2.cvtColor(pred9, cv2.COLOR_BGR2Lab)
    pred10 = cv2.cvtColor(pred10, cv2.COLOR_BGR2Lab)

    dim1_1 = pred1.shape[0]
    dim1_2 = pred1.shape[1]
    dim2_1 = pred2.shape[0]
    dim2_2 = pred2.shape[1]
    dim3_1 = pred3.shape[0]
    dim3_2 = pred3.shape[1]
    dim4_1 = pred4.shape[0]
    dim4_2 = pred4.shape[1]
    dim5_1 = pred5.shape[0]
    dim5_2 = pred5.shape[1]
    dim6_1 = pred6.shape[0]
    dim6_2 = pred6.shape[1]
    dim7_1 = pred7.shape[0]
    dim7_2 = pred7.shape[1]
    dim8_1 = pred8.shape[0]
    dim8_2 = pred8.shape[1]
    dim9_1 = pred9.shape[0]
    dim9_2 = pred9.shape[1]
    dim10_1 = pred10.shape[0]
    dim10_2 = pred10.shape[1]

    pred1 = np.reshape(pred1, (-1,3))
    pred2 = np.reshape(pred2, (-1,3))
    pred3 = np.reshape(pred3, (-1,3))
    pred4 = np.reshape(pred4, (-1,3))
    pred5 = np.reshape(pred5, (-1,3))
    pred6 = np.reshape(pred6, (-1,3))
    pred7 = np.reshape(pred7, (-1,3))
    pred8 = np.reshape(pred8, (-1,3))
    pred9 = np.reshape(pred9, (-1,3))
    pred10 = np.reshape(pred10, (-1,3))

    output = kmeans.predict(pred1)
    output = (output/2)*255
    output = np.reshape(output, (dim1_1, dim1_2))
    cv2.imwrite('filaria_pred_5_1.png', output)

    output = kmeans.predict(pred2)
    output = (output/2)*255
    output = np.reshape(output, (dim2_1, dim2_2))
    cv2.imwrite('filaria_pred_5_2.png', output)

    output = kmeans.predict(pred3)
    output = (output/2)*255
    output = np.reshape(output, (dim3_1, dim3_2))
    cv2.imwrite('filaria_pred_5_3.png', output)

    output = kmeans.predict(pred4)
    output = (output/2)*255
    output = np.reshape(output, (dim4_1, dim4_2))
    cv2.imwrite('filaria_pred_5_4.png', output)

    output = kmeans.predict(pred5)
    output = (output/2)*255
    output = np.reshape(output, (dim5_1, dim5_2))
    cv2.imwrite('filaria_pred_5_5.png', output)

    output = kmeans.predict(pred6)
    output = (output/2)*255
    output = np.reshape(output, (dim6_1, dim6_2))
    cv2.imwrite('filaria_pred_5_6.png', output)

    output = kmeans.predict(pred7)
    output = (output/2)*255
    output = np.reshape(output, (dim7_1, dim7_2))
    cv2.imwrite('filaria_pred_5_7.png', output)

    output = kmeans.predict(pred8)
    output = (output/2)*255
    output = np.reshape(output, (dim8_1, dim8_2))
    cv2.imwrite('filaria_pred_5_8.png', output)

    output = kmeans.predict(pred9)
    output = (output/2)*255
    output = np.reshape(output, (dim9_1, dim9_2))
    cv2.imwrite('filaria_pred_5_9.png', output)

    output = kmeans.predict(pred10)
    output = (output/2)*255
    output = np.reshape(output, (dim10_1, dim10_2))
    cv2.imwrite('filaria_pred_5_10.png', output)

    #PLASMODIUM

    #Load the images for training
    img = cv2.imread('plasmodium/1c.JPG')
    img2 = cv2.imread('plasmodium/6c.JPG')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2Lab)
    trainer = np.vstack((img, img2))

    #Save image dimensions
    dim1 = img.shape[0]
    dim2 = img.shape[1]

    #Reshape the image from 3D to 2D
    trainer = np.reshape(trainer, (-1,3))

    #Perform KMeans (no extra cluster for black portions)
    kmeans = KMeans(n_clusters=2).fit(trainer)

    #Upscale the labels
    labs = (kmeans.labels_)*255

    #Reshape the upscaled labels from 1D to 2D
    labs = np.reshape(labs, (dim1*2,dim2))

    #Predicting part
    pred1 = cv2.imread('plasmodium/1c.JPG')
    pred2 = cv2.imread('plasmodium/3c.JPG')
    pred3 = cv2.imread('plasmodium/6c.JPG')
    pred4 = cv2.imread('plasmodium/7c.JPG')
    pred5 = cv2.imread('plasmodium/11c.JPG')
    pred6 = cv2.imread('plasmodium/19c.JPG')
    pred7 = cv2.imread('plasmodium/55c.JPG')
    pred8 = cv2.imread('plasmodium/79c.JPG')
    pred9 = cv2.imread('plasmodium/94c.JPG')
    pred10 = cv2.imread('plasmodium/105c.JPG')

    pred1 = cv2.cvtColor(pred1, cv2.COLOR_BGR2Lab)
    pred2 = cv2.cvtColor(pred2, cv2.COLOR_BGR2Lab)
    pred3 = cv2.cvtColor(pred3, cv2.COLOR_BGR2Lab)
    pred4 = cv2.cvtColor(pred4, cv2.COLOR_BGR2Lab)
    pred5 = cv2.cvtColor(pred5, cv2.COLOR_BGR2Lab)
    pred6 = cv2.cvtColor(pred6, cv2.COLOR_BGR2Lab)
    pred7 = cv2.cvtColor(pred7, cv2.COLOR_BGR2Lab)
    pred8 = cv2.cvtColor(pred8, cv2.COLOR_BGR2Lab)
    pred9 = cv2.cvtColor(pred9, cv2.COLOR_BGR2Lab)
    pred10 = cv2.cvtColor(pred10, cv2.COLOR_BGR2Lab)

    dim1_1 = pred1.shape[0]
    dim1_2 = pred1.shape[1]
    dim2_1 = pred2.shape[0]
    dim2_2 = pred2.shape[1]
    dim3_1 = pred3.shape[0]
    dim3_2 = pred3.shape[1]
    dim4_1 = pred4.shape[0]
    dim4_2 = pred4.shape[1]
    dim5_1 = pred5.shape[0]
    dim5_2 = pred5.shape[1]
    dim6_1 = pred6.shape[0]
    dim6_2 = pred6.shape[1]
    dim7_1 = pred7.shape[0]
    dim7_2 = pred7.shape[1]
    dim8_1 = pred8.shape[0]
    dim8_2 = pred8.shape[1]
    dim9_1 = pred9.shape[0]
    dim9_2 = pred9.shape[1]
    dim10_1 = pred10.shape[0]
    dim10_2 = pred10.shape[1]

    pred1 = np.reshape(pred1, (-1,3))
    pred2 = np.reshape(pred2, (-1,3))
    pred3 = np.reshape(pred3, (-1,3))
    pred4 = np.reshape(pred4, (-1,3))
    pred5 = np.reshape(pred5, (-1,3))
    pred6 = np.reshape(pred6, (-1,3))
    pred7 = np.reshape(pred7, (-1,3))
    pred8 = np.reshape(pred8, (-1,3))
    pred9 = np.reshape(pred9, (-1,3))
    pred10 = np.reshape(pred10, (-1,3))

    output = kmeans.predict(pred1)
    output = (output)*255
    output = np.reshape(output, (dim1_1, dim1_2))
    cv2.imwrite('plasmodium_pred_5_1.png', output)

    output = kmeans.predict(pred2)
    output = (output)*255
    output = np.reshape(output, (dim2_1, dim2_2))
    cv2.imwrite('plasmodium_pred_5_2.png', output)

    output = kmeans.predict(pred3)
    output = (output)*255
    output = np.reshape(output, (dim3_1, dim3_2))
    cv2.imwrite('plasmodium_pred_5_3.png', output)

    output = kmeans.predict(pred4)
    output = (output)*255
    output = np.reshape(output, (dim4_1, dim4_2))
    cv2.imwrite('plasmodium_pred_5_4.png', output)

    output = kmeans.predict(pred5)
    output = (output)*255
    output = np.reshape(output, (dim5_1, dim5_2))
    cv2.imwrite('plasmodium_pred_5_5.png', output)

    output = kmeans.predict(pred6)
    output = (output)*255
    output = np.reshape(output, (dim6_1, dim6_2))
    cv2.imwrite('plasmodium_pred_5_6.png', output)

    output = kmeans.predict(pred7)
    output = (output)*255
    output = np.reshape(output, (dim7_1, dim7_2))
    cv2.imwrite('plasmodium_pred_5_7.png', output)

    output = kmeans.predict(pred8)
    output = (output)*255
    output = np.reshape(output, (dim8_1, dim8_2))
    cv2.imwrite('plasmodium_pred_5_8.png', output)

    output = kmeans.predict(pred9)
    output = (output)*255
    output = np.reshape(output, (dim9_1, dim9_2))
    cv2.imwrite('plasmodium_pred_5_9.png', output)

    output = kmeans.predict(pred10)
    output = (output)*255
    output = np.reshape(output, (dim10_1, dim10_2))
    cv2.imwrite('plasmodium_pred_5_10.png', output)

    #SCHISTOSOMA

    #Load the images for training
    img = cv2.imread('schistosoma/schistosoma.jpg')
    img2 = cv2.imread('schistosoma/schistosoma2.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2Lab)
    trainer = np.vstack((img, img2))

    #Save image dimensions
    dim1 = img.shape[0]
    dim2 = img.shape[1]

    #Reshape the image from 3D to 2D
    trainer = np.reshape(trainer, (-1,3))

    #Perform KMeans (extra cluster for black portions)
    kmeans = KMeans(n_clusters=3).fit(trainer)

    #Upscale the labels
    labs = (kmeans.labels_/2)*255

    #Reshape the upscaled labels from 1D to 2D
    labs = np.reshape(labs, (dim1*2,dim2))

    #Predicting part
    pred1 = cv2.imread('schistosoma/schistosoma.jpg')
    pred2 = cv2.imread('schistosoma/schistosoma2.jpg')
    pred3 = cv2.imread('schistosoma/schistosoma3.jpg')
    pred4 = cv2.imread('schistosoma/schistosoma4.jpg')
    pred5 = cv2.imread('schistosoma/schistosoma5.jpg')
    pred6 = cv2.imread('schistosoma/schistosoma6.jpg')
    pred7 = cv2.imread('schistosoma/schistosoma7.jpg')
    pred8 = cv2.imread('schistosoma/schistosoma8.jpg')
    pred9 = cv2.imread('schistosoma/schistosoma9.jpg')
    pred10 = cv2.imread('schistosoma/schistosoma10.jpg')

    pred1 = cv2.cvtColor(pred1, cv2.COLOR_BGR2Lab)
    pred2 = cv2.cvtColor(pred2, cv2.COLOR_BGR2Lab)
    pred3 = cv2.cvtColor(pred3, cv2.COLOR_BGR2Lab)
    pred4 = cv2.cvtColor(pred4, cv2.COLOR_BGR2Lab)
    pred5 = cv2.cvtColor(pred5, cv2.COLOR_BGR2Lab)
    pred6 = cv2.cvtColor(pred6, cv2.COLOR_BGR2Lab)
    pred7 = cv2.cvtColor(pred7, cv2.COLOR_BGR2Lab)
    pred8 = cv2.cvtColor(pred8, cv2.COLOR_BGR2Lab)
    pred9 = cv2.cvtColor(pred9, cv2.COLOR_BGR2Lab)
    pred10 = cv2.cvtColor(pred10, cv2.COLOR_BGR2Lab)

    dim1_1 = pred1.shape[0]
    dim1_2 = pred1.shape[1]
    dim2_1 = pred2.shape[0]
    dim2_2 = pred2.shape[1]
    dim3_1 = pred3.shape[0]
    dim3_2 = pred3.shape[1]
    dim4_1 = pred4.shape[0]
    dim4_2 = pred4.shape[1]
    dim5_1 = pred5.shape[0]
    dim5_2 = pred5.shape[1]
    dim6_1 = pred6.shape[0]
    dim6_2 = pred6.shape[1]
    dim7_1 = pred7.shape[0]
    dim7_2 = pred7.shape[1]
    dim8_1 = pred8.shape[0]
    dim8_2 = pred8.shape[1]
    dim9_1 = pred9.shape[0]
    dim9_2 = pred9.shape[1]
    dim10_1 = pred10.shape[0]
    dim10_2 = pred10.shape[1]

    pred1 = np.reshape(pred1, (-1,3))
    pred2 = np.reshape(pred2, (-1,3))
    pred3 = np.reshape(pred3, (-1,3))
    pred4 = np.reshape(pred4, (-1,3))
    pred5 = np.reshape(pred5, (-1,3))
    pred6 = np.reshape(pred6, (-1,3))
    pred7 = np.reshape(pred7, (-1,3))
    pred8 = np.reshape(pred8, (-1,3))
    pred9 = np.reshape(pred9, (-1,3))
    pred10 = np.reshape(pred10, (-1,3))

    output = kmeans.predict(pred1)
    output = (output/2)*255
    output = np.reshape(output, (dim1_1, dim1_2))
    cv2.imwrite('schistosoma_pred_5_1.png', output)

    output = kmeans.predict(pred2)
    output = (output/2)*255
    output = np.reshape(output, (dim2_1, dim2_2))
    cv2.imwrite('schistosoma_pred_5_2.png', output)

    output = kmeans.predict(pred3)
    output = (output/2)*255
    output = np.reshape(output, (dim3_1, dim3_2))
    cv2.imwrite('schistosoma_pred_5_3.png', output)

    output = kmeans.predict(pred4)
    output = (output/2)*255
    output = np.reshape(output, (dim4_1, dim4_2))
    cv2.imwrite('schistosoma_pred_5_4.png', output)

    output = kmeans.predict(pred5)
    output = (output/2)*255
    output = np.reshape(output, (dim5_1, dim5_2))
    cv2.imwrite('schistosoma_pred_5_5.png', output)

    output = kmeans.predict(pred6)
    output = (output/2)*255
    output = np.reshape(output, (dim6_1, dim6_2))
    cv2.imwrite('schistosoma_pred_5_6.png', output)

    output = kmeans.predict(pred7)
    output = (output/2)*255
    output = np.reshape(output, (dim7_1, dim7_2))
    cv2.imwrite('schistosoma_pred_5_7.png', output)

    output = kmeans.predict(pred8)
    output = (output/2)*255
    output = np.reshape(output, (dim8_1, dim8_2))
    cv2.imwrite('schistosoma_pred_5_8.png', output)

    output = kmeans.predict(pred9)
    output = (output/2)*255
    output = np.reshape(output, (dim9_1, dim9_2))
    cv2.imwrite('schistosoma_pred_5_9.png', output)

    output = kmeans.predict(pred10)
    output = (output/2)*255
    output = np.reshape(output, (dim10_1, dim10_2))
    cv2.imwrite('schistosoma_pred_5_10.png', output)

else:
    print("Lmao")
