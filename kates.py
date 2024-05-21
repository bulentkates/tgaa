import cv2
import numpy as np
import os
import torch
import time 
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
print("TRAFİK GÖRÜNTÜ ANALİZ ARACI")
print("Şerit Değiştirme Algılama ve Araç Sınıflandırması (Seçenek 1):")
print("Araç Sayma (Seçenek 2 ve 3):")
print("Araç Yoğunluk Isı Haritası (Seçenek 4):")
print("Trafik İşareti ve Araç Tespiti (Seçenek 5):")
print("Hız Algılama (Seçenek 6):")
print("Şerit Çıkış Uyarı Sistemi (Seçenek 7):")
print("İnsan Kalabalık Isı Haritası (Seçenek 8):")
print("Trafikteki Yaya ve Bisikletlileri tespit etme (Seçenek 9) ")
print("Şerit Takibi Seçenek(10) ")
while 1:
    a = int(input("Seçenek..."))

    if a == 1:#Şerit değişimi araç algılama
        video_path = 'video3.mp4'
        kamera = cv2.VideoCapture(video_path)

        
        region1 = np.array([(180,0),(320,0),(210,360),(-70,360)])
        region1 = region1.reshape((-1,1,2))

        region2 = np.array([(320,0),(405,0),(470,360),(210,360)])
        region2 = region2.reshape((-1,1,2))

        region3 = np.array([(405,0),(500,0),(690,360),(470,360)])
        region3 = region3.reshape((-1,1,2))

        
        sol_bolge = set()
        orta_bolge = set()
        sag_bolge = set()

        while True:
            
            ret, frame = kamera.read()

            if not ret:
                break
                
            
            rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            
            results = model(rgb_img)

            
            detections = results.xyxy[0]  

            for det in detections:
                x1, y1, x2, y2, conf, cls = det.tolist()  

                
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2

                
                if cv2.pointPolygonTest(region1, (cx, cy), False) > 0:
                    
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                elif cv2.pointPolygonTest(region2, (cx, cy), False) > 0:
                    
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                elif cv2.pointPolygonTest(region3, (cx, cy), False) > 0:
                    
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

            
            cv2.imshow("frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        kamera.release()
        cv2.destroyAllWindows()
    elif a == 2: #arac sayisi
        video_path = 'video3.mp4'
        kamera = cv2.VideoCapture(video_path)

        
        region1 = np.array([(180,0),(320,0),(210,360),(-70,360)])
        region1 = region1.reshape((-1,1,2))

        region2 = np.array([(320,0),(405,0),(470,360),(210,360)])
        region2 = region2.reshape((-1,1,2))

        region3 = np.array([(405,0),(500,0),(690,360),(470,360)])
        region3 = region3.reshape((-1,1,2))

        
        sol_bolge = set()
        orta_bolge = set()
        sag_bolge = set()

        while True:
            
            ret, frame = kamera.read()

            if not ret:
                break
                
            
            rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            
            results = model(rgb_img)

            
            detections = results.xyxy[0]  

            for det in detections:
                x1, y1, x2, y2, conf, cls = det.tolist()  

                
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            
            cv2.imshow("frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        kamera.release()
        cv2.destroyAllWindows()
    elif a == 3: #arac sayisi 
        image_path = 'image.jpg'
        image = cv2.imread(image_path)
        rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = model(rgb_img)
        detections = results.xyxy[0]  
        nesne_sayisi = {"araba": 0, "motorsiklet": 0, "bisiklet": 0}
        for det in detections:
            x1, y1, x2, y2, conf, cls = det.tolist()  
            if cls == 2:  
                nesne_sayisi["araba"] += 1
                label = "Araba"
            elif cls == 3:  
                nesne_sayisi["motorsiklet"] += 1
                label = "Motorsiklet"
            elif cls == 4:  
                nesne_sayisi["bisiklet"] += 1
                label = "Bisiklet"
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(image, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        for nesne, sayi in nesne_sayisi.items():
            cv2.putText(image, f"{nesne.capitalize()} Sayisi: {sayi}", (10, 30 * (list(nesne_sayisi.keys()).index(nesne) + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Detection Results", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    elif a == 4 : #arac isi haritasi
        kamera = cv2.VideoCapture('video3.mp4')
        font = cv2.FONT_HERSHEY_SIMPLEX

        while True:
            ret, kare = kamera.read()
            if not ret:
                break

            imgs = cv2.cvtColor(kare, cv2.COLOR_BGR2RGB)
            results = model(imgs)

            
            points = []

            
            for i in range(len(results.xyxy[0])):
                x1, y1, x2, y2, score, label = results.xyxy[0][i].tolist()
                x1, y1, x2, y2, score, label = int(x1), int(y1), int(x2), int(y2), float(score), int(label)
                name = results.names[label]

                if score < 0.1:
                    continue

                if name != 'car':
                    continue

                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                points.append([center_x, center_y])

            
            if len(points) > 0:
                points = np.array(points)
                heatmap, xedges, yedges = np.histogram2d(points[:, 1], points[:, 0], bins=15, range=[[0, kare.shape[0]], [0, kare.shape[1]]])
                heatmap = cv2.resize(heatmap, (kare.shape[1], kare.shape[0]))
                heatmap = np.uint8(255 * heatmap)
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                result = cv2.addWeighted(kare, 0.6, heatmap, 0.4, 0)
                cv2.imshow('Isı Haritası', result)

            cv2.imshow("kamera", kare)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        kamera.release()
        cv2.destroyAllWindows()

    elif a == 5: #işaret ve araba tanıma
        kamera = cv2.VideoCapture('a.mp4')
        
        font = cv2.FONT_HERSHEY_SIMPLEX

        
        target_classes = ['stop sign', 'traffic light', 'car', 'motorcycle']

        while True:
            
            ret, kare = kamera.read()
            if not ret:
                break

            
            imgs = cv2.cvtColor(kare, cv2.COLOR_BGR2RGB)
            
            results = model(imgs)

            
            for i in range(len(results.xyxy[0])):
                
                x1, y1, x2, y2, score, label = results.xyxy[0][i].tolist()
                x1, y1, x2, y2, score, label = int(x1), int(y1), int(x2), int(y2), float(score), int(label)
                name = results.names[label]

                
                if score < 0.1:
                    continue

                
                if name not in target_classes:
                    continue

                
                cv2.rectangle(kare, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                cv2.putText(kare, f'{name} {score:.2f}', (x1, y1 - 10), font, 0.5, (0, 255, 0), 2)

            
            cv2.imshow("kamera", kare)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        
        kamera.release()
        cv2.destroyAllWindows()
    elif a == 6: #hiz algılama
        cap = cv2.VideoCapture('video4.mp4')
        car_cascade = cv2.CascadeClassifier('cars.xml')
        fps = cap.get(cv2.CAP_PROP_FPS)
        prev_frame_time = 0
        new_frame_time = 0
        while cap.isOpened():
            ret, frame = cap.read()
            
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cars = car_cascade.detectMultiScale(gray, 1.1, 1)
            for (x, y, w, h) in cars:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                new_frame_time = time.time()
                elapsed_time = new_frame_time - prev_frame_time
                prev_frame_time = new_frame_time
                car_width = w
                speed = car_width / elapsed_time
                
                cv2.putText(frame, f'Speed: {speed:.2f} units/s', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

            cv2.imshow('araba ve hiz', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
    elif a == 7 : #kendi serit uyarı
        cap = cv2.VideoCapture(0)  
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([30, 255, 255])

        while True:
            
            ret, frame = cap.read()
            if not ret:
                break

            
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            
            mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

            
            res = cv2.bitwise_and(frame, frame, mask=mask)

            
            edges = cv2.Canny(mask, 75, 150)

            
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        
            if contours:
                
                largest_contour = max(contours, key=cv2.contourArea)
                
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    
                    cv2.circle(frame, (cX, cY), 7, (255, 255, 255), -1)
                    cv2.putText(frame, "Merkez", (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                
                if cX < 100 or cX > 540:  
                    cv2.putText(frame, " SERİT DİSİ !", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            
            cv2.imshow("Frame", frame)

            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
    elif a == 8: # insan kalabalık ısı haritası
        kamera = cv2.VideoCapture('video.mp4')
        font = cv2.FONT_HERSHEY_SIMPLEX

        while True:
            ret, kare = kamera.read()
            if not ret:
                break

            imgs = cv2.cvtColor(kare, cv2.COLOR_BGR2RGB)
            results = model(imgs)

            
            points = []

            
            for i in range(len(results.xyxy[0])):
                x1, y1, x2, y2, score, label = results.xyxy[0][i].tolist()
                x1, y1, x2, y2, score, label = int(x1), int(y1), int(x2), int(y2), float(score), int(label)
                name = results.names[label]

                if score < 0.1:
                    continue

                if name != 'person':
                    continue

                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                points.append([center_x, center_y])

            
            if len(points) > 0:
                points = np.array(points)
                heatmap, xedges, yedges = np.histogram2d(points[:, 1], points[:, 0], bins=15, range=[[0, kare.shape[0]], [0, kare.shape[1]]])
                heatmap = cv2.resize(heatmap, (kare.shape[1], kare.shape[0]))
                heatmap = np.uint8(255 * heatmap)
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                result = cv2.addWeighted(kare, 0.6, heatmap, 0.4, 0)
                cv2.imshow('Isı Haritası', result)

            cv2.imshow("kamera", kare)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        kamera.release()
        cv2.destroyAllWindows()

    elif a == 9 : #spesifik nesneler
        kamera = cv2.VideoCapture('a.mp4')
        
        font = cv2.FONT_HERSHEY_SIMPLEX

        
        target_classes = ['person', 'bicycle','cat', 'dog']

        while True:
            
            ret, kare = kamera.read()
            if not ret:
                break

            
            imgs = cv2.cvtColor(kare, cv2.COLOR_BGR2RGB)
            
            results = model(imgs)

            
            for i in range(len(results.xyxy[0])):
                
                x1, y1, x2, y2, score, label = results.xyxy[0][i].tolist()
                x1, y1, x2, y2, score, label = int(x1), int(y1), int(x2), int(y2), float(score), int(label)
                name = results.names[label]

                
                if score < 0.1:
                    continue

                
                if name not in target_classes:
                    continue

                
                cv2.rectangle(kare, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                cv2.putText(kare, f'{name} {score:.2f}', (x1, y1 - 10), font, 0.5, (0, 255, 0), 2)

            
            cv2.imshow("kamera", kare)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        
        kamera.release()
        cv2.destroyAllWindows()
    elif a == 10: #yol-şerit
        def region_of_interest(img, vertices):
            mask = np.zeros_like(img)
            cv2.fillPoly(mask, vertices, 255)
            masked_image = cv2.bitwise_and(img, mask)
            return masked_image

        def draw_the_lines(img, lines):
            img = np.copy(img)
            blank_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

            for line in lines:
                for x1, y1, x2, y2 in line:
                    cv2.line(blank_image, (x1, y1), (x2, y2), (0, 255, 0), thickness=10)

            img = cv2.addWeighted(img, 0.8, blank_image, 1, 0.0)
            return img

        def process(image):
            height = image.shape[0]
            width = image.shape[1]
            region_of_interest_vertices = [
                (0, height),
                (width/2, height/2),
                (width, height)
            ]
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            canny_image = cv2.Canny(gray_image, 100, 120)
            cropped_image = region_of_interest(canny_image,
                            np.array([region_of_interest_vertices], np.int32),)
            lines = cv2.HoughLinesP(cropped_image,
                                    rho=2,
                                    theta=np.pi/180,
                                    threshold=50,
                                    lines=np.array([]),
                                    minLineLength=40,
                                    maxLineGap=100)
            if lines is not None:
                image_with_lines = draw_the_lines(image, lines)
                return image_with_lines
            else:
                return image

        cap = cv2.VideoCapture('tt.mp4')

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = process(frame)
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()