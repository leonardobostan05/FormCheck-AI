import numpy as np

# berechnet den Winkel am Punkt b, also am Scheitelpunkt
# das ist später wichtig um zum Beispiel den Winkel von der Beugung im Ellenbogen zu messen

def calculate_angles(a, b, c):

    # a, b und c sind jeweils die x und y Koordinaten

    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    # Vektoren vom Scheitelpunkt aus 
    
    vector_ab = a - b
    vector_cb = c - b

    # Skalarprodukt und Vektorlängen

    dot = np.dot(vector_ab, vector_cb)
    norm_ab = np.linalg.norm(vector_ab)
    norm_cb = np.linalg.norm(vector_cb)

    #Skalarprodukt durch das Produkt beider Vektorlänger, return: Wert zwischen -1 und 1 -> für Wertebereich für Cosinus 

    cos_angle = dot / (norm_ab * norm_cb)

    #Auf-/Abrunden von Werten für Stabilität der Endwerte (Sicherheitsnetz)
    cos_angle = np.clip(cos_angle, -1.0, 1.0) 

    #Den Cosinuswert mit arccos in Radiant Wert umrechnen, und diesen dann mithilfe von .degrees in Winkel Wert
    angle = np.degrees(np.arccos(cos_angle))

    return angle



# Diese Methode soll alle Winkel aller Videos pro Frame berechnen und zurückgeben

def extract_angles(video_array):

    # doppelt verkettete Schleife (äußere Schleife == Videos ; innere Schleife == Frames)
    # initialisieren der beiden arrays für jeweils die frames pro video & array für videos selbst

    
    alle_videos = []
    for i in range(video_array.shape[0]):
        pro_video = []
        for j in range(video_array.shape[1]):

            # -------------------------------
            # Position der Körperteile pro Frame
            
            # Schulter (Landmark L=11 & R=12)

            xSchulterLinks = video_array[i][j][22]
            ySchulterLinks = video_array[i][j][23]
            xSchulterRechts = video_array[i][j][24]
            ySchulterRechts = video_array[i][j][25]

            SchulterLinks = [xSchulterLinks, ySchulterLinks]
            SchulterRechts = [xSchulterRechts, ySchulterRechts]
            

            # Ellenbogen (Landmark L=13 & R=14)
            
            xEllenbogenLinks = video_array[i][j][26]
            yEllenbogenLinks = video_array[i][j][27]
            xEllenbogenRechts = video_array[i][j][28]
            yEllenbogenRechts = video_array[i][j][29]

            EllenbogenLinks = [xEllenbogenLinks, yEllenbogenLinks]
            EllenbogenRechts = [xEllenbogenRechts, yEllenbogenRechts]

            # Handgelenk (Landmark L=15 & R=16)
            
            xHandgelenkLinks = video_array[i][j][30]
            yHandgelenkLinks = video_array[i][j][31]
            xHandgelenkRechts = video_array[i][j][32]
            yHandgelenkRechts = video_array[i][j][33]

            HandgelenkLinks = [xHandgelenkLinks, yHandgelenkLinks]
            HandgelenkRechts = [xHandgelenkRechts, yHandgelenkRechts]

             # Hüfte (Landmark L=23 & R=24)
            
            xHuefteLinks = video_array[i][j][46]
            yHuefteLinks = video_array[i][j][47]
            xHuefteRechts = video_array[i][j][48]
            yHuefteRechts = video_array[i][j][49]

            HuefteLinks = [xHuefteLinks, yHuefteLinks]
            HuefteRechts = [xHuefteRechts, yHuefteRechts]

            # Knie (Landmark L=25 & R=26)
            
            xKnieLinks = video_array[i][j][50]
            yKnieLinks = video_array[i][j][51]
            xKnieRechts = video_array[i][j][52]
            yKnieRechts = video_array[i][j][53]

            KnieLinks = [xKnieLinks, yKnieLinks]
            KnieRechts = [xKnieRechts, yKnieRechts]

            # Knöchel (Landmark L=27 & R=28)
            
            xKnoechelLinks = video_array[i][j][54]
            yKnoechelLinks = video_array[i][j][55]
            xKnoechelRechts = video_array[i][j][56]
            yKnoechelRechts = video_array[i][j][57]

            KnoechelLinks = [xKnoechelLinks, yKnoechelLinks]
            KnoechelRechts = [xKnoechelRechts, yKnoechelRechts]

            # Winkel Ellenbogen (Schulter, Ellenbogen & Handgelenk)

            WinkelEllenbogenLinks = calculate_angles(SchulterLinks, EllenbogenLinks, HandgelenkLinks)
            WinkelEllenbogenRechts = calculate_angles(SchulterRechts, EllenbogenRechts, HandgelenkRechts)

             # Winkel Rücken (Hüfte, Knie, Knöchel)

            WinkelRueckenLinks = calculate_angles(HuefteLinks, KnieLinks, KnoechelLinks)
            WinkelRueckenRechts = calculate_angles(HuefteRechts, KnieRechts, KnoechelRechts)

            #NEU: Schulter Hüfte Knie Winkel implementieren

            WinkelSHKLinks = calculate_angles(SchulterLinks, HuefteLinks, KnieLinks)
            WinkelSHKRechts = calculate_angles(SchulterRechts, HuefteRechts, KnieRechts)

            # Durchschnitt für die jeweiligen Seite der Arme / Beine, weil asynchrone Ausführung der Übung ebenfalls möglich
            
            WinkelEllenbogenAvg = (WinkelEllenbogenLinks + WinkelEllenbogenRechts) / 2
            WinkelRueckenAvg = (WinkelRueckenLinks + WinkelRueckenRechts) / 2
            WinkelSHKAvg = (WinkelSHKLinks + WinkelSHKRechts) / 2


            #Anhängen der Winkel an Array pro einzelnes Video
            
            pro_video.append([WinkelEllenbogenAvg, WinkelRueckenAvg, WinkelSHKAvg])

        alle_videos.append(pro_video)

    return alle_videos

#-------
# Methode um die Features zu reduzieren, sodass scikit damit arbeiten kann

def reduce_features(features_array):

    #Verwandeln in ein numpy array
    npArr = np.array(features_array)

    # durchschnitt, min & max der Winkel des gegebenen featurearrays
    #axis = 1 -> bezieht sich auf Frames

    #der Knackpunkt ist hier: hier behandeln wir das Padding-Problem
    durchschnittArr = np.nanmean(npArr, axis = 1)
    maxArr = np.nanmax(npArr, axis = 1)
    minArr = np.nanmin(npArr, axis = 1)
    #zusätzlich: Standardabweichung hinzufügen
    stdArr = np.nanstd(npArr, axis=1)

    #zusammenfügen zu einem Array
    npArrCombined = np.concatenate([durchschnittArr, maxArr, minArr, stdArr], axis=1)

    return npArrCombined   

#------