'''Ok Claude hier eine schwierigere aufgabe. 
Ich habe diesen code welcher ein Interleavetes mp3 einliest und deinterleaven soll. 
jetzt habe ich folgende punkte die angepasst werden sollen. 

1. Der code soll echtzeitfähig werden. d.h. er soll ein signal durch den aux anschluss einlesen und über die lautsprecher des PCs gefiltert nur eines der erkannten signale ausgeben. dabei gillt Folgende bedingung für das eingangssignal : Es ist nicht bekannt wie viele originalsignale im eingangssignal vorhanden sind, es ist gegeben das die segmente des interleavten 10 - 50 ms lang sein können, es können leere segmente kommen bei denen kein signal oder ein extrem kleines anliegen diese sollen ignoriert werden. 





 Der Code soll eine erkennung bekommen wann ein signal beginnt. also die ganze bearbeitung soll erst starten wenn erkannt wird das ein signal anliegt. denn es ist nicht bekannt wann das signal startet es muss nocht direkt zum start des Programms anliegen. 





 Ich habe gesagt Echtzeitfähig das heist nicht in der selben ms, damit der code zeit hat die signale zu erkennen, zu sortieren und zu bearbeiten hat er einen puffer von 600 ms. 
 '''