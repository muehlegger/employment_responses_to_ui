# employment_responses_to_ui
Code used for the numerical analysis in my master thesis: "Employment Responses to Unemployment Benefits"
--------------------------------------------------------------------------------------
ENGLISH:

To run the calculations, python3 has to be installed on the system. 
It is available at: https://www.python.org/downloads/

In addition to Python, the following packages have to be installed (command to install the module using pip is provided subsequently):

- Numpy
	> pip install numpy
- Pandas
	> pip install pandas
- Matplotlib
	> pip install matplotlib
- Numba
	> pip install numba
- Scipy
	> pip install scipy
- Statsmodels
	> pip install statsmodels
- Fredapi
	> pip install fredapi

In the program, the path to the folder where figures and tables are stored have to be provided.
Enter the complete path in the code in line 39 (complete URL with "/" instead of "\").

In lines 28, 33, and 36, the type of calibration that is performed can be adjusted.

- 	To use the baseline parameters (from table 1) specify in line 28: "calibrated = True"

-	To run the complete calibration, adjust line 28 to: "calibrated = False" and line 33 to: "default_calibration = True"

-	To manually set a value for parameter "l" in line 36, additionally adjust line 28 to: "calibrated = False" and line 33 to: "default_calibration = False".

To then run the program, use a console, navigate to the folder with the file containing the code, and run the following command:

	> python Muehlegger_Numerical_Analysis.py.py

Calculation times are:

- 	using the baseline parameters:	about 30 min
-	running the full calibration:	about 20 h

--------------------------------------------------------------------------------------------------------
DEUTSCH:

Um den Programmcode auszuführen muss Python3 auf dem System installiert sein.
Dies ist verfügbar unter: https://www.python.org/downloads/
Zusätzlich müssen folgende Pakete installiert sein (der Befehl um das Paket über die Konsole zu installieren ist nachfolgend angegeben):

- Numpy
	> pip install numpy
- Pandas
	> pip install pandas
- Matplotlib
	> pip install matplotlib
- Numba
	> pip install numba
- Scipy
	> pip install scipy
- Statsmodels
	> pip install statsmodels
- Fredapi
	> pip install fredapi

Im Programmcode (mit geeignetem Editor öffnen) muss noch der Dateipfad angegeben werden wo die Grafiken und Tabellen gepeichert werden sollen.
Hierzu in Zeile 39 im Code die gesamte URL zum gewünschten Ordner mit "/" antatt "\" angeben.

In Zeile 28, 33, und 36 kann angegeben werden ob die Simulation 
-	mit dem bereits kalibrierten Modell (Parameter aus Tabelle 1) durchgeführt werden soll 
	(Zeile 28: calibrated = True), 

-	oder alternativ das Modell noch einmal komplett kalibriert werden soll 
	(Zeile 28: calibrated = False, UND Zeile 33: default_calibration = True), 

-	oder ein manueller Wert für den Parameter "l" in Zeile 36 angegeben werden soll und alle anderen parameter neu calibriert werden sollen 
	(Zeile 28: calibrated = False, UND Zeile 33: default_calibration = False).

Um das Programm zu starten muss dann mithilfe der Konsole in den Ordner navigiert werden wo sich die Programmdatei befindet, und dann in die Konsole folgender Befehl ausgeführt werden:

	> python Muehlegger_Numerical_Analysis.py

Berechnungszeiten sind:

- Default Parameter: 		ca. 30 min
- komplette Kalibrierung:	ca. 20 h
