# Data Generator for Classification Case

with logical equation in background

Use Case: Interview


number_of _rows = 100

- personal score: 3 (Score range: 1-5)
- academic degree (0: no academic, 1:bachelor, 2:master, 3:phd)
- Technical Score Rage: 0-5
  - [3] python: 4
  - [3] sql: 4
  - [1] nosql: 3
  - [5] functionaloriented: 4
  - [3] objectoriented: 3
  
- Technical score (calc) : 3*4 + 3*4 + 1*3 + 5*4 + 3*3 
- Summe Scoring: personal + academic degree + technical 
- 
- Classes:
  - < 20 : absagen
  - < 30 : HR Senior Review
  - < 40 : Fachabteilung Review
  - >= 40 : Phase 2 Interview Termin einstellen



Write a PROGRAM (Python) to generate the csv file with certain number of rows . The fields and the scoring should be defined in a separete JSON file.

Use the generated CSV File first of all to train your model. After finishing the training of the model. Use the model to predict a NEW csv file will following data: 
- Personal Score
- Academic Degree
- Technical Score

And predict the required class 


Individual Work ONLY




