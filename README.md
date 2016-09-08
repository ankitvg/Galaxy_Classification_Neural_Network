# Galaxy_Classification_Neural_Network
This is a pure Tensorflow implemenation of the network described in the paper: A CATALOG OF VISUAL-LIKE MORPHOLOGIES IN THE 5 CANDELS FIELDS USING DEEP-LEARNING 

Paper URL: http://arxiv.org/abs/1509.05429v1 

A couple of notes: 

1. main.py is the starting point for executing the model (usage: python main.py) 

2. You need to populate the directory data/jpeg_redundant_f160 with the f160 images 

3. the f160 images have the dimensions 454x454x3, however in the paper it says that images need to be 45x45x3 so they are resized in datahelper.py 

4. Also, there isn't a standard max-out layer in tensorflow, which is what the paper reccomends for the fully connected layers, so I use standard ReLU. This should be changed in the future sometime. 

5. After training, saved models are placed in the ./models directory and reporting along the way is stored in ./report


P.S.: The paper and presentation we made for this project is also uploaded. Check it out :)

