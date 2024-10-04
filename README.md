![hacktoberfest](https://img.shields.io/badge/hacktoberfest-green?style=flat)
# Caption-Craft
This project is aimed to caption the images which is passed to this model.


`v1.0`

<hr>

`Using Colab`
```
!git clone https://github.com/HackSlashNITP/Caption-Craft
%cd Caption-Craft
!pip install -r requirements.txt
!python main.py "imagepath"
```

<hr>

`Using local environment`
```
git clone https://github.com/HackSlashNITP/Caption-Craft
cd Caption-Craft
pip install -r requirements.txt
python flask.py
```

<hr>


### Multi-Image Captioning

`Run on Local System`

* `git clone https://github.com/HackSlashNITP/Caption-Craft`
* `cd Caption-Craft`
* Make a folder named `content`
* Make another folder `Caption` as `content/Caption`
* Save your images inside the `Caption` folder
  
File Structure(before run):

- Caption-Craft
  - content
    - Caption
      - 1.jpg
      - 2.jpg
      - 3.png
      - 4.png
      - 5.jpeg
  - multicaption.py

* `python multicaption.py` , run this code in `Caption-Craft` directory

File Structure(after run):

- Caption-Craft
  - content
    - Caption
      - 1.jpg
      - 2.jpg
      - 3.png
      - 4.png
      - 5.jpeg
     - image_captions.csv
  - multicaption.py

A `.csv` file will be formed which has `Sl_No`, `Image_name` and `Image_caption`, and a 3x3 matrix will be displayed which has at most nine images randomly selected from the folder `Caption-Craft/content/Caption` and displayed with their caption as labels.

`To run on Google Colab`

* copy the code in `multicaption.py` and run it.
* Save your images inside the `Caption` folder at the base dir, which is `/content` by default

#### Code for reference: 
```python
folder_path = "/content/Caption"
ImageMatrix('/content/Caption', '/content/image_captions.csv')
```
