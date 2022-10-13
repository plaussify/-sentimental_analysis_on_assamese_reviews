# Sentimental analysis of Assamese song reviews
To use the app (hot loading), visit : https://assamese-sentiment-analyzer.herokuapp.com/

## Dependencies:

1. Python 3.6^: [Refer to install on linux](https://docs.python.org/3/using/unix.html) || [Refer to install on Windows](https://docs.python.org/3/using/windows.html)
2. Pip
3. Streamlit
4. dotenv
5. Numpy, Pandas
6. NLTK
7. Pickle
8. Scikit learn

## Approach:

We have used two approaches to classify the sentiment of Assamese reviews as positive or Negative.

1. IN language Semantic Analysis. : This approach is based on training the classifiers on the same language as text.
2. Machine Translation Based Semantic Analysis. : In this approach we train the classifier on English reviews and for testing, we translate the Assamese reviews into English using [Microsoft Translator api](https://docs.microsoft.com/en-us/azure/cognitive-services/translator/quickstart-translator?tabs=python#translate-text) and then we classify the Sentiment of the review.

## Dataset Used:

We have used a total of 1028 Assamese songs reviews for the Sentiment Analysis. We have manually collected all of the reviews from Youtube comments and labeled them as 1 (for positive comment) and 0 (for negative comment). Out of 1028 reviews collected manually, 505 reviews are positive and the rest 523 are negative reviews. For Machine Translation based approach, we also need english reviews, by using [Microsoft Azure Cognitive Services(Translator)](https://docs.microsoft.com/en-us/azure/cognitive-services/translator/quickstart-translator?tabs=python#translate-text) the native language reviews are translated to relative English reviews.

## Files Description:

[in_language.ipynb](https://github.com/Reckon77/sentimental_analysis_on_assamese_reviews/blob/main/in_language.ipynb) : This module is used to do In-language classification. We have used TF-IDF for data vectorization.

[machine_translate.ipynb](https://github.com/Reckon77/sentimental_analysis_on_assamese_reviews/blob/main/machine_translate.ipynb) : This module is used to do Machine Translation Based Semantic analysis. We have used Count Vectorizer for data vectorization.

[app.py](https://github.com/Reckon77/sentimental_analysis_on_assamese_reviews/blob/main/app.py) : This module contains the streamlit app scripts to handle the views and model.

[reviews.csv](https://github.com/Reckon77/sentimental_analysis_on_assamese_reviews/blob/main/reviews.csv) : Native language dataset

[translated.csv](https://github.com/Reckon77/sentimental_analysis_on_assamese_reviews/blob/main/translated.csv) : Translated version of Assamese dataset using Microsoft translator.

[model_inl.pkl](https://github.com/Reckon77/sentimental_analysis_on_assamese_reviews/blob/main/model_inl.pkl) : Extra trees classifier model for in language sentiment analysis.

[model_mt.pkl](https://github.com/Reckon77/sentimental_analysis_on_assamese_reviews/blob/main/model_mt.pkl) : Logistic regression classifier model for machine translation based language sentiment analysis.

[vectorizer_inl.pkl](https://github.com/Reckon77/sentimental_analysis_on_assamese_reviews/blob/main/vectorizer_inl.pkl) : TFIDF vectorizer model for Assamese language data vectorization.

[vectorizer_mt.pkl](https://github.com/Reckon77/sentimental_analysis_on_assamese_reviews/blob/main/vectorizer_mt.pkl) : Count vectorizer model for English language data vectorization.

[nltk.txt](https://github.com/Reckon77/sentimental_analysis_on_assamese_reviews/blob/main/nltk.txt), [Procfile](https://github.com/Reckon77/sentimental_analysis_on_assamese_reviews/blob/main/Procfile), [requirements.txt](https://github.com/Reckon77/sentimental_analysis_on_assamese_reviews/blob/main/requirements.txt), [setup.sh](https://github.com/Reckon77/sentimental_analysis_on_assamese_reviews/blob/main/setup.sh) : Deployement dependencies.

## Setting Up The Project locally

## Installation

There are several ways to run your own instance of the project depending on your OS:

### Windows:

- Install Streamlit on Windows, Streamlit’s officially-supported environment manager on Windows is [Anaconda Navigator.](https://docs.anaconda.com/anaconda/navigator/)
  - Install Anaconda, if you don’t have Anaconda install yet, follow the steps provided on the [Anaconda installation page.](https://docs.anaconda.com/anaconda/install/windows/)
  - Create a new environment with Streamlit:
  - Next you’ll need to set up your environment:
  1.  Follow the steps provided by Anaconda to set up and manage your environment using the Anaconda Navigator.
  2.  Select the “▶” icon next to your new environment. Then select “Open terminal”:
      ![](https://i.stack.imgur.com/EiiFc.png)
  3.  In the terminal that appears, type:
      `pip install streamlit`
  4.  Test that the installation worked:
      `streamlit hello`
  5.  Streamlit’s Hello app should appear in a new tab in your web browser!
  6.  Run `pip install nltk`
  7.  Run `pip install scikit-learn`(depends)
  8.  Run ` pip install python-dotenv`
  9.  Get your API key and location to use the Microsoft translation API ( refer [here](https://azure.microsoft.com/en-in/services/cognitive-services/translator/))
  10. Create a .env file and write this code (replace "yourAPIkey" and "yourLocation" with the one that you obtained in step ix.)
  ```
  API_KEY=yourAPIkey
  LOCATION=yourLocation
  ```
  11. Finally `streamlit run app.py`

### Linux / macOS:

- Install pip.

  - On a macOS: `sudo easy_install pip` On Ubuntu with Python 3: `sudo apt-get install python3-pip`
  - Install pipenv. `pip3 install pipenv`
  - Create a new virtual environment for python:
  - Next you’ll need to set up your environment:

  1.  the recommended way to create a virtual environment is to use the venv module.
      `sudo apt install python3-venv`
  2.  Switch to the directory where you would like to store your Python 3 virtual environments. Within the directory run the following command to create your new virtual environment: `python3 -m venv my-project-env`
  3.  To start using this virtual environment, you need to activate it by running the activate script: `source my-project-env/bin/activate`
  4.  Once activated, the virtual environment’s bin directory will be added at the beginning of the $PATH variable. Also your shell’s prompt will change and it will show the name of the virtual environment you’re currently using. In our case that is my-project-env:

      ```
      $ source my-project-env/bin/activate
      (my-project-env) $

      ```

  5.  Now that the virtual environment is activated, we can start installing, upgrading, and removing packages using pip.
  6.  In the terminal that appears, type:
      `pip install streamlit`
  7.  Test that the installation worked:
      `streamlit hello`
  8.  Streamlit’s Hello app should appear in a new tab in your web browser!
  9.  Run `pip install nltk`
  10. Run `pip install scikit-learn`(depends)
  11. Run ` pip install python-dotenv`
  12. Get your API key and location to use the Microsoft translation API ( refer [here](https://azure.microsoft.com/en-in/services/cognitive-services/translator/))
  13. Create a .env file and write this code (replace "yourAPIkey" and "yourLocation" with the one that you obtained in step xii.)

  ```
  API_KEY=yourAPIkey
  LOCATION=yourLocation
  ```

  14. Finally `streamlit run app.py`

  ## Releases and Assets

  [ v1.0-beta](https://github.com/Reckon77/sentimental_analysis_on_assamese_reviews/releases/tag/v1.0-beta) : First release including only in-language classification. Used TFIDF as a feature.
  ## Authors

- [Reckon Mazumder](https://github.com/Reckon77)
- [Akash Chetia](https://github.com/AkashChetia)
- [Kunjal Sarma](https://github.com/KunjalSarma)
- [Srimanjyoti Dutta](https://github.com/plaussify)
- [Samarjit Sharma](https://github.com/UntrainedAnimal)

 
## Screenshots

![screenshot1](https://user-images.githubusercontent.com/62415937/127906429-ba83d649-1cc2-47d1-b73c-7af11788e195.jpeg
)

![screenshot2](https://user-images.githubusercontent.com/62415937/127906434-1aa3d6fc-68a8-4c33-8461-3c6b299d3023.jpeg)
![screenshot3](https://user-images.githubusercontent.com/62415937/127906435-53cb75fc-bf75-445c-838c-a907fbe6739a.jpeg)
![screenshot4](https://user-images.githubusercontent.com/62415937/127906437-28041879-f6b2-4ac1-b826-ab1f9c6e670d.jpeg)
![screenshot5](https://user-images.githubusercontent.com/62415937/127906439-1159a641-fc4e-417d-accf-8594788d2091.jpeg)
![screenshot6](https://user-images.githubusercontent.com/62415937/127906442-01f05539-2ff8-4720-9906-39e97df5321d.jpeg)

